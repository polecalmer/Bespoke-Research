"""Walk-forward backtest: flat-stake betting on test set where model diverges from market.

For each match in the test set (2023-24, 2024-25):
- Compare FPL+Odds model probability vs market implied probability
- If divergence > threshold, place a flat bet
- Track ROI, max drawdown, Sharpe ratio, statistical significance

Outputs: data/processed/backtest.parquet, summary stats
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATA_PROCESSED, MODEL_DIR, save_parquet

THRESHOLDS = [0.03, 0.05, 0.07, 0.10]
OUTCOMES = {'H': 2, 'D': 1, 'A': 0}


def load_test_predictions() -> pd.DataFrame:
    """Load test set with model predictions."""
    return pd.read_parquet(MODEL_DIR / 'test_predictions.parquet')


def run_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """Run divergence-based betting simulation across all thresholds."""
    all_bets = []

    for _, row in df.iterrows():
        for outcome, outcome_label in [('H', 'Home'), ('D', 'Draw'), ('A', 'Away')]:
            p_model = row[f'p_fpl_odds_{outcome}']
            p_market = row[f'implied_prob_{outcome.lower()}']
            odds_col = {'H': 'B365H', 'D': 'B365D', 'A': 'B365A'}[outcome]
            decimal_odds = row[odds_col]

            if pd.isna(p_model) or pd.isna(p_market) or pd.isna(decimal_odds):
                continue

            divergence = p_model - p_market

            for threshold in THRESHOLDS:
                if divergence > threshold:
                    actual = row['result']
                    won = actual == outcome
                    profit = (decimal_odds - 1) if won else -1.0

                    all_bets.append({
                        'season': row['season'],
                        'gw': row['gw'],
                        'team_h_name': row['team_h_name'],
                        'team_a_name': row['team_a_name'],
                        'bet_outcome': outcome,
                        'threshold': threshold,
                        'decimal_odds': decimal_odds,
                        'actual_result': actual,
                        'won': won,
                        'profit': profit,
                        'p_model': p_model,
                        'p_market': p_market,
                        'divergence': divergence,
                    })

    return pd.DataFrame(all_bets)


def compute_stats(bets_df: pd.DataFrame) -> list:
    """Compute summary stats per threshold."""
    results = []

    for threshold in THRESHOLDS:
        subset = bets_df[bets_df['threshold'] == threshold]
        if len(subset) == 0:
            results.append({
                'threshold': threshold, 'n_bets': 0, 'wins': 0,
                'hit_pct': 0, 'roi_pct': 0, 'max_drawdown': 0,
                'sharpe': 0, 'p_value': 1.0, 'ci_lower': 0, 'ci_upper': 0,
            })
            continue

        n_bets = len(subset)
        wins = subset['won'].sum()
        hit_pct = 100.0 * wins / n_bets
        total_profit = subset['profit'].sum()
        roi_pct = 100.0 * total_profit / n_bets

        # Max drawdown
        cumulative = np.cumsum(subset['profit'].values)
        peak = np.maximum.accumulate(cumulative)
        drawdown = peak - cumulative
        max_dd = drawdown.max() if len(drawdown) > 0 else 0

        # Sharpe ratio (annualized, approximate)
        returns = subset['profit'].values
        sharpe = (np.mean(returns) / (np.std(returns) + 1e-8)) * np.sqrt(len(returns))

        # Statistical significance: binomial test
        avg_implied = subset['p_market'].mean()
        try:
            pval = binomtest(int(wins), n_bets, avg_implied, alternative='greater').pvalue
        except Exception:
            pval = 1.0

        # Bootstrap 95% CI on ROI
        rng = np.random.RandomState(42)
        bootstrap_rois = []
        for _ in range(10000):
            sample = rng.choice(returns, size=n_bets, replace=True)
            bootstrap_rois.append(np.mean(sample))
        ci_lower = 100.0 * np.percentile(bootstrap_rois, 2.5)
        ci_upper = 100.0 * np.percentile(bootstrap_rois, 97.5)

        results.append({
            'threshold': threshold,
            'n_bets': n_bets,
            'wins': int(wins),
            'hit_pct': round(hit_pct, 1),
            'roi_pct': round(roi_pct, 1),
            'max_drawdown': round(max_dd, 1),
            'sharpe': round(sharpe, 2),
            'p_value': round(pval, 4),
            'ci_lower': round(ci_lower, 1),
            'ci_upper': round(ci_upper, 1),
        })

    return results


def main():
    print("=" * 70)
    print("BACKTEST: Divergence-Based Betting Simulation")
    print("=" * 70)

    df = load_test_predictions()
    print(f"Test set: {len(df)} matches")

    # Fix column name inconsistency (implied_prob columns are lowercase h/d/a)
    # Model predictions use uppercase
    for outcome in ['h', 'd', 'a']:
        col_lower = f'implied_prob_{outcome}'
        if col_lower not in df.columns:
            col_upper = f'implied_prob_{outcome.upper()}'
            if col_upper in df.columns:
                df[col_lower] = df[col_upper]

    bets_df = run_backtest(df)
    print(f"Total bets placed: {len(bets_df)}")

    if len(bets_df) == 0:
        print("No bets qualified at any threshold. Model doesn't diverge from market enough.")
        return

    # Summary
    stats = compute_stats(bets_df)

    print(f"\n{'Threshold':<12} {'Bets':<8} {'Wins':<8} {'Hit%':<8} {'ROI%':<10} "
          f"{'MaxDD':<8} {'Sharpe':<8} {'p-val':<8} {'95% CI'}")
    print("-" * 90)
    for s in stats:
        print(f"{s['threshold']:<12.2f} {s['n_bets']:<8} {s['wins']:<8} {s['hit_pct']:<8.1f} "
              f"{s['roi_pct']:<10.1f} {s['max_drawdown']:<8.1f} {s['sharpe']:<8.2f} "
              f"{s['p_value']:<8.4f} [{s['ci_lower']:.1f}%, {s['ci_upper']:.1f}%]")

    # Breakdown by season
    print("\nPer-season breakdown (threshold=0.05):")
    t05 = bets_df[bets_df['threshold'] == 0.05]
    if len(t05) > 0:
        for season in sorted(t05['season'].unique()):
            ss = t05[t05['season'] == season]
            roi = 100.0 * ss['profit'].sum() / len(ss) if len(ss) > 0 else 0
            print(f"  {season}: {len(ss)} bets, {ss['won'].sum()} wins, ROI={roi:.1f}%")

    # Breakdown by outcome type
    print("\nPer-outcome breakdown (threshold=0.05):")
    if len(t05) > 0:
        for outcome in ['H', 'D', 'A']:
            oo = t05[t05['bet_outcome'] == outcome]
            if len(oo) > 0:
                roi = 100.0 * oo['profit'].sum() / len(oo)
                print(f"  {outcome}: {len(oo)} bets, {oo['won'].sum()} wins, ROI={roi:.1f}%")

    # Save
    save_parquet(bets_df, 'backtest.parquet')

    # Save stats as JSON
    with open(MODEL_DIR / 'backtest_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nBacktest complete. Results saved.")


if __name__ == '__main__':
    main()
