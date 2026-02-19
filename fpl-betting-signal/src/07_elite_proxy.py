"""Build elite proxy signals and test whether informed/active FPL managers
carry stronger predictive signal than the full crowd.

Three elite proxy approaches:
1. Price-Movers Signal: Aggregate only from players whose price rose (heavy buying = informed)
2. High-Intensity Signal: Weight by transfer_intensity (transfers_in / selected)
3. Contrarian Signal: Focus on non-template picks (low ownership, high transfers)

Retrains models and compares to baseline.
"""

import sys
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEASONS, TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS,
    DATA_RAW_FPL, DATA_PROCESSED, MODEL_DIR,
    load_parquet, save_parquet,
)


def load_and_enrich_player_gw(season: str) -> pd.DataFrame:
    """Load merged_gw with price change and transfer intensity columns."""
    path = DATA_RAW_FPL / f'{season}_merged_gw.csv'
    df = pd.read_csv(path, encoding='latin-1', low_memory=False)
    df.columns = [c.strip().strip('"') for c in df.columns]

    for col in ['element', 'GW', 'value', 'selected', 'transfers_in', 'transfers_out',
                'transfers_balance', 'fixture', 'opponent_team', 'team_h_score', 'team_a_score']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'transfers_balance' not in df.columns:
        df['transfers_balance'] = df['transfers_in'] - df['transfers_out']

    if df['was_home'].dtype == object:
        df['was_home'] = df['was_home'].astype(str).str.strip().str.lower() == 'true'

    df['gw'] = df['GW'].astype('Int64')
    df['season'] = season

    # Price change (GW-over-GW within season)
    df = df.sort_values(['element', 'gw'])
    df['value_prev'] = df.groupby('element')['value'].shift(1)
    df['price_rose'] = (df['value'] > df['value_prev']).fillna(False)

    # Transfer intensity: what fraction of current owners actively bought this GW
    df['transfer_intensity'] = df['transfers_in'] / (df['selected'] + 1)

    return df


def get_team_id_for_player(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Assign team_id to each player row (same logic as 02_build_match_dataset)."""
    # Load existing matches to get fixture->team mapping
    matches = load_parquet('matches.parquet')
    season_matches = matches[matches['season'] == season]

    # For seasons with team column in merged_gw
    if 'team' in df.columns and df['team'].notna().any():
        # Map team name -> team_id from matches
        team_name_to_id = {}
        for _, row in season_matches.iterrows():
            team_name_to_id[row['team_h_name']] = int(row['team_h_id'])
            team_name_to_id[row['team_a_name']] = int(row['team_a_id'])
        df['team_id'] = df['team'].map(team_name_to_id)
    else:
        # Use players_raw fallback
        pr_path = DATA_RAW_FPL / f'{season}_players_raw.csv'
        if pr_path.exists():
            pr = pd.read_csv(pr_path, encoding='latin-1', low_memory=False)
            pr.columns = [c.strip().strip('"') for c in pr.columns]
            team_map = dict(zip(pr['id'].astype(int), pr['team'].astype(int)))
            df['team_id'] = df['element'].map(team_map)
        else:
            df['team_id'] = -1

    df['team_id'] = df['team_id'].fillna(-1).astype(int)
    return df


def build_elite_match_signals(season: str) -> pd.DataFrame:
    """Build elite proxy signals for each match in a season."""
    df = load_and_enrich_player_gw(season)
    df = get_team_id_for_player(df, season)
    df = df[df['team_id'] > 0]

    matches = load_parquet('matches.parquet')
    season_matches = matches[matches['season'] == season].copy()

    # For each match, compute elite signals from player data
    results = []
    for _, match in season_matches.iterrows():
        gw = match['gw']
        th_id = match['team_h_id']
        ta_id = match['team_a_id']

        home_players = df[(df['gw'] == gw) & (df['team_id'] == th_id)]
        away_players = df[(df['gw'] == gw) & (df['team_id'] == ta_id)]

        if len(home_players) == 0 or len(away_players) == 0:
            results.append({
                'season': season, 'gw': gw, 'fixture_id': match['fixture_id'],
                **{k: np.nan for k in [
                    'elite_transfer_ratio', 'elite_ownership_ratio',
                    'intensity_weighted_ratio', 'contrarian_ratio',
                    'price_mover_count_h', 'price_mover_count_a',
                ]}
            })
            continue

        # --- Signal 1: Price-Movers Transfer Signal ---
        # Only count net transfers from players whose price rose (informed buying)
        pm_h = home_players[home_players['price_rose'] == True]['transfers_balance'].sum()
        pm_a = away_players[away_players['price_rose'] == True]['transfers_balance'].sum()
        elite_transfer_ratio = pm_h / (abs(pm_h) + abs(pm_a) + 1)

        # Count of price movers per team
        pm_count_h = home_players['price_rose'].sum()
        pm_count_a = away_players['price_rose'].sum()

        # --- Signal 2: Price-Mover Ownership ---
        # Ownership only from price-rising players (elite conviction picks)
        pm_own_h = (home_players[home_players['price_rose']]['selected'] * home_players[home_players['price_rose']]['value']).sum()
        pm_own_a = (away_players[away_players['price_rose']]['selected'] * away_players[away_players['price_rose']]['value']).sum()
        elite_ownership_ratio = pm_own_h / (pm_own_h + pm_own_a + 1)

        # --- Signal 3: Intensity-Weighted Transfer Signal ---
        # Weight each player's transfers by their transfer_intensity
        # High intensity = active buying relative to existing ownership = more "informed"
        iw_h = (home_players['transfers_balance'] * home_players['transfer_intensity']).sum()
        iw_a = (away_players['transfers_balance'] * away_players['transfer_intensity']).sum()
        intensity_weighted_ratio = iw_h / (abs(iw_h) + abs(iw_a) + 1)

        # --- Signal 4: Contrarian Signal ---
        # Non-template picks: selected < median, but high transfer intensity
        median_selected = df[df['gw'] == gw]['selected'].median()
        contrarian_h = home_players[
            (home_players['selected'] < median_selected) &
            (home_players['transfer_intensity'] > home_players['transfer_intensity'].quantile(0.5))
        ]['transfers_balance'].sum()
        contrarian_a = away_players[
            (away_players['selected'] < median_selected) &
            (away_players['transfer_intensity'] > away_players['transfer_intensity'].quantile(0.5))
        ]['transfers_balance'].sum()
        contrarian_ratio = contrarian_h / (abs(contrarian_h) + abs(contrarian_a) + 1)

        results.append({
            'season': season, 'gw': gw, 'fixture_id': match['fixture_id'],
            'elite_transfer_ratio': elite_transfer_ratio,
            'elite_ownership_ratio': elite_ownership_ratio,
            'intensity_weighted_ratio': intensity_weighted_ratio,
            'contrarian_ratio': contrarian_ratio,
            'price_mover_count_h': pm_count_h,
            'price_mover_count_a': pm_count_a,
        })

    return pd.DataFrame(results)


def multiclass_brier(y_true, y_proba, n_classes=3):
    return np.mean([
        brier_score_loss((y_true == c).astype(int), y_proba[:, c])
        for c in range(n_classes)
    ])


def main():
    print("=" * 70)
    print("ELITE PROXY ANALYSIS")
    print("Building signals from price-movers and high-intensity transfers...")
    print("=" * 70)

    # Build elite signals for all seasons
    all_elite = []
    for season in SEASONS:
        print(f"  Processing {season}...")
        elite = build_elite_match_signals(season)
        all_elite.append(elite)

    elite_df = pd.concat(all_elite, ignore_index=True)
    print(f"\nElite signals computed for {len(elite_df)} matches")

    # Merge with existing signals
    signals = load_parquet('signals.parquet')
    merged = signals.merge(elite_df, on=['season', 'gw', 'fixture_id'], how='left')

    # Report elite signal stats
    elite_cols = ['elite_transfer_ratio', 'elite_ownership_ratio',
                  'intensity_weighted_ratio', 'contrarian_ratio']

    print("\nElite signal statistics:")
    for col in elite_cols:
        s = merged[col].dropna()
        print(f"  {col:<30} mean={s.mean():>8.4f}  std={s.std():>8.4f}  NaN={merged[col].isna().sum()}")

    # Correlation with odds and result
    print("\nCorrelation with implied_prob_h:")
    for col in elite_cols:
        valid = merged[['implied_prob_h', col]].dropna()
        r = valid['implied_prob_h'].corr(valid[col])
        print(f"  {col:<30} r={r:>7.4f}")

    # Compare with original crowd signals
    print("\n(Original crowd signal correlations for reference:)")
    for col in ['transfer_ratio', 'ownership_ratio']:
        valid = merged[['implied_prob_h', col]].dropna()
        r = valid['implied_prob_h'].corr(valid[col])
        print(f"  {col:<30} r={r:>7.4f}")

    # --- Train models with elite signals ---
    print("\n" + "=" * 70)
    print("TRAINING MODELS WITH ELITE SIGNALS")
    print("=" * 70)

    ODDS_FEATURES = ['implied_prob_h', 'implied_prob_d', 'implied_prob_a']
    CROWD_FPL = ['transfer_ratio', 'ownership_ratio', 'captain_proxy', 'dcs_ratio', 'velocity_delta']
    ELITE_FPL = elite_cols
    ELITE_PLUS_CROWD = CROWD_FPL + ELITE_FPL

    feature_sets = {
        'odds_only': ODDS_FEATURES,
        'crowd_fpl': CROWD_FPL,
        'elite_fpl': ELITE_FPL,
        'elite+crowd_fpl': ELITE_PLUS_CROWD,
        'elite+odds': ODDS_FEATURES + ELITE_FPL,
        'crowd+odds': ODDS_FEATURES + CROWD_FPL,
        'all_features': ODDS_FEATURES + ELITE_PLUS_CROWD,
    }

    # Prepare data
    all_features = list(set(ODDS_FEATURES + ELITE_PLUS_CROWD))
    clean = merged.dropna(subset=all_features + ['result'])

    train = clean[clean['season'].isin(TRAIN_SEASONS)]
    val = clean[clean['season'].isin(VAL_SEASONS)]
    test = clean[clean['season'].isin(TEST_SEASONS)]

    print(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")

    RESULT_MAP = {'A': 0, 'D': 1, 'H': 2}
    y_train = train['result'].map(RESULT_MAP).values
    y_val = val['result'].map(RESULT_MAP).values
    y_test = test['result'].map(RESULT_MAP).values

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(train[all_features]), columns=all_features, index=train.index)
    X_val = pd.DataFrame(scaler.transform(val[all_features]), columns=all_features, index=val.index)
    X_test = pd.DataFrame(scaler.transform(test[all_features]), columns=all_features, index=test.index)

    print(f"\n{'Model':<25} {'Val Brier':>10} {'Test Brier':>11} {'Val Acc':>8} {'Test Acc':>9} {'BSS vs Odds':>12}")
    print("-" * 80)

    results = {}
    for name, features in feature_sets.items():
        base = LogisticRegression(solver='lbfgs', max_iter=1000, C=0.01)
        model = CalibratedClassifierCV(base, cv=5, method='isotonic')
        model.fit(X_train[features].values, y_train)

        val_proba = model.predict_proba(X_val[features].values)
        test_proba = model.predict_proba(X_test[features].values)

        val_brier = multiclass_brier(y_val, val_proba)
        test_brier = multiclass_brier(y_test, test_proba)
        val_acc = accuracy_score(y_val, model.predict(X_val[features].values))
        test_acc = accuracy_score(y_test, model.predict(X_test[features].values))

        results[name] = {
            'val_brier': val_brier, 'test_brier': test_brier,
            'val_acc': val_acc, 'test_acc': test_acc,
        }

    # Compute BSS vs odds_only
    odds_test_brier = results['odds_only']['test_brier']
    for name, r in results.items():
        bss = 1 - (r['test_brier'] / odds_test_brier) if name != 'odds_only' else 0
        print(f"{name:<25} {r['val_brier']:>10.4f} {r['test_brier']:>11.4f} "
              f"{r['val_acc']:>8.3f} {r['test_acc']:>9.3f} {bss:>+12.4f}")

    # --- Naive baseline for context ---
    naive_brier = multiclass_brier(y_test, np.tile(
        [(y_train == 0).mean(), (y_train == 1).mean(), (y_train == 2).mean()],
        (len(y_test), 1)
    ))

    print(f"\nNaive baseline test Brier: {naive_brier:.4f}")
    print(f"\nSkill vs Naive (% of gap closed):")
    for name, r in results.items():
        skill = 1 - r['test_brier'] / naive_brier
        pct_of_odds = skill / (1 - odds_test_brier / naive_brier) * 100 if name != 'odds_only' else 100
        marker = " <<<" if pct_of_odds > 100 else ""
        print(f"  {name:<25} skill={skill*100:>5.1f}%  ({pct_of_odds:>5.1f}% of odds skill){marker}")

    # Save elite signals
    save_parquet(merged, 'signals_with_elite.parquet')

    # Save results
    with open(MODEL_DIR / 'elite_comparison.json', 'w') as f:
        json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()}, f, indent=2)


if __name__ == '__main__':
    main()
