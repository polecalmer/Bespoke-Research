"""NBA Season-Arc Baseline: How predictable are NBA season outcomes from mid-season stats?

This is Phase 1 of the NBA DFS feasibility analysis. We measure how well
simple mid-season statistics (win%, point differential, form) predict final
outcomes (playoff seeding, playoff appearance, top seed). This establishes
the BASELINE that any crowd signal (DFS ownership) would need to beat.

If win% at game 40 already predicts final seeding with rho > 0.95, there's
essentially no room for DFS signals to add value. If there's meaningful
headroom (rho < 0.90), pursuing DFS data is justified.

Data: NocturneBear/NBA-Data-2010-2024 GitHub repo (team game logs, 2010-2024)

Outputs:
  - nba-season-arc/data/processed/nba_checkpoints.parquet
  - nba-season-arc/outputs/charts/nba_baseline_*.png
  - Console output with model comparisons
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Reuse chart styling from EPL project
FPL_SRC = Path(__file__).resolve().parent.parent.parent / 'fpl-betting-signal' / 'src'
sys.path.insert(0, str(FPL_SRC))
from utils import setup_chart_style, CHART_COLORS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
CHARTS_DIR = PROJECT_ROOT / 'outputs' / 'charts'

# NBA conference assignments (current era, post-2014 realignment)
EAST = {'ATL', 'BKN', 'BOS', 'CHA', 'CHI', 'CLE', 'DET', 'IND',
        'MIA', 'MIL', 'NYK', 'ORL', 'PHI', 'TOR', 'WAS'}
WEST = {'DAL', 'DEN', 'GSW', 'HOU', 'LAC', 'LAL', 'MEM', 'MIN',
        'NOP', 'OKC', 'PHX', 'POR', 'SAC', 'SAS', 'UTA'}
# NOP was NOH before 2013-14, handle in code

CHECKPOINTS = [10, 20, 30, 40, 50, 60, 70]

# Known NBA champions (for validation)
CHAMPIONS = {
    '2017-18': 'GSW', '2018-19': 'TOR', '2019-20': 'LAL',
    '2020-21': 'MIL', '2021-22': 'GSW', '2022-23': 'DEN', '2023-24': 'BOS',
}

# Train/test split: train on 2010-11 through 2020-21 (11 seasons), test on 2021-22 through 2023-24 (3 seasons)
TRAIN_SEASONS = [f'{y}-{str(y+1)[2:]}' for y in range(2010, 2021)]
TEST_SEASONS = ['2021-22', '2022-23', '2023-24']


# ---------------------------------------------------------------------------
# 1. Load and preprocess
# ---------------------------------------------------------------------------

def load_nba_data() -> pd.DataFrame:
    """Load and preprocess NBA team game logs."""
    path = DATA_RAW / 'regular_season_totals.csv'
    df = pd.read_csv(path)

    # Parse home/away from MATCHUP
    df['is_home'] = df['MATCHUP'].str.contains('vs\\.').astype(int)

    # Extract opponent
    df['opponent'] = df['MATCHUP'].str.extract(r'(?:vs\.|@)\s*(\w+)')

    # Parse date
    df['game_date'] = pd.to_datetime(df['GAME_DATE'])

    # Assign game number within season per team
    df = df.sort_values(['SEASON_YEAR', 'TEAM_ABBREVIATION', 'game_date'])
    df['game_num'] = df.groupby(['SEASON_YEAR', 'TEAM_ABBREVIATION']).cumcount() + 1

    # Win as numeric
    df['win'] = (df['WL'] == 'W').astype(int)

    # Conference
    df['conference'] = df['TEAM_ABBREVIATION'].apply(
        lambda x: 'East' if x in EAST else ('West' if x in WEST else 'Unknown')
    )

    # Points allowed (need to derive from opponent's game data)
    # Use PLUS_MINUS: PTS_allowed = PTS - PLUS_MINUS
    df['pts_allowed'] = df['PTS'] - df['PLUS_MINUS']

    print(f"Loaded {len(df)} team-game records")
    print(f"Seasons: {sorted(df['SEASON_YEAR'].unique())}")
    return df


def compute_final_standings(df: pd.DataFrame) -> pd.DataFrame:
    """Compute final season standings with conference seeding."""
    standings = df.groupby(['SEASON_YEAR', 'TEAM_ABBREVIATION', 'TEAM_NAME', 'conference']).agg(
        wins=('win', 'sum'),
        losses=('win', lambda x: len(x) - x.sum()),
        total_points=('PTS', 'sum'),
        total_allowed=('pts_allowed', 'sum'),
        total_plus_minus=('PLUS_MINUS', 'sum'),
        games_played=('win', 'count'),
    ).reset_index()

    standings['win_pct'] = standings['wins'] / standings['games_played']
    standings['ppg'] = standings['total_points'] / standings['games_played']
    standings['papg'] = standings['total_allowed'] / standings['games_played']
    standings['point_diff_pg'] = standings['total_plus_minus'] / standings['games_played']

    # Conference seeding (rank by win% within conference)
    standings['conf_seed'] = standings.groupby(['SEASON_YEAR', 'conference'])['win_pct'].rank(
        ascending=False, method='min'
    ).astype(int)

    # Overall league ranking
    standings['league_rank'] = standings.groupby('SEASON_YEAR')['win_pct'].rank(
        ascending=False, method='min'
    ).astype(int)

    # Playoff: top 10 in each conference make play-in, top 6 guaranteed (pre-2020: top 8)
    # For simplicity, use top 8 in each conference (traditional cutoff)
    standings['playoff'] = (standings['conf_seed'] <= 8).astype(int)
    standings['top4_seed'] = (standings['conf_seed'] <= 4).astype(int)
    standings['top_seed'] = (standings['conf_seed'] == 1).astype(int)

    return standings


# ---------------------------------------------------------------------------
# 2. Build checkpoint features
# ---------------------------------------------------------------------------

def build_checkpoints(df: pd.DataFrame, final_standings: pd.DataFrame) -> pd.DataFrame:
    """At each checkpoint, compute team features and join with final outcomes."""
    rows = []

    for season in df['SEASON_YEAR'].unique():
        sdf = df[df['SEASON_YEAR'] == season]
        standings = final_standings[final_standings['SEASON_YEAR'] == season]
        max_game = sdf['game_num'].max()

        for team in sdf['TEAM_ABBREVIATION'].unique():
            team_data = sdf[sdf['TEAM_ABBREVIATION'] == team].sort_values('game_num')
            final = standings[standings['TEAM_ABBREVIATION'] == team]

            if final.empty:
                continue

            final_row = final.iloc[0]

            for cp in CHECKPOINTS:
                if cp > max_game:
                    continue

                cp_data = team_data[team_data['game_num'] <= cp]
                if len(cp_data) < 5:
                    continue

                n = len(cp_data)
                wins = cp_data['win'].sum()
                win_pct = wins / n
                point_diff = cp_data['PLUS_MINUS'].sum()
                pd_pg = point_diff / n
                ppg = cp_data['PTS'].mean()
                papg = cp_data['pts_allowed'].mean()

                # Home/away splits
                home_games = cp_data[cp_data['is_home'] == 1]
                away_games = cp_data[cp_data['is_home'] == 0]
                home_wpct = home_games['win'].mean() if len(home_games) > 0 else win_pct
                away_wpct = away_games['win'].mean() if len(away_games) > 0 else win_pct

                # Recent form (last 10 games or all if < 10)
                recent = cp_data.tail(min(10, n))
                recent_wpct = recent['win'].mean()
                recent_pd = recent['PLUS_MINUS'].mean()

                # Shooting efficiency
                fg_pct = cp_data['FG_PCT'].mean()
                fg3_pct = cp_data['FG3_PCT'].mean()
                ft_pct = cp_data['FT_PCT'].mean()

                # Assist/turnover
                ast_pg = cp_data['AST'].mean()
                tov_pg = cp_data['TOV'].mean()
                ast_tov = ast_pg / (tov_pg + 0.1)

                # Rebounding
                reb_pg = cp_data['REB'].mean()
                oreb_pg = cp_data['OREB'].mean()

                # Form trend: last 10 win% minus first 10 win%
                if n >= 20:
                    early_wpct = cp_data.head(10)['win'].mean()
                    late_wpct = cp_data.tail(10)['win'].mean()
                    form_trend = late_wpct - early_wpct
                else:
                    form_trend = 0.0

                # Clutch proxy: games decided by <= 5 points
                cp_data_close = cp_data[cp_data['PLUS_MINUS'].abs() <= 5]
                close_wpct = cp_data_close['win'].mean() if len(cp_data_close) > 0 else 0.5

                # Point differential trend
                if n >= 20:
                    early_pd = cp_data.head(10)['PLUS_MINUS'].mean()
                    late_pd = cp_data.tail(10)['PLUS_MINUS'].mean()
                    pd_trend = late_pd - early_pd
                else:
                    pd_trend = 0.0

                row = {
                    'season': season,
                    'team': team,
                    'team_name': final_row['TEAM_NAME'],
                    'conference': final_row['conference'],
                    'checkpoint': cp,
                    'n_games': n,
                    # Core features
                    'win_pct': win_pct,
                    'point_diff_pg': pd_pg,
                    'ppg': ppg,
                    'papg': papg,
                    'home_wpct': home_wpct,
                    'away_wpct': away_wpct,
                    # Form features
                    'recent_wpct': recent_wpct,
                    'recent_pd': recent_pd,
                    'form_trend': form_trend,
                    'pd_trend': pd_trend,
                    # Efficiency features
                    'fg_pct': fg_pct,
                    'fg3_pct': fg3_pct,
                    'ft_pct': ft_pct,
                    'ast_tov': ast_tov,
                    'reb_pg': reb_pg,
                    'oreb_pg': oreb_pg,
                    # Clutch
                    'close_wpct': close_wpct,
                    # Targets
                    'final_conf_seed': final_row['conf_seed'],
                    'final_league_rank': final_row['league_rank'],
                    'final_win_pct': final_row['win_pct'],
                    'final_wins': final_row['wins'],
                    'playoff': final_row['playoff'],
                    'top4_seed': final_row['top4_seed'],
                    'top_seed': final_row['top_seed'],
                }
                rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Models and evaluation
# ---------------------------------------------------------------------------

def evaluate_seeding_prediction(cp_df: pd.DataFrame, feature_sets: dict,
                                 checkpoint: int) -> dict:
    """Predict final conference seed using Ridge regression."""
    data = cp_df[cp_df['checkpoint'] == checkpoint].dropna()

    train = data[data['season'].isin(TRAIN_SEASONS)]
    test = data[data['season'].isin(TEST_SEASONS)]

    if len(train) < 30 or len(test) < 10:
        return {}

    results = {}
    for name, features in feature_sets.items():
        available = [f for f in features if f in data.columns]
        if not available:
            continue

        X_train = train[available].fillna(0).values
        y_train = train['final_conf_seed'].values
        X_test = test[available].fillna(0).values
        y_test = test['final_conf_seed'].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = Ridge(alpha=10.0)
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        rho, p_val = stats.spearmanr(y_test, y_pred)
        mae = np.mean(np.abs(y_test - y_pred))

        # Per-season breakdown
        season_rhos = []
        for s in TEST_SEASONS:
            mask = test['season'] == s
            if mask.sum() >= 10:
                s_rho, _ = stats.spearmanr(y_test[mask.values], y_pred[mask.values])
                season_rhos.append((s, s_rho))

        results[name] = {
            'spearman_rho': rho,
            'p_value': p_val,
            'mae': mae,
            'season_rhos': season_rhos,
        }

    return results


def evaluate_binary(cp_df: pd.DataFrame, feature_sets: dict,
                     checkpoint: int, target: str) -> dict:
    """Predict binary outcome (playoff, top4_seed) using logistic regression."""
    data = cp_df[cp_df['checkpoint'] == checkpoint].dropna(subset=[target])

    train = data[data['season'].isin(TRAIN_SEASONS)]
    test = data[data['season'].isin(TEST_SEASONS)]

    if len(train) < 30 or len(test) < 10:
        return {}

    results = {}
    for name, features in feature_sets.items():
        available = [f for f in features if f in data.columns]
        if not available:
            continue

        X_train = train[available].fillna(0).values
        y_train = train[target].values
        X_test = test[available].fillna(0).values
        y_test = test[target].values

        if y_test.sum() == 0 or y_test.sum() == len(y_test):
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
        model.fit(X_train_s, y_train)

        y_prob = model.predict_proba(X_test_s)
        pos_idx = list(model.classes_).index(1)
        y_prob_pos = y_prob[:, pos_idx]
        y_pred = model.predict(X_test_s)

        brier = brier_score_loss(y_test, y_prob_pos)
        acc = accuracy_score(y_test, y_pred)

        results[name] = {
            'brier': brier,
            'accuracy': acc,
            'n_pos': int(y_test.sum()),
            'n_total': len(y_test),
        }

    return results


# ---------------------------------------------------------------------------
# 4. Visualization
# ---------------------------------------------------------------------------

def plot_seeding_by_checkpoint(results_by_cp: dict, save_path: str):
    """Plot Spearman rho vs checkpoint for conference seed prediction."""
    setup_chart_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = [CHART_COLORS['navy'], CHART_COLORS['teal'],
              CHART_COLORS['coral'], CHART_COLORS['purple']]

    for i, (name, data) in enumerate(results_by_cp.items()):
        cps = sorted(data.keys())
        rhos = [data[cp]['spearman_rho'] for cp in cps]
        ax.plot(cps, rhos, 'o-', color=colors[i % len(colors)],
                linewidth=2, markersize=8, label=name)

    ax.set_xlabel('Games Played (Checkpoint)', fontsize=13)
    ax.set_ylabel('Spearman Rank Correlation (rho)', fontsize=13)
    ax.set_title('NBA Conference Seed Prediction by Mid-Season Checkpoint', fontsize=15, pad=15)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_nba_vs_epl_comparison(nba_results: dict, save_path: str):
    """Compare NBA baseline predictiveness against EPL at matched season %."""
    setup_chart_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    # NBA: Win% Baseline
    nba_cps = sorted(nba_results.get('Win% Baseline', {}).keys())
    nba_rhos = [nba_results['Win% Baseline'][cp]['spearman_rho'] for cp in nba_cps]
    nba_pcts = [cp / 82 * 100 for cp in nba_cps]

    ax.plot(nba_pcts, nba_rhos, 'o-', color=CHART_COLORS['teal'],
            linewidth=2.5, markersize=9, label='NBA (Win% → Conf Seed)')

    # EPL baseline from our analysis (approximate values from results)
    epl_cps = [5, 10, 15, 20, 25, 30]
    epl_pcts = [cp / 38 * 100 for cp in epl_cps]
    epl_rhos = [0.773, 0.840, 0.875, 0.917, 0.902, 0.939]  # From our 09_season_arc.py results

    ax.plot(epl_pcts, epl_rhos, 's--', color=CHART_COLORS['coral'],
            linewidth=2.5, markersize=9, label='EPL (PPG → Final Position)')

    ax.set_xlabel('Season Progress (%)', fontsize=13)
    ax.set_ylabel('Spearman Rank Correlation (rho)', fontsize=13)
    ax.set_title('NBA vs EPL: How Quickly Do Standings Become Predictable?', fontsize=15, pad=15)
    ax.set_ylim(0.5, 1.05)
    ax.set_xlim(0, 100)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add headroom annotation
    ax.axhline(y=0.95, color=CHART_COLORS['white'], linestyle=':', alpha=0.4)
    ax.text(5, 0.955, 'rho=0.95 (minimal headroom)', color=CHART_COLORS['white'],
            alpha=0.5, fontsize=10)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_binary_comparison(playoff_results: dict, top4_results: dict, save_path: str):
    """Brier scores for playoff and top-4 seed predictions."""
    setup_chart_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, results, title in [(axes[0], playoff_results, 'Playoff Appearance'),
                                (axes[1], top4_results, 'Top 4 Seed')]:
        if not results:
            ax.set_title(f'{title}: No data')
            continue

        colors = [CHART_COLORS['navy'], CHART_COLORS['teal'],
                  CHART_COLORS['coral'], CHART_COLORS['purple']]

        for i, (name, data) in enumerate(results.items()):
            cps = sorted(data.keys())
            briers = [data[cp]['brier'] for cp in cps]
            ax.plot(cps, briers, 'o-', color=colors[i % len(colors)],
                    linewidth=2, markersize=8, label=name)

        ax.set_xlabel('Games Played', fontsize=12)
        ax.set_ylabel('Brier Score (lower = better)', fontsize=12)
        ax.set_title(f'{title} Prediction', fontsize=13, pad=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle('NBA Season-Long Binary Outcomes: Baseline Predictiveness',
                 fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_headroom_bar(seeding_results: dict, save_path: str):
    """Bar chart showing prediction error (1 - rho) = headroom at each checkpoint."""
    setup_chart_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    baseline = seeding_results.get('Win% Baseline', {})
    extended = seeding_results.get('Extended Stats', {})

    cps = sorted(baseline.keys())
    baseline_headroom = [1 - baseline[cp]['spearman_rho'] for cp in cps]
    extended_headroom = [1 - extended.get(cp, {}).get('spearman_rho', 0) for cp in cps]

    x = np.arange(len(cps))
    width = 0.35

    ax.bar(x - width/2, baseline_headroom, width, label='After Win%',
           color=CHART_COLORS['coral'], edgecolor=CHART_COLORS['edge'])
    ax.bar(x + width/2, extended_headroom, width, label='After Extended Stats',
           color=CHART_COLORS['teal'], edgecolor=CHART_COLORS['edge'])

    ax.set_xticks(x)
    ax.set_xticklabels([f'G{cp}' for cp in cps])
    ax.set_ylabel('Prediction Error (1 - rho)', fontsize=13)
    ax.set_title('Headroom for DFS Crowd Signals at Each Checkpoint', fontsize=15, pad=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate: headroom threshold
    ax.axhline(y=0.10, color=CHART_COLORS['white'], linestyle=':', alpha=0.4)
    ax.text(0, 0.105, 'Meaningful headroom (>0.10)', color=CHART_COLORS['white'],
            alpha=0.5, fontsize=10)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("NBA SEASON-ARC BASELINE: How Predictable Are NBA Outcomes?")
    print("=" * 70)

    # 1. Load data
    print("\n[1/5] Loading NBA data...")
    df = load_nba_data()

    # Filter to seasons we want (2017-18 through 2023-24)
    target_seasons = TRAIN_SEASONS + TEST_SEASONS
    df = df[df['SEASON_YEAR'].isin(target_seasons)]
    print(f"  Filtered to {df['SEASON_YEAR'].nunique()} seasons: {sorted(df['SEASON_YEAR'].unique())}")

    # 2. Compute final standings
    print("\n[2/5] Computing final standings...")
    standings = compute_final_standings(df)

    # Validate against known champions
    for season, champ in CHAMPIONS.items():
        s = standings[(standings['SEASON_YEAR'] == season) & (standings['TEAM_ABBREVIATION'] == champ)]
        if not s.empty:
            print(f"  {season} champion {champ}: seed={s['conf_seed'].values[0]}, "
                  f"{s['wins'].values[0]}-{s['losses'].values[0]}")

    # Show test season standings
    for season in TEST_SEASONS:
        st = standings[standings['SEASON_YEAR'] == season].sort_values('league_rank')
        print(f"\n  {season} (top 5 / bottom 3):")
        for _, row in st.head(5).iterrows():
            print(f"    {row['league_rank']:>2}. {row['TEAM_NAME']:<28} "
                  f"{row['wins']}-{row['losses']}  ({row['conference']} #{row['conf_seed']})")
        print(f"    ...")
        for _, row in st.tail(3).iterrows():
            print(f"    {row['league_rank']:>2}. {row['TEAM_NAME']:<28} "
                  f"{row['wins']}-{row['losses']}  ({row['conference']} #{row['conf_seed']})")

    # 3. Build checkpoint dataset
    print("\n[3/5] Building checkpoint dataset...")
    cp_df = build_checkpoints(df, standings)
    print(f"  {len(cp_df)} checkpoint observations")

    # Save
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    cp_df.to_parquet(DATA_PROCESSED / 'nba_checkpoints.parquet', index=False)
    print(f"  Saved nba_checkpoints.parquet")

    # --- Feature sets ---
    baseline_features = ['win_pct', 'point_diff_pg']
    form_features = ['win_pct', 'point_diff_pg', 'recent_wpct', 'recent_pd',
                     'form_trend', 'pd_trend']
    extended_features = ['win_pct', 'point_diff_pg', 'recent_wpct', 'recent_pd',
                         'form_trend', 'pd_trend', 'home_wpct', 'away_wpct',
                         'fg_pct', 'fg3_pct', 'ast_tov', 'reb_pg', 'close_wpct']

    feature_sets = {
        'Win% Baseline': baseline_features,
        'Win% + Form': form_features,
        'Extended Stats': extended_features,
    }

    # 4. Run models
    print("\n[4/5] Running models...")
    print("\n" + "=" * 70)
    print("CONFERENCE SEED PREDICTION (Spearman Rank Correlation)")
    print("=" * 70)

    seed_results = {name: {} for name in feature_sets}
    for cp in CHECKPOINTS:
        results = evaluate_seeding_prediction(cp_df, feature_sets, cp)
        print(f"\n--- Game {cp} ---")
        for name, r in results.items():
            rho = r['spearman_rho']
            mae = r['mae']
            p = r['p_value']
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
            per_season = '  '.join([f'{s}:{sr:.3f}' for s, sr in r['season_rhos']])
            print(f"  {name:<20} rho={rho:.4f}{sig:<4} MAE={mae:.2f}  [{per_season}]")
            seed_results[name][cp] = r

    # Playoff prediction
    print("\n" + "=" * 70)
    print("PLAYOFF APPEARANCE (Logistic Regression)")
    print("=" * 70)

    playoff_results = {name: {} for name in feature_sets}
    for cp in CHECKPOINTS:
        results = evaluate_binary(cp_df, feature_sets, cp, 'playoff')
        print(f"\n--- Game {cp} ---")
        for name, r in results.items():
            print(f"  {name:<20} Brier={r['brier']:.4f}  Acc={r['accuracy']:.3f}  "
                  f"({r['n_pos']}/{r['n_total']} positive)")
            playoff_results[name][cp] = r

    # Top 4 seed prediction
    print("\n" + "=" * 70)
    print("TOP 4 CONFERENCE SEED (Logistic Regression)")
    print("=" * 70)

    top4_results = {name: {} for name in feature_sets}
    for cp in CHECKPOINTS:
        results = evaluate_binary(cp_df, feature_sets, cp, 'top4_seed')
        print(f"\n--- Game {cp} ---")
        for name, r in results.items():
            print(f"  {name:<20} Brier={r['brier']:.4f}  Acc={r['accuracy']:.3f}  "
                  f"({r['n_pos']}/{r['n_total']} positive)")
            top4_results[name][cp] = r

    # 5. Headroom analysis
    print("\n" + "=" * 70)
    print("HEADROOM ANALYSIS: Room for DFS Crowd Signals?")
    print("=" * 70)

    print(f"\n{'Checkpoint':>12} | {'Win% rho':>10} | {'Extended rho':>12} | {'Headroom':>10} | {'Assessment'}")
    print("-" * 70)
    for cp in CHECKPOINTS:
        b_rho = seed_results.get('Win% Baseline', {}).get(cp, {}).get('spearman_rho', 0)
        e_rho = seed_results.get('Extended Stats', {}).get(cp, {}).get('spearman_rho', 0)
        headroom = 1 - e_rho
        pct = cp / 82 * 100

        if headroom > 0.15:
            assess = "GOOD headroom"
        elif headroom > 0.05:
            assess = "Some headroom"
        else:
            assess = "Minimal headroom"

        print(f"  Game {cp:>3} ({pct:>4.0f}%) | {b_rho:>10.4f} | {e_rho:>12.4f} | {headroom:>10.4f} | {assess}")

    # EPL comparison
    print("\n" + "=" * 70)
    print("NBA vs EPL COMPARISON (at equivalent season %)")
    print("=" * 70)

    epl_data = {
        13: ('GW5', 0.773, 0.813),   # 13% through season
        26: ('GW10', 0.840, 0.857),   # 26%
        39: ('GW15', 0.875, 0.870),   # 39%
        53: ('GW20', 0.917, 0.930),   # 53%
        66: ('GW25', 0.902, 0.924),   # 66%
        79: ('GW30', 0.939, 0.949),   # 79%
    }

    print(f"\n{'Season %':>10} | {'NBA Baseline':>12} | {'NBA Extended':>12} | {'EPL Baseline':>12} | {'EPL Extended':>12} | {'Difference'}")
    print("-" * 85)
    for cp in CHECKPOINTS:
        pct = cp / 82 * 100
        nba_b = seed_results.get('Win% Baseline', {}).get(cp, {}).get('spearman_rho', 0)
        nba_e = seed_results.get('Extended Stats', {}).get(cp, {}).get('spearman_rho', 0)

        # Find closest EPL checkpoint
        closest_epl = min(epl_data.keys(), key=lambda x: abs(x - pct))
        epl_name, epl_b, epl_e = epl_data[closest_epl]

        diff = nba_e - epl_e
        print(f"  {pct:>7.0f}%  | {nba_b:>12.4f} | {nba_e:>12.4f} | "
              f"{epl_b:>12.4f} | {epl_e:>12.4f} | {diff:>+.4f} "
              f"({'NBA more predictable' if diff > 0 else 'EPL more predictable'})")

    # 6. Charts
    print("\n" + "=" * 70)
    print("GENERATING CHARTS...")
    print("=" * 70)

    plot_seeding_by_checkpoint(
        seed_results,
        str(CHARTS_DIR / 'nba_baseline_seeding.png')
    )

    plot_nba_vs_epl_comparison(
        seed_results,
        str(CHARTS_DIR / 'nba_vs_epl_comparison.png')
    )

    plot_binary_comparison(
        playoff_results, top4_results,
        str(CHARTS_DIR / 'nba_baseline_binary.png')
    )

    plot_headroom_bar(
        seed_results,
        str(CHARTS_DIR / 'nba_headroom.png')
    )

    # 7. Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    mid_season_rho = seed_results.get('Extended Stats', {}).get(40, {}).get('spearman_rho', 0)
    early_rho = seed_results.get('Extended Stats', {}).get(20, {}).get('spearman_rho', 0)

    print(f"\n  At game 20 (24% through): extended rho = {early_rho:.4f}")
    print(f"  At game 40 (49% through): extended rho = {mid_season_rho:.4f}")
    print(f"  Headroom at game 40: {1 - mid_season_rho:.4f}")

    if mid_season_rho > 0.95:
        print("\n  VERDICT: Minimal headroom. NBA season outcomes are highly")
        print("  predictable from mid-season stats alone. DFS crowd signals")
        print("  are unlikely to add meaningful value.")
        print("  RECOMMENDATION: Do not invest in DFS ownership data.")
    elif mid_season_rho > 0.90:
        print("\n  VERDICT: Some headroom exists, but the bar is high.")
        print("  DFS crowd signals would need to capture something beyond")
        print("  box score stats — possible but uncertain.")
        print("  RECOMMENDATION: Only pursue if DFS data is cheap (<$30).")
    else:
        print("\n  VERDICT: Meaningful headroom exists!")
        print("  NBA season outcomes have enough prediction error that")
        print("  DFS crowd signals could plausibly add value.")
        print("  RECOMMENDATION: Invest in DFS ownership data (Analytics by Adam).")

    print(f"\n  Done. Results saved to {DATA_PROCESSED / 'nba_checkpoints.parquet'}")
    print(f"  Charts saved to {CHARTS_DIR}/nba_baseline_*.png")


if __name__ == '__main__':
    main()
