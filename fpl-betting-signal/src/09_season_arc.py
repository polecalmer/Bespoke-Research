"""Season-arc analysis: Do FPL signals predict season-long outcomes?

Instead of predicting individual matches, this tests whether cumulative FPL
crowd behavior at mid-season checkpoints (GW5, GW10, GW15, GW20, GW25, GW30)
predicts final league position, top-4 finish, and relegation.

Baseline: actual PPG and market-implied expected points at each checkpoint.
Test: do FPL signals (ownership trajectory, transfer momentum, quality drift)
add predictive value beyond these baselines?

This is the season-long market equivalent of our match-level analysis.

Outputs:
  - data/processed/season_arc.parquet
  - outputs/charts/season_arc_*.png
  - Console output with model comparisons
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, accuracy_score, log_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (load_parquet, save_parquet, setup_chart_style,
                   SEASONS, TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS,
                   DATA_RAW_FPL, CHARTS_DIR, CHART_COLORS)


CHECKPOINTS = [5, 10, 15, 20, 25, 30]


# ---------------------------------------------------------------------------
# 1. Build team-GW panel from match data
# ---------------------------------------------------------------------------

def build_team_gw_panel(matches: pd.DataFrame) -> pd.DataFrame:
    """From match-level data, create a team-gw panel with points and xPoints."""

    rows = []
    for _, m in matches.iterrows():
        season, gw = m['season'], m['gw']
        # Home team
        h_pts = 3 if m['result'] == 'H' else (1 if m['result'] == 'D' else 0)
        h_xpts = m['implied_prob_h'] * 3 + m['implied_prob_d'] * 1
        rows.append({
            'season': season, 'gw': gw,
            'team': m['team_h_name'],
            'points': h_pts,
            'xPoints': h_xpts,
            'goals_for': m['team_h_score'],
            'goals_against': m['team_a_score'],
            'is_home': 1,
            'total_selected': m['total_selected_h'],
            'net_transfers': m['net_transfers_h'],
            'price_weighted_selected': m['price_weighted_selected_h'],
            'def_gk_selected': m['def_gk_selected_h'],
            'opp_total_selected': m['total_selected_a'],
        })
        # Away team
        a_pts = 3 if m['result'] == 'A' else (1 if m['result'] == 'D' else 0)
        a_xpts = m['implied_prob_a'] * 3 + m['implied_prob_d'] * 1
        rows.append({
            'season': season, 'gw': gw,
            'team': m['team_a_name'],
            'points': a_pts,
            'xPoints': a_xpts,
            'goals_for': m['team_a_score'],
            'goals_against': m['team_h_score'],
            'is_home': 0,
            'total_selected': m['total_selected_a'],
            'net_transfers': m['net_transfers_a'],
            'price_weighted_selected': m['price_weighted_selected_a'],
            'def_gk_selected': m['def_gk_selected_a'],
            'opp_total_selected': m['total_selected_h'],
        })

    panel = pd.DataFrame(rows)
    panel = panel.sort_values(['season', 'team', 'gw']).reset_index(drop=True)
    return panel


# ---------------------------------------------------------------------------
# 2. Build FPL team-level signals from raw player data
# ---------------------------------------------------------------------------

def load_player_gw_data(season: str) -> pd.DataFrame:
    """Load merged_gw.csv for a season with basic columns."""
    path = DATA_RAW_FPL / f'{season}_merged_gw.csv'
    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, low_memory=False, encoding='latin-1')

    # Normalize column names
    col_map = {}
    if 'GW' in df.columns:
        col_map['GW'] = 'gw'
    if 'round' in df.columns and 'GW' not in df.columns:
        col_map['round'] = 'gw'

    df = df.rename(columns=col_map)

    # Ensure 'team' column exists
    if 'team' not in df.columns:
        return pd.DataFrame()

    # Keep essential columns
    keep = ['element', 'team', 'gw', 'selected', 'transfers_in', 'transfers_out',
            'transfers_balance', 'value', 'total_points', 'minutes',
            'ict_index', 'influence', 'creativity', 'threat',
            'goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
            'position']
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    # Convert types
    for col in ['selected', 'transfers_in', 'transfers_out', 'transfers_balance',
                'value', 'total_points', 'minutes', 'gw']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['season'] = season
    return df


def compute_team_gw_fpl_signals(player_data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player-level FPL data to team-GW signals.

    Returns one row per (season, team, gw) with FPL-derived features.
    """
    if player_data.empty:
        return pd.DataFrame()

    # Group by season, team, gw
    grouped = player_data.groupby(['season', 'team', 'gw'])

    agg = grouped.agg(
        fpl_total_selected=('selected', 'sum'),
        fpl_net_transfers=('transfers_balance', 'sum'),
        fpl_transfers_in=('transfers_in', 'sum'),
        fpl_transfers_out=('transfers_out', 'sum'),
        fpl_avg_value=('value', 'mean'),
        fpl_total_value=('value', 'sum'),
        fpl_total_points=('total_points', 'sum'),
        fpl_avg_points=('total_points', 'mean'),
        fpl_total_minutes=('minutes', 'sum'),
        fpl_n_players=('element', 'nunique'),
    ).reset_index()

    # ICT signals (if available)
    if 'ict_index' in player_data.columns:
        ict_agg = grouped.agg(
            fpl_total_ict=('ict_index', lambda x: pd.to_numeric(x, errors='coerce').sum()),
            fpl_total_threat=('threat', lambda x: pd.to_numeric(x, errors='coerce').sum()),
            fpl_total_influence=('influence', lambda x: pd.to_numeric(x, errors='coerce').sum()),
            fpl_total_creativity=('creativity', lambda x: pd.to_numeric(x, errors='coerce').sum()),
        ).reset_index()
        agg = agg.merge(ict_agg, on=['season', 'team', 'gw'], how='left')

    return agg


# ---------------------------------------------------------------------------
# 3. Build checkpoint features
# ---------------------------------------------------------------------------

def build_checkpoint_dataset(team_panel: pd.DataFrame,
                              fpl_signals: pd.DataFrame,
                              final_standings: pd.DataFrame) -> pd.DataFrame:
    """At each checkpoint GW, compute cumulative features for each team.

    Returns a dataset with one row per (season, team, checkpoint).
    """
    rows = []

    for season in team_panel['season'].unique():
        sp = team_panel[team_panel['season'] == season]
        sf = fpl_signals[fpl_signals['season'] == season] if not fpl_signals.empty else pd.DataFrame()
        standings = final_standings[final_standings['season'] == season]

        if standings.empty:
            continue

        max_gw = sp['gw'].max()

        for team in sp['team'].unique():
            team_data = sp[sp['team'] == team].sort_values('gw')
            team_fpl = sf[sf['team'] == team].sort_values('gw') if not sf.empty else pd.DataFrame()

            # Final outcome
            final = standings[standings['team'] == team]
            if final.empty:
                continue
            final_pos = final['final_position'].values[0]
            final_pts = final['final_points'].values[0]
            top4 = 1 if final_pos <= 4 else 0
            relegated = 1 if final_pos >= 18 else 0
            champion = 1 if final_pos == 1 else 0

            for cp in CHECKPOINTS:
                if cp > max_gw:
                    continue

                # Matches up to checkpoint
                cp_data = team_data[team_data['gw'] <= cp]
                if len(cp_data) == 0:
                    continue

                n_matches = len(cp_data)
                cum_points = cp_data['points'].sum()
                cum_xpoints = cp_data['xPoints'].sum()
                ppg = cum_points / n_matches
                xppg = cum_xpoints / n_matches
                gf = cp_data['goals_for'].sum()
                ga = cp_data['goals_against'].sum()
                gd = gf - ga

                # Over/under-performance vs odds
                overperf = cum_points - cum_xpoints

                # Points trend (last 5 GWs vs first 5 GWs)
                if cp >= 10:
                    early = cp_data[cp_data['gw'] <= cp_data['gw'].min() + 4]['points'].mean()
                    late = cp_data[cp_data['gw'] > cp - 5]['points'].mean()
                    form_trend = late - early
                else:
                    form_trend = 0.0

                # Home/away split
                home_ppg = cp_data[cp_data['is_home'] == 1]['points'].mean() if (cp_data['is_home'] == 1).any() else ppg
                away_ppg = cp_data[cp_data['is_home'] == 0]['points'].mean() if (cp_data['is_home'] == 0).any() else ppg

                row = {
                    'season': season,
                    'team': team,
                    'checkpoint': cp,
                    'n_matches': n_matches,
                    # Baseline features
                    'cum_points': cum_points,
                    'cum_xpoints': cum_xpoints,
                    'ppg': ppg,
                    'xppg': xppg,
                    'gd': gd,
                    'gd_per_game': gd / n_matches,
                    'overperf': overperf,
                    'form_trend': form_trend,
                    'home_ppg': home_ppg,
                    'away_ppg': away_ppg,
                    # Targets
                    'final_position': final_pos,
                    'final_points': final_pts,
                    'top4': top4,
                    'relegated': relegated,
                    'champion': champion,
                }

                # FPL signals at checkpoint
                if not team_fpl.empty:
                    cp_fpl = team_fpl[team_fpl['gw'] <= cp]
                    if len(cp_fpl) > 0:
                        # Cumulative ownership level
                        row['fpl_selected_latest'] = cp_fpl['fpl_total_selected'].iloc[-1]
                        row['fpl_selected_mean'] = cp_fpl['fpl_total_selected'].mean()

                        # Ownership trajectory (slope of selected over GWs)
                        if len(cp_fpl) >= 3:
                            x = cp_fpl['gw'].values.astype(float)
                            y = cp_fpl['fpl_total_selected'].values.astype(float)
                            # Normalize y to avoid numerical issues
                            y_norm = y / (y.mean() + 1)
                            slope, _, _, _, _ = stats.linregress(x, y_norm)
                            row['fpl_ownership_slope'] = slope
                        else:
                            row['fpl_ownership_slope'] = 0.0

                        # Transfer momentum (average net transfers)
                        row['fpl_transfer_momentum'] = cp_fpl['fpl_net_transfers'].mean()

                        # Recent transfer momentum (last 5 GWs)
                        recent_fpl = cp_fpl[cp_fpl['gw'] > cp - 5]
                        if len(recent_fpl) > 0:
                            row['fpl_recent_transfer_mom'] = recent_fpl['fpl_net_transfers'].mean()
                        else:
                            row['fpl_recent_transfer_mom'] = 0.0

                        # Transfer acceleration
                        if len(cp_fpl) >= 6:
                            early_tr = cp_fpl[cp_fpl['gw'] <= cp_fpl['gw'].min() + 4]['fpl_net_transfers'].mean()
                            late_tr = cp_fpl[cp_fpl['gw'] > cp - 5]['fpl_net_transfers'].mean()
                            row['fpl_transfer_accel'] = late_tr - early_tr
                        else:
                            row['fpl_transfer_accel'] = 0.0

                        # Value drift (are managers investing more in this team?)
                        row['fpl_avg_value_latest'] = cp_fpl['fpl_avg_value'].iloc[-1]
                        if len(cp_fpl) >= 3:
                            x = cp_fpl['gw'].values.astype(float)
                            y = cp_fpl['fpl_avg_value'].values.astype(float)
                            slope, _, _, _, _ = stats.linregress(x, y)
                            row['fpl_value_slope'] = slope
                        else:
                            row['fpl_value_slope'] = 0.0

                        # Points per GW (FPL points earned by team's players)
                        row['fpl_points_per_gw'] = cp_fpl['fpl_total_points'].mean()

                        # Quality signal: ownership-weighted team performance
                        row['fpl_quality'] = (
                            cp_fpl['fpl_total_points'].sum() /
                            (cp_fpl['fpl_total_selected'].mean() / 1e6 + 1)
                        )

                        # ICT signals (if available)
                        if 'fpl_total_ict' in cp_fpl.columns:
                            row['fpl_ict_per_gw'] = pd.to_numeric(
                                cp_fpl['fpl_total_ict'], errors='coerce'
                            ).mean()
                            row['fpl_threat_per_gw'] = pd.to_numeric(
                                cp_fpl['fpl_total_threat'], errors='coerce'
                            ).mean()
                        else:
                            row['fpl_ict_per_gw'] = 0.0
                            row['fpl_threat_per_gw'] = 0.0

                        # Relative ownership vs league average
                        season_fpl_at_cp = sf[sf['gw'] <= cp]
                        if len(season_fpl_at_cp) > 0:
                            league_avg = season_fpl_at_cp.groupby('gw')['fpl_total_selected'].mean()
                            team_avg = cp_fpl.set_index('gw')['fpl_total_selected']
                            common_gws = league_avg.index.intersection(team_avg.index)
                            if len(common_gws) > 0:
                                rel_own = (team_avg[common_gws] / (league_avg[common_gws] + 1)).mean()
                                row['fpl_relative_ownership'] = rel_own
                            else:
                                row['fpl_relative_ownership'] = 1.0
                        else:
                            row['fpl_relative_ownership'] = 1.0
                    else:
                        # No FPL data for this checkpoint
                        for col in ['fpl_selected_latest', 'fpl_selected_mean',
                                    'fpl_ownership_slope', 'fpl_transfer_momentum',
                                    'fpl_recent_transfer_mom', 'fpl_transfer_accel',
                                    'fpl_avg_value_latest', 'fpl_value_slope',
                                    'fpl_points_per_gw', 'fpl_quality',
                                    'fpl_ict_per_gw', 'fpl_threat_per_gw',
                                    'fpl_relative_ownership']:
                            row[col] = np.nan

                rows.append(row)

    return pd.DataFrame(rows)


def compute_final_standings(team_panel: pd.DataFrame) -> pd.DataFrame:
    """Compute final league standings from team-GW panel."""
    standings = team_panel.groupby(['season', 'team']).agg(
        final_points=('points', 'sum'),
        matches_played=('points', 'count'),
    ).reset_index()

    # Rank within each season
    standings['final_position'] = standings.groupby('season')['final_points'].rank(
        ascending=False, method='min'
    ).astype(int)

    # Sort for display
    standings = standings.sort_values(['season', 'final_position'])
    return standings


# ---------------------------------------------------------------------------
# 4. Models and evaluation
# ---------------------------------------------------------------------------

def evaluate_position_prediction(df: pd.DataFrame, feature_sets: dict,
                                  checkpoint: int, split_col: str = 'season') -> dict:
    """Train Ridge regression to predict final_position at a given checkpoint.

    Uses Spearman rank correlation as the primary metric.
    """
    cp_data = df[df['checkpoint'] == checkpoint].dropna(subset=['final_position'])

    train = cp_data[cp_data['season'].isin(TRAIN_SEASONS)]
    test = cp_data[cp_data['season'].isin(TEST_SEASONS)]

    if len(train) < 20 or len(test) < 10:
        return {}

    results = {}
    for name, features in feature_sets.items():
        available = [f for f in features if f in cp_data.columns]
        if len(available) == 0:
            continue

        X_train = train[available].fillna(0).values
        y_train = train['final_position'].values
        X_test = test[available].fillna(0).values
        y_test = test['final_position'].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = Ridge(alpha=10.0)
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        # Spearman correlation
        rho, p_val = stats.spearmanr(y_test, y_pred)

        # MAE
        mae = np.mean(np.abs(y_test - y_pred))

        # Per-season breakdown
        season_rhos = []
        for s in TEST_SEASONS:
            s_mask = test['season'] == s
            if s_mask.sum() >= 5:
                s_rho, _ = stats.spearmanr(y_test[s_mask.values], y_pred[s_mask.values])
                season_rhos.append(s_rho)

        results[name] = {
            'spearman_rho': rho,
            'p_value': p_val,
            'mae': mae,
            'season_rhos': season_rhos,
            'n_features': len(available),
            'features': available,
        }

    return results


def evaluate_binary_prediction(df: pd.DataFrame, feature_sets: dict,
                                checkpoint: int, target: str) -> dict:
    """Train logistic regression to predict binary outcome (top4, relegated).

    Returns Brier score, accuracy, and log loss.
    """
    cp_data = df[df['checkpoint'] == checkpoint].dropna(subset=[target])

    train = cp_data[cp_data['season'].isin(TRAIN_SEASONS)]
    test = cp_data[cp_data['season'].isin(TEST_SEASONS)]

    if len(train) < 20 or len(test) < 10:
        return {}

    results = {}
    for name, features in feature_sets.items():
        available = [f for f in features if f in cp_data.columns]
        if len(available) == 0:
            continue

        X_train = train[available].fillna(0).values
        y_train = train[target].values
        X_test = test[available].fillna(0).values
        y_test = test[target].values

        # Skip if no positive cases in test
        if y_test.sum() == 0 or y_test.sum() == len(y_test):
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
        model.fit(X_train_s, y_train)

        y_prob = model.predict_proba(X_test_s)
        # Ensure we get the positive class probability
        pos_idx = list(model.classes_).index(1) if 1 in model.classes_ else 0
        y_prob_pos = y_prob[:, pos_idx]
        y_pred = model.predict(X_test_s)

        brier = brier_score_loss(y_test, y_prob_pos)
        acc = accuracy_score(y_test, y_pred)

        results[name] = {
            'brier': brier,
            'accuracy': acc,
            'n_pos': int(y_test.sum()),
            'n_total': len(y_test),
            'n_features': len(available),
        }

    return results


# ---------------------------------------------------------------------------
# 5. Feature importance analysis
# ---------------------------------------------------------------------------

def get_feature_importance(df: pd.DataFrame, features: list,
                            target: str, checkpoint: int) -> pd.DataFrame:
    """Extract feature importance from Ridge regression."""
    cp_data = df[df['checkpoint'] == checkpoint].dropna(subset=[target])
    train = cp_data[cp_data['season'].isin(TRAIN_SEASONS)]

    available = [f for f in features if f in cp_data.columns]
    if len(available) == 0:
        return pd.DataFrame()

    X = train[available].fillna(0).values
    y = train[target].values

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    model = Ridge(alpha=10.0)
    model.fit(X_s, y)

    imp = pd.DataFrame({
        'feature': available,
        'coefficient': model.coef_,
        'abs_importance': np.abs(model.coef_),
    }).sort_values('abs_importance', ascending=False)

    return imp


# ---------------------------------------------------------------------------
# 6. Visualization
# ---------------------------------------------------------------------------

def plot_spearman_by_checkpoint(results_by_cp: dict, title: str, save_path: str):
    """Plot Spearman rho vs checkpoint GW for each model."""
    setup_chart_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = [CHART_COLORS['navy'], CHART_COLORS['teal'],
              CHART_COLORS['coral'], CHART_COLORS['purple']]

    for i, (model_name, data) in enumerate(results_by_cp.items()):
        cps = sorted(data.keys())
        rhos = [data[cp]['spearman_rho'] for cp in cps]
        ax.plot(cps, rhos, 'o-', color=colors[i % len(colors)],
                linewidth=2, markersize=8, label=model_name)

    ax.set_xlabel('Checkpoint Gameweek', fontsize=13)
    ax.set_ylabel('Spearman Rank Correlation (rho)', fontsize=13)
    ax.set_title(title, fontsize=15, pad=15)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_fpl_signal_value_added(results_by_cp: dict, save_path: str):
    """Bar chart showing FPL signal improvement over PPG baseline at each checkpoint."""
    setup_chart_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    cps = sorted(list(results_by_cp.get('PPG Baseline', {}).keys()))
    if not cps:
        return

    baseline_rhos = []
    fpl_rhos = []
    combined_rhos = []
    for cp in cps:
        b = results_by_cp.get('PPG Baseline', {}).get(cp, {}).get('spearman_rho', 0)
        f = results_by_cp.get('FPL Signals', {}).get(cp, {}).get('spearman_rho', 0)
        c = results_by_cp.get('PPG + FPL', {}).get(cp, {}).get('spearman_rho', 0)
        baseline_rhos.append(b)
        fpl_rhos.append(f)
        combined_rhos.append(c)

    x = np.arange(len(cps))
    width = 0.25

    ax.bar(x - width, baseline_rhos, width, label='PPG Baseline',
           color=CHART_COLORS['navy'], edgecolor=CHART_COLORS['edge'])
    ax.bar(x, fpl_rhos, width, label='FPL Signals Only',
           color=CHART_COLORS['teal'], edgecolor=CHART_COLORS['edge'])
    ax.bar(x + width, combined_rhos, width, label='PPG + FPL Combined',
           color=CHART_COLORS['coral'], edgecolor=CHART_COLORS['edge'])

    ax.set_xticks(x)
    ax.set_xticklabels([f'GW{cp}' for cp in cps])
    ax.set_ylabel('Spearman rho (higher = better)', fontsize=13)
    ax.set_title('Final Position Prediction: FPL Signals vs PPG Baseline', fontsize=15, pad=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_ownership_trajectory(checkpoint_df: pd.DataFrame, save_path: str):
    """Scatter: FPL ownership slope vs final position at GW20."""
    setup_chart_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for idx, cp in enumerate([15, 25]):
        ax = axes[idx]
        data = checkpoint_df[
            (checkpoint_df['checkpoint'] == cp) &
            checkpoint_df['fpl_ownership_slope'].notna()
        ]

        if data.empty:
            continue

        # Color by outcome
        colors_map = []
        for _, row in data.iterrows():
            if row['final_position'] <= 4:
                colors_map.append(CHART_COLORS['teal'])
            elif row['final_position'] >= 18:
                colors_map.append(CHART_COLORS['coral'])
            else:
                colors_map.append(CHART_COLORS['navy'])

        ax.scatter(data['fpl_ownership_slope'], data['final_position'],
                  c=colors_map, alpha=0.6, s=50, edgecolors=CHART_COLORS['edge'])

        rho, p = stats.spearmanr(data['fpl_ownership_slope'], data['final_position'])
        ax.set_xlabel('FPL Ownership Slope', fontsize=12)
        ax.set_ylabel('Final League Position', fontsize=12)
        ax.set_title(f'GW{cp}: Ownership Trend vs Final Position (rho={rho:.3f})',
                     fontsize=13, pad=10)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=CHART_COLORS['teal'], label='Top 4'),
            Patch(facecolor=CHART_COLORS['navy'], label='Mid-table'),
            Patch(facecolor=CHART_COLORS['coral'], label='Relegated'),
        ]
        ax.legend(handles=legend_elements, fontsize=10)

    plt.suptitle('FPL Ownership Trajectory vs Season Outcome', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_binary_brier(top4_results: dict, releg_results: dict, save_path: str):
    """Compare Brier scores for top-4 and relegation predictions."""
    setup_chart_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, results, title in [(axes[0], top4_results, 'Top 4 Finish'),
                                (axes[1], releg_results, 'Relegation')]:
        if not results:
            ax.set_title(f'{title}: No data')
            continue

        cps = sorted(list(next(iter(results.values())).keys()))
        colors = [CHART_COLORS['navy'], CHART_COLORS['teal'],
                  CHART_COLORS['coral'], CHART_COLORS['purple']]

        for i, (model_name, data) in enumerate(results.items()):
            briers = [data.get(cp, {}).get('brier', np.nan) for cp in cps]
            ax.plot(cps, briers, 'o-', color=colors[i % len(colors)],
                    linewidth=2, markersize=8, label=model_name)

        ax.set_xlabel('Checkpoint Gameweek', fontsize=12)
        ax.set_ylabel('Brier Score (lower = better)', fontsize=12)
        ax.set_title(f'{title} Prediction', fontsize=13, pad=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Season-Long Binary Outcomes: FPL Signals vs Baseline',
                 fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("SEASON-ARC ANALYSIS: FPL Signals vs Season-Long Outcomes")
    print("=" * 70)

    # 1. Load match data
    print("\n[1/6] Loading match data...")
    matches = load_parquet('matches.parquet')
    print(f"  {len(matches)} matches across {matches['season'].nunique()} seasons")

    # 2. Build team-GW panel
    print("\n[2/6] Building team-GW panel...")
    team_panel = build_team_gw_panel(matches)
    print(f"  {len(team_panel)} team-GW records")

    # 3. Compute final standings
    print("\n[3/6] Computing final standings...")
    final_standings = compute_final_standings(team_panel)

    # Filter to complete seasons (38 matches per team minimum = 17 teams * 2)
    complete = final_standings[final_standings['matches_played'] >= 34]
    complete_seasons = complete.groupby('season').filter(lambda x: len(x) >= 19)['season'].unique()
    print(f"  Complete seasons: {list(complete_seasons)}")

    # Show actual standings for validation
    for s in ['2022-23', '2023-24']:
        st = final_standings[final_standings['season'] == s].sort_values('final_position')
        print(f"\n  {s} standings (top 5 / bottom 3):")
        for _, row in st.head(5).iterrows():
            print(f"    {row['final_position']:>2}. {row['team']:<20} {row['final_points']} pts")
        print(f"    ...")
        for _, row in st.tail(3).iterrows():
            print(f"    {row['final_position']:>2}. {row['team']:<20} {row['final_points']} pts")

    # 4. Load FPL player data and compute team signals
    print("\n[4/6] Computing FPL team-level signals from raw player data...")
    all_fpl = []
    for season in SEASONS:
        pdata = load_player_gw_data(season)
        if not pdata.empty:
            signals = compute_team_gw_fpl_signals(pdata)
            all_fpl.append(signals)
            print(f"  {season}: {len(signals)} team-GW records from {len(pdata)} player-GW rows")
        else:
            print(f"  {season}: No player data with team column")

    fpl_signals = pd.concat(all_fpl, ignore_index=True) if all_fpl else pd.DataFrame()
    print(f"  Total FPL team-GW signals: {len(fpl_signals)}")

    # 5. Build checkpoint dataset
    print("\n[5/6] Building checkpoint dataset...")
    checkpoint_df = build_checkpoint_dataset(team_panel, fpl_signals, final_standings)
    print(f"  {len(checkpoint_df)} checkpoint observations")
    print(f"  Checkpoints: {sorted(checkpoint_df['checkpoint'].unique())}")

    # Save
    save_parquet(checkpoint_df, 'season_arc.parquet')

    # --- Feature sets ---
    baseline_features = ['ppg', 'gd_per_game']
    xpoints_features = ['ppg', 'xppg', 'gd_per_game', 'overperf']
    extended_baseline = ['ppg', 'xppg', 'gd_per_game', 'overperf', 'form_trend',
                         'home_ppg', 'away_ppg']

    fpl_features = ['fpl_ownership_slope', 'fpl_transfer_momentum',
                    'fpl_recent_transfer_mom', 'fpl_transfer_accel',
                    'fpl_value_slope', 'fpl_points_per_gw',
                    'fpl_relative_ownership', 'fpl_ict_per_gw',
                    'fpl_threat_per_gw', 'fpl_quality']

    combined_features = extended_baseline + fpl_features

    feature_sets = {
        'PPG Baseline': baseline_features,
        'xPoints Model': xpoints_features,
        'Extended Baseline': extended_baseline,
        'FPL Signals': fpl_features,
        'PPG + FPL': combined_features,
    }

    # --- 6. Run models ---
    print("\n[6/6] Running models...")
    print("\n" + "=" * 70)
    print("FINAL POSITION PREDICTION (Spearman Rank Correlation)")
    print("=" * 70)

    pos_results = {name: {} for name in feature_sets}
    for cp in CHECKPOINTS:
        results = evaluate_position_prediction(checkpoint_df, feature_sets, cp)
        print(f"\n--- Checkpoint GW{cp} ---")
        for name, r in results.items():
            rho = r['spearman_rho']
            mae = r['mae']
            p = r['p_value']
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
            print(f"  {name:<22} rho={rho:.4f}{sig:<4} MAE={mae:.2f}  ({r['n_features']} features)")
            pos_results[name][cp] = r

    # --- Top 4 prediction ---
    print("\n" + "=" * 70)
    print("TOP 4 PREDICTION (Logistic Regression)")
    print("=" * 70)

    top4_results = {name: {} for name in feature_sets}
    for cp in CHECKPOINTS:
        results = evaluate_binary_prediction(checkpoint_df, feature_sets, cp, 'top4')
        print(f"\n--- Checkpoint GW{cp} ---")
        for name, r in results.items():
            print(f"  {name:<22} Brier={r['brier']:.4f}  Acc={r['accuracy']:.3f}  "
                  f"({r['n_pos']}/{r['n_total']} positive)")
            top4_results[name][cp] = r

    # --- Relegation prediction ---
    print("\n" + "=" * 70)
    print("RELEGATION PREDICTION (Logistic Regression)")
    print("=" * 70)

    releg_results = {name: {} for name in feature_sets}
    for cp in CHECKPOINTS:
        results = evaluate_binary_prediction(checkpoint_df, feature_sets, cp, 'relegated')
        print(f"\n--- Checkpoint GW{cp} ---")
        for name, r in results.items():
            print(f"  {name:<22} Brier={r['brier']:.4f}  Acc={r['accuracy']:.3f}  "
                  f"({r['n_pos']}/{r['n_total']} positive)")
            releg_results[name][cp] = r

    # --- Feature importance at GW15 ---
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE (Combined model, GW15, predicting final position)")
    print("=" * 70)

    imp = get_feature_importance(checkpoint_df, combined_features, 'final_position', 15)
    if not imp.empty:
        for _, row in imp.iterrows():
            sign = '+' if row['coefficient'] > 0 else '-'
            kind = 'FPL' if row['feature'].startswith('fpl_') else 'Baseline'
            print(f"  {row['feature']:<30} {sign}{row['abs_importance']:.4f}  [{kind}]")

    # --- Key finding: FPL signal independence ---
    print("\n" + "=" * 70)
    print("FPL SIGNAL CORRELATIONS WITH PPG (GW20)")
    print("=" * 70)

    gw20 = checkpoint_df[checkpoint_df['checkpoint'] == 20]
    if not gw20.empty:
        for feat in fpl_features:
            if feat in gw20.columns:
                valid = gw20[['ppg', feat]].dropna()
                if len(valid) > 5:
                    r, p = stats.pearsonr(valid['ppg'], valid[feat])
                    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
                    print(f"  {feat:<30} r={r:>7.4f} {sig}")

    # --- Ownership slope as standalone predictor ---
    print("\n" + "=" * 70)
    print("OWNERSHIP SLOPE: Season-by-season rank correlation with final position")
    print("=" * 70)

    for cp in [10, 15, 20, 25]:
        print(f"\n  Checkpoint GW{cp}:")
        cp_data = checkpoint_df[
            (checkpoint_df['checkpoint'] == cp) &
            checkpoint_df['fpl_ownership_slope'].notna()
        ]
        for s in sorted(cp_data['season'].unique()):
            sdata = cp_data[cp_data['season'] == s]
            if len(sdata) >= 10:
                # Negative slope = managers dumping → should correlate with HIGH position number
                rho, p = stats.spearmanr(sdata['fpl_ownership_slope'], sdata['final_position'])
                sig = '*' if p < 0.05 else ''
                print(f"    {s}: rho={rho:>7.4f} {sig:<2} (n={len(sdata)})")

    # --- Plots ---
    print("\n" + "=" * 70)
    print("GENERATING CHARTS...")
    print("=" * 70)

    plot_spearman_by_checkpoint(
        pos_results,
        'Final League Position Prediction by Checkpoint',
        str(CHARTS_DIR / 'season_arc_position_prediction.png')
    )

    plot_fpl_signal_value_added(
        pos_results,
        str(CHARTS_DIR / 'season_arc_fpl_value_added.png')
    )

    plot_ownership_trajectory(
        checkpoint_df,
        str(CHARTS_DIR / 'season_arc_ownership_trajectory.png')
    )

    plot_binary_brier(
        top4_results, releg_results,
        str(CHARTS_DIR / 'season_arc_binary_outcomes.png')
    )

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Find best checkpoint for FPL improvement
    best_improvement = 0
    best_cp = None
    for cp in CHECKPOINTS:
        baseline_rho = pos_results.get('PPG Baseline', {}).get(cp, {}).get('spearman_rho', 0)
        combined_rho = pos_results.get('PPG + FPL', {}).get(cp, {}).get('spearman_rho', 0)
        improvement = combined_rho - baseline_rho
        if improvement > best_improvement:
            best_improvement = improvement
            best_cp = cp

    print(f"\n  Best FPL improvement over PPG: +{best_improvement:.4f} rho at GW{best_cp}")

    # Compare FPL-only vs PPG at early checkpoints
    for cp in [5, 10, 15]:
        ppg_rho = pos_results.get('PPG Baseline', {}).get(cp, {}).get('spearman_rho', 0)
        fpl_rho = pos_results.get('FPL Signals', {}).get(cp, {}).get('spearman_rho', 0)
        print(f"  GW{cp:>2}: PPG={ppg_rho:.4f}  FPL={fpl_rho:.4f}  "
              f"{'FPL better' if fpl_rho > ppg_rho else 'PPG better'}")

    print("\n  Done. All results saved to data/processed/season_arc.parquet")
    print("  Charts saved to outputs/charts/season_arc_*.png")


if __name__ == '__main__':
    main()
