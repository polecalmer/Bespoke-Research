"""Backfill the live tracker with 2025-26 historical data (GW1 through latest).

Downloads 2025-26 FPL data from vaastav/Fantasy-Premier-League, processes it
through the same signal pipeline as the research, generates predictions using
pre-trained models, and populates the live tracker's scorecard.

Usage:
    python src/backfill_2025_26.py

This produces:
    - data/live/2025-26/predictions.csv (all match predictions + results)
    - data/live/2025-26/scorecard.json (cumulative accuracy metrics)
    - data/live/2025-26/snapshots/gw*_ownership.json (for velocity calc)
    - outputs/charts/live_tracker.png
"""

import sys
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    setup_chart_style, CHART_COLORS, CHARTS_DIR, MODEL_DIR,
    DATA_PROCESSED, DATA_RAW_FPL, DATA_RAW_ODDS,
    FPL_TO_ODDS, ELEMENT_TYPE_MAP,
)

# --- Constants ---
SEASON = '2025-26'
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LIVE_DIR = PROJECT_ROOT / 'data' / 'live' / SEASON
SNAPSHOTS_DIR = LIVE_DIR / 'snapshots'

FPL_FEATURES = ['transfer_ratio', 'ownership_ratio', 'captain_proxy',
                'dcs_ratio', 'velocity_delta']
ODDS_FEATURES = ['implied_prob_h', 'implied_prob_d', 'implied_prob_a']
ALL_FEATURES = ODDS_FEATURES + FPL_FEATURES

RESULT_MAP = {'A': 0, 'D': 1, 'H': 2}
RESULT_LABELS = {0: 'A', 1: 'D', 2: 'H'}


def load_models():
    """Load pre-trained models and scaler."""
    models = {}
    for name in ['fpl_only', 'odds_only', 'fpl_odds']:
        path = MODEL_DIR / f'model_{name}.pkl'
        with open(path, 'rb') as f:
            models[name] = pickle.load(f)
    with open(MODEL_DIR / 'scaler.pkl', 'rb') as f:
        models['scaler'] = pickle.load(f)
    return models


def multiclass_brier(y_true, y_proba, n_classes=3):
    """Mean Brier score across classes."""
    from sklearn.metrics import brier_score_loss
    return np.mean([
        brier_score_loss((y_true == c).astype(int), y_proba[:, c])
        for c in range(n_classes)
    ])


def build_matches():
    """Build match-level dataset from 2025-26 raw data."""
    print("\n[1/5] Loading raw data...")

    # Load player-GW data
    gw_path = DATA_RAW_FPL / f'{SEASON}_merged_gw.csv'
    gw_df = pd.read_csv(gw_path, low_memory=False)
    print(f"  Player-GW rows: {len(gw_df)}")

    # Load teams
    teams_path = DATA_RAW_FPL / f'{SEASON}_teams.csv'
    teams = pd.read_csv(teams_path)
    team_id_to_name = dict(zip(teams['id'], teams['name']))
    print(f"  Teams: {len(teams)}")

    # Load fixtures
    fix_path = DATA_RAW_FPL / f'{SEASON}_fixtures.csv'
    fixtures = pd.read_csv(fix_path)
    finished = fixtures[fixtures['finished'] == True].copy()
    print(f"  Finished fixtures: {len(finished)}")

    # Standardize column names
    col_map = {}
    for c in gw_df.columns:
        cl = c.lower().strip()
        if cl == 'gw':
            col_map[c] = 'gw'
        elif cl == 'element':
            col_map[c] = 'element'
        elif cl == 'selected':
            col_map[c] = 'selected'
        elif cl == 'transfers_in':
            col_map[c] = 'transfers_in'
        elif cl == 'transfers_out':
            col_map[c] = 'transfers_out'
        elif cl == 'value':
            col_map[c] = 'value'
        elif cl == 'was_home':
            col_map[c] = 'was_home'
        elif cl == 'opponent_team':
            col_map[c] = 'opponent_team'
        elif cl == 'team':
            col_map[c] = 'team'
        elif cl == 'fixture':
            col_map[c] = 'fixture'
        elif cl == 'position':
            col_map[c] = 'position'
    gw_df = gw_df.rename(columns=col_map)

    # Map team names to IDs
    name_to_id = {v: k for k, v in team_id_to_name.items()}
    if gw_df['team'].dtype.kind in ('U', 'O') or gw_df['team'].dtype.name in ('str', 'string', 'object'):
        gw_df['team_id'] = gw_df['team'].map(name_to_id)
    else:
        gw_df['team_id'] = gw_df['team']

    # Position mapping
    pos_map = {'GK': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}
    if 'position' in gw_df.columns and gw_df['position'].dtype.kind in ('U', 'O') or (
            'position' in gw_df.columns and gw_df['position'].dtype.name in ('str', 'string', 'object')):
        gw_df['element_type'] = gw_df['position'].map(pos_map)
    else:
        gw_df['element_type'] = 0

    # Load odds (if available)
    odds_path = DATA_RAW_ODDS / f'{SEASON}.csv'
    odds_df = None
    if odds_path.exists():
        try:
            odds_df = pd.read_csv(odds_path, encoding='utf-8', on_bad_lines='skip')
            print(f"  Odds matches: {len(odds_df)}")
        except Exception:
            try:
                odds_df = pd.read_csv(odds_path, encoding='latin-1', on_bad_lines='skip')
            except Exception:
                pass

    return gw_df, fixtures, finished, team_id_to_name, odds_df


def compute_fixture_signals(gw_df, fixture_row, team_id_to_name,
                            prev_ownership=None):
    """Compute FPL signals for a single fixture."""
    gw = fixture_row['event']
    team_h_id = fixture_row['team_h']
    team_a_id = fixture_row['team_a']

    # Get players in this GW
    gw_players = gw_df[gw_df['gw'] == gw].copy()

    home_players = gw_players[gw_players['team_id'] == team_h_id]
    away_players = gw_players[gw_players['team_id'] == team_a_id]

    if len(home_players) == 0 or len(away_players) == 0:
        return None

    def team_agg(players):
        selected = players['selected'].sum() if 'selected' in players.columns else 0
        net_transfers = (players['transfers_in'] - players['transfers_out']).sum()
        value = players['value'].fillna(50) if 'value' in players.columns else pd.Series([50] * len(players))
        price_weighted = (players['selected'] * value / 10).sum()
        def_gk_mask = players['element_type'].isin([1, 2])
        def_gk_selected = players.loc[def_gk_mask, 'selected'].sum()
        return {
            'total_selected': float(selected),
            'net_transfers': float(net_transfers),
            'price_weighted_selected': float(price_weighted),
            'def_gk_selected': float(def_gk_selected),
        }

    h = team_agg(home_players)
    a = team_agg(away_players)

    # Transfer ratio
    abs_sum = abs(h['net_transfers']) + abs(a['net_transfers']) + 1
    transfer_ratio = h['net_transfers'] / abs_sum

    # Ownership ratio (price-weighted)
    ownership_ratio = h['price_weighted_selected'] / (
        h['price_weighted_selected'] + a['price_weighted_selected'] + 1)

    # Captain proxy
    captain_proxy = (h['net_transfers'] / (h['total_selected'] + 1) -
                     a['net_transfers'] / (a['total_selected'] + 1))

    # Defensive confidence ratio
    dcs_ratio = h['def_gk_selected'] / (h['def_gk_selected'] + a['def_gk_selected'] + 1)

    # Ownership velocity
    velocity_delta = 0.0
    if prev_ownership is not None:
        ov_h = h['total_selected'] - prev_ownership.get(team_h_id, h['total_selected'])
        ov_a = a['total_selected'] - prev_ownership.get(team_a_id, a['total_selected'])
        velocity_delta = ov_h - ov_a

    return {
        'transfer_ratio': transfer_ratio,
        'ownership_ratio': ownership_ratio,
        'captain_proxy': captain_proxy,
        'dcs_ratio': dcs_ratio,
        'velocity_delta': velocity_delta,
        'total_selected_h': h['total_selected'],
        'total_selected_a': a['total_selected'],
        'net_transfers_h': h['net_transfers'],
        'net_transfers_a': a['net_transfers'],
    }


def build_ownership_snapshots(gw_df, team_id_to_name):
    """Build per-GW team ownership totals for velocity calculation."""
    ownership = {}
    for gw in sorted(gw_df['gw'].unique()):
        gw_data = gw_df[gw_df['gw'] == gw]
        gw_ownership = {}
        for team_id in team_id_to_name:
            team_players = gw_data[gw_data['team_id'] == team_id]
            gw_ownership[team_id] = float(team_players['selected'].sum())
        ownership[gw] = gw_ownership
    return ownership


def main():
    print("=" * 60)
    print(f"BACKFILL: {SEASON} Live Tracker Data")
    print("=" * 60)

    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Build match data
    gw_df, all_fixtures, finished, team_id_to_name, odds_df = build_matches()

    # 2. Build ownership snapshots
    print("\n[2/5] Building ownership snapshots...")
    ownership_by_gw = build_ownership_snapshots(gw_df, team_id_to_name)
    for gw, own in ownership_by_gw.items():
        with open(SNAPSHOTS_DIR / f'gw{gw}_ownership.json', 'w') as f:
            json.dump({str(k): v for k, v in own.items()}, f)
    print(f"  Saved {len(ownership_by_gw)} GW ownership snapshots")

    # 3. Build odds lookup
    print("\n[3/5] Building odds lookup...")
    odds_lookup = {}
    if odds_df is not None and 'HomeTeam' in odds_df.columns:
        for _, row in odds_df.iterrows():
            key = (str(row.get('HomeTeam', '')), str(row.get('AwayTeam', '')))
            if 'B365H' in row.index:
                odds_lookup[key] = {
                    'B365H': row['B365H'], 'B365D': row['B365D'], 'B365A': row['B365A']
                }
        print(f"  {len(odds_lookup)} matches with odds")
    else:
        print("  No odds data available — using base rate fallback")
        print("  TIP: Download odds from football-data.co.uk/mmz4281/2526/E0.csv")
        print(f"  Save to: {DATA_RAW_ODDS / f'{SEASON}.csv'}")

    # 4. Process each finished fixture
    print("\n[4/5] Computing signals and predictions...")
    models = load_models()
    scaler = models['scaler']

    match_rows = []
    for _, fix in finished.iterrows():
        gw = fix['event']
        team_h_id = fix['team_h']
        team_a_id = fix['team_a']
        team_h_name = team_id_to_name.get(team_h_id, f'Team{team_h_id}')
        team_a_name = team_id_to_name.get(team_a_id, f'Team{team_a_id}')

        # Get previous GW ownership for velocity
        prev_ownership = ownership_by_gw.get(gw - 1) if gw > 1 else None

        # Compute signals
        signals = compute_fixture_signals(gw_df, fix, team_id_to_name, prev_ownership)
        if signals is None:
            continue

        # Look up odds
        h_odds_name = FPL_TO_ODDS.get(team_h_name, team_h_name)
        a_odds_name = FPL_TO_ODDS.get(team_a_name, team_a_name)
        odds = odds_lookup.get((h_odds_name, a_odds_name))

        if odds and not pd.isna(odds.get('B365H')):
            b365h, b365d, b365a = float(odds['B365H']), float(odds['B365D']), float(odds['B365A'])
            overround = 1/b365h + 1/b365d + 1/b365a
            implied_h = (1/b365h) / overround
            implied_d = (1/b365d) / overround
            implied_a = (1/b365a) / overround
        else:
            b365h = b365d = b365a = None
            implied_h, implied_d, implied_a = 0.45, 0.27, 0.28

        # Determine result
        h_score = fix.get('team_h_score', None)
        a_score = fix.get('team_a_score', None)
        if pd.notna(h_score) and pd.notna(a_score):
            if h_score > a_score:
                result = 'H'
            elif a_score > h_score:
                result = 'A'
            else:
                result = 'D'
        else:
            result = None

        row = {
            'season': SEASON,
            'gw': int(gw),
            'fixture_id': int(fix['id']),
            'kickoff_time': fix.get('kickoff_time', ''),
            'team_h_id': int(team_h_id),
            'team_h_name': team_h_name,
            'team_a_id': int(team_a_id),
            'team_a_name': team_a_name,
            'team_h_score': h_score,
            'team_a_score': a_score,
            'result': result,
            'B365H': b365h,
            'B365D': b365d,
            'B365A': b365a,
            'implied_prob_h': implied_h,
            'implied_prob_d': implied_d,
            'implied_prob_a': implied_a,
            **signals,
        }

        # Generate predictions
        feat_df = pd.DataFrame([{f: row.get(f, 0) for f in ALL_FEATURES}])
        feat_scaled = pd.DataFrame(
            scaler.transform(feat_df[ALL_FEATURES]),
            columns=ALL_FEATURES
        )

        for model_name in ['fpl_only', 'odds_only', 'fpl_odds']:
            model = models[model_name]
            if model_name == 'fpl_only':
                feats = FPL_FEATURES
            elif model_name == 'odds_only':
                feats = ODDS_FEATURES
            else:
                feats = ALL_FEATURES

            proba = model.predict_proba(feat_scaled[feats].values)
            row[f'p_{model_name}_A'] = proba[0, 0]
            row[f'p_{model_name}_D'] = proba[0, 1]
            row[f'p_{model_name}_H'] = proba[0, 2]

        # Predicted favorites
        fpl_probs = [row['p_fpl_only_A'], row['p_fpl_only_D'], row['p_fpl_only_H']]
        row['crowd_favorite'] = RESULT_LABELS[np.argmax(fpl_probs)]
        odds_probs = [row['p_odds_only_A'], row['p_odds_only_D'], row['p_odds_only_H']]
        row['odds_favorite'] = RESULT_LABELS[np.argmax(odds_probs)]

        match_rows.append(row)

    predictions = pd.DataFrame(match_rows)
    print(f"  Processed {len(predictions)} matches across GW{predictions['gw'].min()}-{predictions['gw'].max()}")

    # Save predictions
    predictions.to_csv(LIVE_DIR / 'predictions.csv', index=False)
    print(f"  Saved predictions.csv")

    # 5. Build scorecard
    print("\n[5/5] Building scorecard...")
    completed = predictions.dropna(subset=['result']).copy()
    y_true = completed['result'].map(RESULT_MAP).values

    has_odds = completed['B365H'].notna().sum()
    print(f"  Matches with results: {len(completed)}")
    print(f"  Matches with odds: {has_odds}")

    scorecard = {
        'season': SEASON,
        'backfilled': True,
        'matches_predicted': int(len(completed)),
        'matches_with_odds': int(has_odds),
        'gameweeks_covered': sorted(completed['gw'].unique().tolist()),
    }

    # Score each model
    for model_name in ['fpl_only', 'odds_only', 'fpl_odds']:
        cols = [f'p_{model_name}_A', f'p_{model_name}_D', f'p_{model_name}_H']
        proba = completed[cols].values
        pred_class = proba.argmax(axis=1)

        brier = multiclass_brier(y_true, proba)
        accuracy = (pred_class == y_true).mean()
        correct = (pred_class == y_true).sum()

        scorecard[model_name] = {
            'brier': round(float(brier), 4),
            'accuracy': round(float(accuracy), 4),
            'correct': int(correct),
            'total': int(len(completed)),
        }

    # Crowd vs Odds comparison
    agree = (completed['crowd_favorite'] == completed['odds_favorite']).sum()
    scorecard['agreement_rate'] = round(float(agree / len(completed) * 100), 1)

    disagree_mask = completed['crowd_favorite'] != completed['odds_favorite']
    if disagree_mask.sum() > 0:
        crowd_right = (completed.loc[disagree_mask, 'crowd_favorite'] ==
                       completed.loc[disagree_mask, 'result']).sum()
        odds_right = (completed.loc[disagree_mask, 'odds_favorite'] ==
                      completed.loc[disagree_mask, 'result']).sum()
        scorecard['disagreement'] = {
            'n_matches': int(disagree_mask.sum()),
            'crowd_correct': int(crowd_right),
            'odds_correct': int(odds_right),
        }

    # Brier Skill Score
    naive_brier = 0.2154  # from training data
    fpl_brier = scorecard['fpl_only']['brier']
    odds_brier = scorecard['odds_only']['brier']
    fpl_skill = 1 - fpl_brier / naive_brier
    odds_skill = 1 - odds_brier / naive_brier
    crowd_pct = (fpl_skill / odds_skill * 100) if odds_skill > 0 else 0
    scorecard['crowd_pct_of_odds_skill'] = round(float(crowd_pct), 1)
    scorecard['fpl_bss_vs_naive'] = round(float(fpl_skill * 100), 1)
    scorecard['odds_bss_vs_naive'] = round(float(odds_skill * 100), 1)

    # Per-GW breakdown
    gw_scores = []
    for gw in sorted(completed['gw'].unique()):
        gw_data = completed[completed['gw'] == gw]
        gw_y = gw_data['result'].map(RESULT_MAP).values
        gw_entry = {'gw': int(gw), 'n_matches': int(len(gw_data))}

        for mn in ['fpl_only', 'odds_only']:
            cols = [f'p_{mn}_A', f'p_{mn}_D', f'p_{mn}_H']
            proba = gw_data[cols].values
            gw_entry[f'{mn}_brier'] = round(float(multiclass_brier(gw_y, proba)), 4)
            gw_entry[f'{mn}_accuracy'] = round(float((proba.argmax(axis=1) == gw_y).mean()), 4)

        gw_scores.append(gw_entry)
    scorecard['per_gameweek'] = gw_scores

    with open(LIVE_DIR / 'scorecard.json', 'w') as f:
        json.dump(scorecard, f, indent=2)

    # --- Print results ---
    print("\n" + "=" * 60)
    print(f"BACKFILL SCORECARD — {SEASON} (GW1-{predictions['gw'].max()})")
    print("=" * 60)

    print(f"\n{'Model':<20} {'Brier':>8} {'Accuracy':>10} {'Correct':>10}")
    print("-" * 52)
    for mn in ['fpl_only', 'odds_only', 'fpl_odds']:
        m = scorecard[mn]
        label = {'fpl_only': 'FPL Crowd', 'odds_only': 'Bookmaker',
                 'fpl_odds': 'Combined'}[mn]
        print(f"  {label:<18} {m['brier']:>8.4f} {m['accuracy']:>9.1%} "
              f"{m['correct']}/{m['total']}")

    print(f"\n  Crowd captures {scorecard['crowd_pct_of_odds_skill']}% of bookmaker skill")
    print(f"  (Historical benchmark: 72%)")
    print(f"  FPL BSS vs naive: {scorecard['fpl_bss_vs_naive']}%")
    print(f"  Odds BSS vs naive: {scorecard['odds_bss_vs_naive']}%")
    print(f"\n  Crowd-Bookmaker agreement: {scorecard['agreement_rate']}%")
    if 'disagreement' in scorecard:
        d = scorecard['disagreement']
        print(f"  On {d['n_matches']} disagreements: "
              f"Crowd right {d['crowd_correct']}, Odds right {d['odds_correct']}")

    print(f"\n  Per-Gameweek Brier Scores:")
    print(f"  {'GW':>4} {'Matches':>8} {'FPL':>10} {'Odds':>10} {'FPL Acc':>8} {'Odds Acc':>9} {'Winner':>8}")
    print(f"  {'-'*4} {'-'*8} {'-'*10} {'-'*10} {'-'*8} {'-'*9} {'-'*8}")
    fpl_wins = 0
    odds_wins = 0
    for gw in gw_scores:
        fb = gw.get('fpl_only_brier', None)
        ob = gw.get('odds_only_brier', None)
        fa = gw.get('fpl_only_accuracy', None)
        oa = gw.get('odds_only_accuracy', None)
        winner = ''
        if fb is not None and ob is not None:
            if fb < ob:
                winner = 'CROWD'
                fpl_wins += 1
            elif ob < fb:
                winner = 'ODDS'
                odds_wins += 1
            else:
                winner = 'TIE'
        print(f"  {gw['gw']:>4} {gw['n_matches']:>8} {fb:>10.4f} {ob:>10.4f} "
              f"{fa:>7.1%} {oa:>8.1%} {winner:>8}")

    print(f"\n  GW wins: Crowd {fpl_wins}, Odds {odds_wins}")

    # --- Generate chart ---
    print("\n  Generating live tracker chart...")
    generate_chart(gw_scores, scorecard)

    print(f"\n  All data saved to {LIVE_DIR}/")
    print("=" * 60)


def generate_chart(gw_scores, scorecard):
    """Generate the live tracking visualization."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    setup_chart_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    TEAL = CHART_COLORS['teal']
    CORAL = CHART_COLORS['coral']
    WHITE = CHART_COLORS['white']

    gws = [g['gw'] for g in gw_scores]
    fpl_briers = [g['fpl_only_brier'] for g in gw_scores]
    odds_briers = [g['odds_only_brier'] for g in gw_scores]
    fpl_accs = [g['fpl_only_accuracy'] for g in gw_scores]
    odds_accs = [g['odds_only_accuracy'] for g in gw_scores]

    # Panel 1: Cumulative Brier
    ax = axes[0, 0]
    fpl_cum = np.cumsum(fpl_briers) / np.arange(1, len(fpl_briers) + 1)
    odds_cum = np.cumsum(odds_briers) / np.arange(1, len(odds_briers) + 1)
    ax.plot(gws, fpl_cum, 'o-', color=TEAL, markersize=5, label='FPL Crowd')
    ax.plot(gws, odds_cum, 's-', color=CORAL, markersize=5, label='Bookmaker')
    ax.axhline(y=0.2154, color=WHITE, linestyle='--', alpha=0.3, label='Naive baseline')
    ax.set_xlabel('Gameweek')
    ax.set_ylabel('Cumulative Avg Brier Score')
    ax.set_title('Cumulative Prediction Accuracy\n(lower = better)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    # Panel 2: Per-GW Brier bars
    ax = axes[0, 1]
    x = np.arange(len(gws))
    w = 0.35
    ax.bar(x - w/2, fpl_briers, w, color=TEAL, label='FPL Crowd', edgecolor='#444444')
    ax.bar(x + w/2, odds_briers, w, color=CORAL, label='Bookmaker', edgecolor='#444444')
    ax.set_xticks(x[::2])
    ax.set_xticklabels([f'GW{g}' for g in gws[::2]], fontsize=9, rotation=45)
    ax.set_ylabel('Brier Score')
    ax.set_title('Per-Gameweek Brier Score', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.2)

    # Panel 3: Running crowd % of odds skill
    ax = axes[1, 0]
    naive_b = 0.2154
    running_pct = []
    for i in range(len(fpl_briers)):
        fpl_avg = np.mean(fpl_briers[:i+1])
        odds_avg = np.mean(odds_briers[:i+1])
        fpl_s = 1 - fpl_avg / naive_b
        odds_s = 1 - odds_avg / naive_b
        pct = (fpl_s / odds_s * 100) if odds_s > 0 else 0
        running_pct.append(pct)

    ax.plot(gws, running_pct, 'o-', color=TEAL, markersize=5, linewidth=2)
    ax.axhline(y=72, color=WHITE, linestyle='--', alpha=0.4, label='Historical benchmark (72%)')
    ax.set_xlabel('Gameweek')
    ax.set_ylabel('% of Bookmaker Skill')
    ax.set_title('Crowd Wisdom Ratio Over Time\n(crowd as % of bookmaker accuracy)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    ax.set_ylim(0, max(max(running_pct) * 1.1, 100))

    # Panel 4: Per-GW accuracy
    ax = axes[1, 1]
    ax.plot(gws, [a * 100 for a in fpl_accs], 'o-', color=TEAL, markersize=5, label='FPL Crowd')
    ax.plot(gws, [a * 100 for a in odds_accs], 's-', color=CORAL, markersize=5, label='Bookmaker')
    ax.axhline(y=33.3, color=WHITE, linestyle='--', alpha=0.3, label='Random (33%)')
    ax.set_xlabel('Gameweek')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Gameweek Prediction Accuracy', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    fig.suptitle(f'Live Out-of-Sample Tracking: {SEASON}\n'
                 f'Crowd captures {scorecard["crowd_pct_of_odds_skill"]}% of bookmaker skill '
                 f'({len(gw_scores)} GWs, {scorecard["matches_predicted"]} matches)',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS_DIR / 'live_tracker.png', bbox_inches='tight')
    plt.close()
    print(f"  Saved live_tracker.png")


if __name__ == '__main__':
    main()
