"""Live FPL Crowd Wisdom Tracker for the 2025-26 EPL season.

Run weekly before the FPL deadline to:
1. Pull current FPL ownership/transfer data from the FPL API
2. Pull current match odds from football-data.co.uk
3. Compute FPL signals and generate match predictions
4. After matches complete, record results and score predictions
5. Maintain a cumulative scorecard comparing Crowd vs Bookmaker

Usage:
    # Snapshot before gameweek deadline (captures FPL + odds data)
    python live_tracker.py snapshot

    # Record results after matches finish
    python live_tracker.py results

    # View current season scorecard
    python live_tracker.py scorecard

    # Run all steps (snapshot + results for any completed GWs)
    python live_tracker.py update

Data stored in: data/live/2025-26/
"""

import sys
import json
import time
import pickle
import argparse
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    setup_chart_style, CHART_COLORS, CHARTS_DIR, MODEL_DIR,
    DATA_PROCESSED, FPL_TO_ODDS, ODDS_TO_FPL,
    ELEMENT_TYPE_MAP,
)

# --- Constants ---
SEASON = '2025-26'
SEASON_CODE = '2526'
FPL_API_BASE = 'https://fantasy.premierleague.com/api'
ODDS_URL = f'https://www.football-data.co.uk/mmz4281/{SEASON_CODE}/E0.csv'

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LIVE_DIR = PROJECT_ROOT / 'data' / 'live' / SEASON
SNAPSHOTS_DIR = LIVE_DIR / 'snapshots'
RESULTS_FILE = LIVE_DIR / 'results.csv'
SCORECARD_FILE = LIVE_DIR / 'scorecard.json'
PREDICTIONS_FILE = LIVE_DIR / 'predictions.csv'

# FPL signal features (must match training)
FPL_FEATURES = ['transfer_ratio', 'ownership_ratio', 'captain_proxy',
                'dcs_ratio', 'velocity_delta']
ODDS_FEATURES = ['implied_prob_h', 'implied_prob_d', 'implied_prob_a']
ALL_FEATURES = ODDS_FEATURES + FPL_FEATURES

RESULT_MAP = {'A': 0, 'D': 1, 'H': 2}
RESULT_LABELS = {0: 'A', 1: 'D', 2: 'H'}

# Timeout for API calls
TIMEOUT = 30


# =========================================================================
# FPL API Data Fetching
# =========================================================================

def fetch_json(url: str, retries: int = 3) -> dict | None:
    """Fetch JSON from URL with retries and exponential backoff."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=TIMEOUT, headers={
                'User-Agent': 'FPL-Crowd-Wisdom-Tracker/1.0'
            })
            if resp.status_code == 200:
                return resp.json()
            print(f"  HTTP {resp.status_code} for {url}")
        except requests.RequestException as e:
            print(f"  Attempt {attempt + 1}/{retries} failed: {e}")
        if attempt < retries - 1:
            time.sleep(2 ** (attempt + 1))
    return None


def fetch_bootstrap() -> dict | None:
    """Fetch FPL bootstrap-static data (all players, teams, gameweeks)."""
    print("Fetching FPL bootstrap data...")
    data = fetch_json(f'{FPL_API_BASE}/bootstrap-static/')
    if data:
        print(f"  Got {len(data.get('elements', []))} players, "
              f"{len(data.get('teams', []))} teams, "
              f"{len(data.get('events', []))} gameweeks")
    return data


def fetch_fixtures() -> list | None:
    """Fetch all FPL fixtures for the season."""
    print("Fetching fixtures...")
    data = fetch_json(f'{FPL_API_BASE}/fixtures/')
    if data:
        print(f"  Got {len(data)} fixtures")
    return data


def fetch_live_gw(event_id: int) -> dict | None:
    """Fetch live data for a specific gameweek."""
    return fetch_json(f'{FPL_API_BASE}/event/{event_id}/live/')


def fetch_odds() -> pd.DataFrame | None:
    """Fetch current season odds from football-data.co.uk."""
    print("Fetching odds data...")
    try:
        resp = requests.get(ODDS_URL, timeout=TIMEOUT)
        if resp.status_code == 200:
            from io import StringIO
            df = pd.read_csv(StringIO(resp.text), on_bad_lines='skip')
            print(f"  Got {len(df)} matches with odds")
            return df
        print(f"  HTTP {resp.status_code} for odds")
    except Exception as e:
        print(f"  Failed to fetch odds: {e}")
    return None


# =========================================================================
# FPL Team Name Mapping for 2025-26
# =========================================================================

def build_team_map(bootstrap: dict) -> dict:
    """Build team_id -> team_name mapping from bootstrap data."""
    teams = bootstrap.get('teams', [])
    return {t['id']: t['short_name'] for t in teams}


def build_team_name_map(bootstrap: dict) -> dict:
    """Build team_id -> full team name for odds matching."""
    teams = bootstrap.get('teams', [])
    fpl_names = {}
    for t in teams:
        # Use short_name for FPL_TO_ODDS lookup
        fpl_names[t['id']] = t['short_name']
    return fpl_names


# =========================================================================
# Signal Computation (matching 03_engineer_signals.py exactly)
# =========================================================================

def compute_match_signals(
    home_players: pd.DataFrame,
    away_players: pd.DataFrame,
    prev_gw_ownership: dict | None = None,
) -> dict:
    """Compute the 5 FPL signals for a single match.

    home_players / away_players: DataFrame with columns:
        element, team, element_type, now_cost, selected_by_percent,
        transfers_in_event, transfers_out_event

    prev_gw_ownership: optional dict {team_id: total_selected} from previous GW
    """
    def team_agg(players):
        total_selected = (players['selected_by_percent'] * 100).sum()  # proxy for count
        net_transfers = (players['transfers_in_event'] - players['transfers_out_event']).sum()
        price_weighted = (players['selected_by_percent'] * players['now_cost'] / 10).sum()
        def_gk = players[players['element_type'].isin([1, 2])]['selected_by_percent'].sum() * 100
        return {
            'total_selected': total_selected,
            'net_transfers': net_transfers,
            'price_weighted_selected': price_weighted,
            'def_gk_selected': def_gk,
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

    # Defensive confidence (DCS) ratio
    dcs_ratio = h['def_gk_selected'] / (h['def_gk_selected'] + a['def_gk_selected'] + 1)

    # Ownership velocity (needs previous GW data)
    velocity_delta = 0.0
    if prev_gw_ownership:
        h_team = home_players['team'].iloc[0] if len(home_players) > 0 else 0
        a_team = away_players['team'].iloc[0] if len(away_players) > 0 else 0
        ov_h = h['total_selected'] - prev_gw_ownership.get(h_team, h['total_selected'])
        ov_a = a['total_selected'] - prev_gw_ownership.get(a_team, a['total_selected'])
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


# =========================================================================
# Prediction Generation
# =========================================================================

def load_models():
    """Load pre-trained models and scaler."""
    models = {}
    for name in ['fpl_only', 'odds_only', 'fpl_odds']:
        path = MODEL_DIR / f'model_{name}.pkl'
        if path.exists():
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)
    scaler_path = MODEL_DIR / 'scaler.pkl'
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            models['scaler'] = pickle.load(f)
    return models


def generate_predictions(matches: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Generate probability predictions for upcoming matches."""
    scaler = models.get('scaler')
    if scaler is None:
        print("  WARNING: No scaler found, using unscaled features")

    results = []
    for _, match in matches.iterrows():
        row = match.to_dict()

        features = {f: match.get(f, 0) for f in ALL_FEATURES}
        feat_df = pd.DataFrame([features])

        if scaler is not None:
            feat_scaled = pd.DataFrame(
                scaler.transform(feat_df[ALL_FEATURES]),
                columns=ALL_FEATURES
            )
        else:
            feat_scaled = feat_df

        for model_name in ['fpl_only', 'odds_only', 'fpl_odds']:
            model = models.get(model_name)
            if model is None:
                continue
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

        # Crowd prediction: FPL model's predicted favorite
        if 'p_fpl_only_H' in row:
            fpl_probs = [row['p_fpl_only_A'], row['p_fpl_only_D'], row['p_fpl_only_H']]
            row['crowd_favorite'] = RESULT_LABELS[np.argmax(fpl_probs)]
        if 'p_odds_only_H' in row:
            odds_probs = [row['p_odds_only_A'], row['p_odds_only_D'], row['p_odds_only_H']]
            row['odds_favorite'] = RESULT_LABELS[np.argmax(odds_probs)]

        results.append(row)

    return pd.DataFrame(results)


# =========================================================================
# Snapshot: Pre-deadline data capture
# =========================================================================

def snapshot(args=None):
    """Capture FPL ownership + odds data before the GW deadline."""
    print("=" * 60)
    print(f"LIVE SNAPSHOT — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Fetch FPL data
    bootstrap = fetch_bootstrap()
    if bootstrap is None:
        print("ERROR: Could not fetch FPL bootstrap data")
        return

    fixtures = fetch_fixtures()
    if fixtures is None:
        print("ERROR: Could not fetch fixtures")
        return

    # Find current gameweek
    events = bootstrap['events']
    current_gw = None
    for ev in events:
        if ev['is_current']:
            current_gw = ev['id']
            break
    if current_gw is None:
        # Find next unfinished
        for ev in events:
            if not ev['finished']:
                current_gw = ev['id']
                break

    if current_gw is None:
        print("ERROR: Could not determine current gameweek")
        return

    print(f"\nCurrent gameweek: GW{current_gw}")

    # 2. Build player DataFrame
    elements = bootstrap['elements']
    players = pd.DataFrame(elements)
    team_map = build_team_map(bootstrap)
    team_name_map = build_team_name_map(bootstrap)

    # 3. Find upcoming fixtures for this gameweek
    gw_fixtures = [f for f in fixtures if f.get('event') == current_gw]
    print(f"Fixtures in GW{current_gw}: {len(gw_fixtures)}")

    # 4. Load previous GW ownership for velocity calculation
    prev_gw_file = SNAPSHOTS_DIR / f'gw{current_gw - 1}_ownership.json'
    prev_gw_ownership = None
    if prev_gw_file.exists():
        with open(prev_gw_file) as f:
            prev_gw_ownership = {int(k): v for k, v in json.load(f).items()}

    # 5. Save current ownership snapshot (for next GW's velocity)
    current_ownership = {}
    for team_id in team_map:
        team_players = players[players['team'] == team_id]
        current_ownership[team_id] = float(
            (team_players['selected_by_percent'].astype(float) * 100).sum())
    with open(SNAPSHOTS_DIR / f'gw{current_gw}_ownership.json', 'w') as f:
        json.dump(current_ownership, f)

    # 6. Fetch odds
    odds_df = fetch_odds()

    # 7. Compute signals for each fixture
    match_rows = []
    for fix in gw_fixtures:
        team_h_id = fix['team_h']
        team_a_id = fix['team_a']
        team_h_name = team_map.get(team_h_id, f'Team{team_h_id}')
        team_a_name = team_map.get(team_a_id, f'Team{team_a_id}')

        home_players = players[players['team'] == team_h_id].copy()
        away_players = players[players['team'] == team_a_id].copy()

        signals = compute_match_signals(home_players, away_players, prev_gw_ownership)

        # Match odds (if available)
        implied_h, implied_d, implied_a = 0.45, 0.27, 0.28  # fallback base rates
        b365h, b365d, b365a = None, None, None

        if odds_df is not None:
            # Map FPL team names to odds names
            h_odds_name = FPL_TO_ODDS.get(team_h_name, team_h_name)
            a_odds_name = FPL_TO_ODDS.get(team_a_name, team_a_name)

            odds_match = odds_df[
                (odds_df['HomeTeam'] == h_odds_name) &
                (odds_df['AwayTeam'] == a_odds_name)
            ]
            if len(odds_match) == 0:
                # Try reverse lookup with full names
                full_map = build_team_name_map(bootstrap)
                h_full = next((t['name'] for t in bootstrap['teams'] if t['id'] == team_h_id), '')
                a_full = next((t['name'] for t in bootstrap['teams'] if t['id'] == team_a_id), '')
                h_odds_full = FPL_TO_ODDS.get(h_full, h_full)
                a_odds_full = FPL_TO_ODDS.get(a_full, a_full)
                odds_match = odds_df[
                    (odds_df['HomeTeam'] == h_odds_full) &
                    (odds_df['AwayTeam'] == a_odds_full)
                ]

            if len(odds_match) > 0:
                last = odds_match.iloc[-1]  # Most recent if multiple
                b365h = float(last['B365H']) if 'B365H' in last.index else None
                b365d = float(last['B365D']) if 'B365D' in last.index else None
                b365a = float(last['B365A']) if 'B365A' in last.index else None

                if b365h and b365d and b365a:
                    overround = 1/b365h + 1/b365d + 1/b365a
                    implied_h = (1/b365h) / overround
                    implied_d = (1/b365d) / overround
                    implied_a = (1/b365a) / overround

        match_rows.append({
            'season': SEASON,
            'gw': current_gw,
            'fixture_id': fix['id'],
            'kickoff_time': fix.get('kickoff_time', ''),
            'team_h_id': team_h_id,
            'team_h_name': team_h_name,
            'team_a_id': team_a_id,
            'team_a_name': team_a_name,
            'B365H': b365h,
            'B365D': b365d,
            'B365A': b365a,
            'implied_prob_h': implied_h,
            'implied_prob_d': implied_d,
            'implied_prob_a': implied_a,
            **signals,
            'snapshot_time': datetime.now(timezone.utc).isoformat(),
        })

    matches = pd.DataFrame(match_rows)
    print(f"\nComputed signals for {len(matches)} matches")

    # 8. Generate predictions
    models = load_models()
    if models:
        predictions = generate_predictions(matches, models)
        # Save snapshot
        snap_file = SNAPSHOTS_DIR / f'gw{current_gw}_predictions.csv'
        predictions.to_csv(snap_file, index=False)
        print(f"Saved predictions to {snap_file}")

        # Append to master predictions file
        if PREDICTIONS_FILE.exists():
            existing = pd.read_csv(PREDICTIONS_FILE)
            # Remove any existing predictions for this GW (update mode)
            existing = existing[existing['gw'] != current_gw]
            predictions = pd.concat([existing, predictions], ignore_index=True)
        predictions.to_csv(PREDICTIONS_FILE, index=False)

        # Print predictions
        print(f"\n{'Match':<30} {'Crowd Fav':>10} {'Odds Fav':>10} {'Agree?':>8}")
        print("-" * 62)
        for _, row in predictions.iterrows():
            name = f"{row['team_h_name']} vs {row['team_a_name']}"
            crowd = row.get('crowd_favorite', '?')
            odds = row.get('odds_favorite', '?')
            agree = 'YES' if crowd == odds else 'NO'
            print(f"  {name:<28} {crowd:>10} {odds:>10} {agree:>8}")
    else:
        print("WARNING: No models found. Saving raw signals only.")
        snap_file = SNAPSHOTS_DIR / f'gw{current_gw}_signals.csv'
        matches.to_csv(snap_file, index=False)

    # Save raw bootstrap + fixtures for audit trail
    with open(SNAPSHOTS_DIR / f'gw{current_gw}_bootstrap.json', 'w') as f:
        # Save only essential fields to keep file size reasonable
        slim = {
            'events': bootstrap['events'],
            'teams': bootstrap['teams'],
            'elements': [{
                'id': e['id'], 'team': e['team'], 'element_type': e['element_type'],
                'web_name': e['web_name'], 'now_cost': e['now_cost'],
                'selected_by_percent': e['selected_by_percent'],
                'transfers_in_event': e['transfers_in_event'],
                'transfers_out_event': e['transfers_out_event'],
            } for e in elements],
        }
        json.dump(slim, f)

    print(f"\nSnapshot complete for GW{current_gw}")


# =========================================================================
# Results: Post-match result recording
# =========================================================================

def record_results(args=None):
    """Check completed fixtures and record actual results."""
    print("=" * 60)
    print("RECORDING RESULTS")
    print("=" * 60)

    if not PREDICTIONS_FILE.exists():
        print("No predictions file found. Run 'snapshot' first.")
        return

    predictions = pd.read_csv(PREDICTIONS_FILE)
    fixtures = fetch_fixtures()
    if fixtures is None:
        print("ERROR: Could not fetch fixtures")
        return

    # Build fixture result map
    fixture_results = {}
    for fix in fixtures:
        if fix.get('finished') and fix.get('team_h_score') is not None:
            h_score = fix['team_h_score']
            a_score = fix['team_a_score']
            if h_score > a_score:
                result = 'H'
            elif a_score > h_score:
                result = 'A'
            else:
                result = 'D'
            fixture_results[fix['id']] = {
                'result': result,
                'team_h_score': h_score,
                'team_a_score': a_score,
            }

    # Update predictions with results
    new_results = 0
    for idx, row in predictions.iterrows():
        fix_id = row['fixture_id']
        if fix_id in fixture_results and pd.isna(predictions.loc[idx, 'result'] if 'result' in predictions.columns else float('nan')):
            for key, val in fixture_results[fix_id].items():
                predictions.loc[idx, key] = val
            new_results += 1

    predictions.to_csv(PREDICTIONS_FILE, index=False)
    print(f"Updated {new_results} match results")

    # Refresh scorecard
    update_scorecard(predictions)


# =========================================================================
# Scorecard: Cumulative season tracking
# =========================================================================

def multiclass_brier(y_true, y_proba, n_classes=3):
    """Mean Brier score across classes."""
    from sklearn.metrics import brier_score_loss
    return np.mean([
        brier_score_loss((y_true == c).astype(int), y_proba[:, c])
        for c in range(n_classes)
    ])


def update_scorecard(predictions: pd.DataFrame):
    """Compute cumulative scorecard from predictions with results."""
    completed = predictions.dropna(subset=['result']).copy()
    if len(completed) == 0:
        print("No completed matches yet.")
        return

    y_true = completed['result'].map(RESULT_MAP).values

    scorecard = {
        'season': SEASON,
        'last_updated': datetime.now(timezone.utc).isoformat(),
        'matches_predicted': int(len(completed)),
        'gameweeks_covered': sorted(completed['gw'].unique().tolist()),
    }

    # Score each model
    for model_name in ['fpl_only', 'odds_only', 'fpl_odds']:
        cols = [f'p_{model_name}_A', f'p_{model_name}_D', f'p_{model_name}_H']
        if not all(c in completed.columns for c in cols):
            continue

        proba = completed[cols].values
        pred_class = proba.argmax(axis=1)
        pred_labels = [RESULT_LABELS[c] for c in pred_class]

        brier = multiclass_brier(y_true, proba)
        accuracy = (pred_class == y_true).mean()

        scorecard[model_name] = {
            'brier': round(float(brier), 4),
            'accuracy': round(float(accuracy), 4),
            'correct': int((pred_class == y_true).sum()),
            'total': int(len(completed)),
        }

    # Crowd vs Odds agreement
    if 'crowd_favorite' in completed.columns and 'odds_favorite' in completed.columns:
        agree = (completed['crowd_favorite'] == completed['odds_favorite']).sum()
        scorecard['agreement_rate'] = round(float(agree / len(completed) * 100), 1)

        # Accuracy on disagreement
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
    if 'fpl_only' in scorecard and 'odds_only' in scorecard:
        fpl_brier = scorecard['fpl_only']['brier']
        odds_brier = scorecard['odds_only']['brier']
        naive_brier = 0.2154  # from historical training data
        if naive_brier > 0:
            fpl_skill = 1 - fpl_brier / naive_brier
            odds_skill = 1 - odds_brier / naive_brier
            crowd_pct = (fpl_skill / odds_skill * 100) if odds_skill > 0 else 0
            scorecard['crowd_pct_of_odds_skill'] = round(float(crowd_pct), 1)
            scorecard['fpl_bss_vs_naive'] = round(float(fpl_skill * 100), 1)
            scorecard['odds_bss_vs_naive'] = round(float(odds_skill * 100), 1)

    # Per-gameweek breakdown
    gw_scores = []
    for gw in sorted(completed['gw'].unique()):
        gw_data = completed[completed['gw'] == gw]
        gw_y = gw_data['result'].map(RESULT_MAP).values
        gw_entry = {'gw': int(gw), 'n_matches': int(len(gw_data))}

        for model_name in ['fpl_only', 'odds_only']:
            cols = [f'p_{model_name}_A', f'p_{model_name}_D', f'p_{model_name}_H']
            if all(c in gw_data.columns for c in cols):
                proba = gw_data[cols].values
                gw_entry[f'{model_name}_brier'] = round(
                    float(multiclass_brier(gw_y, proba)), 4)
                gw_entry[f'{model_name}_accuracy'] = round(
                    float((proba.argmax(axis=1) == gw_y).mean()), 4)
        gw_scores.append(gw_entry)
    scorecard['per_gameweek'] = gw_scores

    # Save
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    with open(SCORECARD_FILE, 'w') as f:
        json.dump(scorecard, f, indent=2)
    print(f"\nScorecard saved to {SCORECARD_FILE}")

    return scorecard


def show_scorecard(args=None):
    """Display the current season scorecard."""
    print("=" * 60)
    print(f"LIVE SCORECARD — {SEASON}")
    print("=" * 60)

    if not SCORECARD_FILE.exists():
        print("No scorecard yet. Run 'snapshot' then 'results' first.")
        return

    with open(SCORECARD_FILE) as f:
        sc = json.load(f)

    print(f"Last updated: {sc['last_updated']}")
    print(f"Matches scored: {sc['matches_predicted']}")
    print(f"Gameweeks: {sc['gameweeks_covered']}")

    print(f"\n{'Model':<20} {'Brier':>8} {'Accuracy':>10} {'Correct':>10}")
    print("-" * 52)
    for model in ['fpl_only', 'odds_only', 'fpl_odds']:
        if model in sc:
            m = sc[model]
            print(f"  {model:<18} {m['brier']:>8.4f} {m['accuracy']:>9.1%} "
                  f"{m['correct']}/{m['total']}")

    if 'crowd_pct_of_odds_skill' in sc:
        print(f"\nCrowd captures {sc['crowd_pct_of_odds_skill']}% of bookmaker skill")
        print(f"FPL BSS vs naive: {sc['fpl_bss_vs_naive']}%")
        print(f"Odds BSS vs naive: {sc['odds_bss_vs_naive']}%")

    if 'agreement_rate' in sc:
        print(f"\nCrowd-Bookmaker agreement: {sc['agreement_rate']}%")
        if 'disagreement' in sc:
            d = sc['disagreement']
            print(f"On {d['n_matches']} disagreements: "
                  f"Crowd right {d['crowd_correct']}, Odds right {d['odds_correct']}")

    # Per-GW breakdown
    if 'per_gameweek' in sc:
        print(f"\n{'GW':>4} {'Matches':>8} {'FPL Brier':>10} {'Odds Brier':>11} {'FPL Acc':>8} {'Odds Acc':>9}")
        print("-" * 55)
        for gw in sc['per_gameweek']:
            fpl_b = gw.get('fpl_only_brier', 'N/A')
            odds_b = gw.get('odds_only_brier', 'N/A')
            fpl_a = gw.get('fpl_only_accuracy', 'N/A')
            odds_a = gw.get('odds_only_accuracy', 'N/A')
            fpl_b_str = f"{fpl_b:.4f}" if isinstance(fpl_b, float) else fpl_b
            odds_b_str = f"{odds_b:.4f}" if isinstance(odds_b, float) else odds_b
            fpl_a_str = f"{fpl_a:.1%}" if isinstance(fpl_a, float) else fpl_a
            odds_a_str = f"{odds_a:.1%}" if isinstance(odds_a, float) else odds_a
            print(f"  {gw['gw']:>3} {gw['n_matches']:>8} {fpl_b_str:>10} {odds_b_str:>11} "
                  f"{fpl_a_str:>8} {odds_a_str:>9}")


# =========================================================================
# Update: Combined snapshot + results
# =========================================================================

def update(args=None):
    """Run snapshot for current GW and record any new results."""
    snapshot(args)
    record_results(args)
    show_scorecard(args)


# =========================================================================
# Visualization: Live tracking chart
# =========================================================================

def chart_live_tracker():
    """Generate a live tracking chart showing cumulative Brier scores."""
    if not SCORECARD_FILE.exists():
        print("No scorecard data for charting.")
        return

    with open(SCORECARD_FILE) as f:
        sc = json.load(f)

    if 'per_gameweek' not in sc or len(sc['per_gameweek']) < 2:
        print("Need at least 2 gameweeks for chart.")
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    setup_chart_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    gws = [g['gw'] for g in sc['per_gameweek']]
    fpl_briers = [g.get('fpl_only_brier', None) for g in sc['per_gameweek']]
    odds_briers = [g.get('odds_only_brier', None) for g in sc['per_gameweek']]

    # Cumulative average
    fpl_cum = []
    odds_cum = []
    running_fpl = []
    running_odds = []
    for fb, ob in zip(fpl_briers, odds_briers):
        if fb is not None:
            running_fpl.append(fb)
        if ob is not None:
            running_odds.append(ob)
        fpl_cum.append(np.mean(running_fpl) if running_fpl else None)
        odds_cum.append(np.mean(running_odds) if running_odds else None)

    valid_gw = [gws[i] for i in range(len(gws)) if fpl_cum[i] is not None]
    valid_fpl = [fpl_cum[i] for i in range(len(gws)) if fpl_cum[i] is not None]
    valid_odds = [odds_cum[i] for i in range(len(gws)) if odds_cum[i] is not None]

    ax1.plot(valid_gw, valid_fpl, 'o-', color=CHART_COLORS['teal'], label='FPL Crowd', markersize=6)
    ax1.plot(valid_gw[:len(valid_odds)], valid_odds, 's-', color=CHART_COLORS['coral'],
             label='Bookmaker', markersize=6)
    ax1.set_xlabel('Gameweek')
    ax1.set_ylabel('Cumulative Avg Brier Score')
    ax1.set_title(f'Live Tracking: {SEASON}\nCrowd vs Bookmaker (lower = better)',
                  fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.2)

    # Per-GW Brier
    valid_gw_raw = [gws[i] for i in range(len(gws)) if fpl_briers[i] is not None]
    valid_fpl_raw = [fpl_briers[i] for i in range(len(gws)) if fpl_briers[i] is not None]
    valid_odds_raw = [odds_briers[i] for i in range(len(gws)) if odds_briers[i] is not None]

    width = 0.35
    x = np.arange(len(valid_gw_raw))
    ax2.bar(x - width/2, valid_fpl_raw, width, color=CHART_COLORS['teal'],
            label='FPL Crowd', edgecolor='#444444')
    ax2.bar(x + width/2, valid_odds_raw[:len(valid_gw_raw)], width,
            color=CHART_COLORS['coral'], label='Bookmaker', edgecolor='#444444')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'GW{g}' for g in valid_gw_raw], fontsize=9, rotation=45)
    ax2.set_ylabel('Brier Score')
    ax2.set_title('Per-Gameweek Brier Score', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.2)

    plt.tight_layout()
    chart_path = CHARTS_DIR / 'live_tracker.png'
    fig.savefig(chart_path)
    plt.close()
    print(f"Saved live tracker chart to {chart_path}")


# =========================================================================
# Main CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Live FPL Crowd Wisdom Tracker',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  snapshot    Capture FPL ownership + odds before GW deadline
  results     Record actual match results after completion
  scorecard   Display cumulative season scorecard
  update      Run snapshot + results + scorecard
  chart       Generate live tracking visualization
        """)
    parser.add_argument('command', choices=['snapshot', 'results', 'scorecard',
                                            'update', 'chart'],
                        help='Action to perform')

    args = parser.parse_args()

    commands = {
        'snapshot': snapshot,
        'results': record_results,
        'scorecard': show_scorecard,
        'update': update,
        'chart': chart_live_tracker,
    }

    commands[args.command](args)


if __name__ == '__main__':
    main()
