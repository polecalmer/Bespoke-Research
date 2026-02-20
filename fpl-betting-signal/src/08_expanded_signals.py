"""Expanded signal extraction: mine every dimension of FPL manager intuition.

Goes back to raw player-GW data, extracts ALL available columns, and builds
match-level signals that capture different facets of how managers see the game.

Signal categories:
1. STOCK signals    — accumulated state (selected, value)
2. FLOW signals     — active decisions this week (transfers, price changes)
3. QUALITY signals  — underlying performance metrics (ICT, BPS, threat)
4. STRUCTURE signals — tactical choices (position splits, concentration, premium/budget)
5. MOMENTUM signals — trailing performance (prev GW points, form)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEASONS, TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS,
    DATA_RAW_FPL, DATA_PROCESSED, MODEL_DIR, CHARTS_DIR,
    load_parquet, save_parquet, setup_chart_style, CHART_COLORS,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ================================================================
# Data Loading
# ================================================================

def load_player_gw(season: str) -> pd.DataFrame:
    """Load merged_gw with all columns, properly typed."""
    path = DATA_RAW_FPL / f'{season}_merged_gw.csv'
    df = pd.read_csv(path, encoding='latin-1', low_memory=False)
    df.columns = [c.strip().strip('"') for c in df.columns]

    # Core columns — coerce to numeric
    numeric_cols = [
        'element', 'GW', 'fixture', 'opponent_team', 'value', 'selected',
        'transfers_in', 'transfers_out', 'transfers_balance',
        'total_points', 'minutes', 'bonus', 'bps',
        'ict_index', 'influence', 'creativity', 'threat',
        'goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
        'saves', 'yellow_cards', 'red_cards', 'own_goals',
        'penalties_missed', 'penalties_saved',
        'team_h_score', 'team_a_score',
        # Later seasons
        'xP', 'expected_goals', 'expected_assists',
        'expected_goal_involvements', 'expected_goals_conceded', 'starts',
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'transfers_balance' not in df.columns:
        df['transfers_balance'] = df['transfers_in'] - df['transfers_out']

    if df['was_home'].dtype == object:
        df['was_home'] = df['was_home'].astype(str).str.strip().str.lower() == 'true'

    df['gw'] = df['GW'].astype('Int64')
    df['season'] = season

    # Assign team_id
    df = _assign_team_id(df, season)

    # Assign element_type (position)
    df = _assign_element_type(df, season)

    return df


def _assign_team_id(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Assign team_id from team column or players_raw."""
    matches = load_parquet('matches.parquet')
    season_matches = matches[matches['season'] == season]

    if 'team' in df.columns and df['team'].notna().any():
        name_to_id = {}
        for _, row in season_matches.iterrows():
            name_to_id[row['team_h_name']] = int(row['team_h_id'])
            name_to_id[row['team_a_name']] = int(row['team_a_id'])
        df['team_id'] = df['team'].map(name_to_id)
    else:
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


def _assign_element_type(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Assign position (element_type) from position column or players_raw."""
    if 'position' in df.columns and df['position'].notna().any():
        pos_map = {'GK': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}
        df['element_type'] = df['position'].map(pos_map)
    else:
        pr_path = DATA_RAW_FPL / f'{season}_players_raw.csv'
        if pr_path.exists():
            pr = pd.read_csv(pr_path, encoding='latin-1', low_memory=False)
            pr.columns = [c.strip().strip('"') for c in pr.columns]
            if 'element_type' in pr.columns:
                et_map = dict(zip(pr['id'].astype(int), pr['element_type'].astype(int)))
                df['element_type'] = df['element'].map(et_map)

    if 'element_type' not in df.columns:
        df['element_type'] = 0
    df['element_type'] = df['element_type'].fillna(0).astype(int)
    return df


# ================================================================
# Signal Computation — one match at a time
# ================================================================

def compute_match_signals(hp: pd.DataFrame, ap: pd.DataFrame,
                          hp_prev: pd.DataFrame, ap_prev: pd.DataFrame) -> dict:
    """Compute all signals for one match from home/away player data.

    hp/ap: home/away players for THIS gameweek (pre-match: selected, transfers, value)
    hp_prev/ap_prev: home/away players from PREVIOUS gameweek (outcomes: points, ICT, etc.)
    """
    signals = {}

    # ========== 1. STOCK SIGNALS (accumulated state) ==========

    # 1a. Ownership ratio (price-weighted) — EXISTING
    pw_h = (hp['selected'] * hp['value']).sum()
    pw_a = (ap['selected'] * ap['value']).sum()
    signals['ownership_ratio'] = pw_h / (pw_h + pw_a + 1)

    # 1b. Raw ownership ratio (unweighted)
    sel_h = hp['selected'].sum()
    sel_a = ap['selected'].sum()
    signals['raw_ownership_ratio'] = sel_h / (sel_h + sel_a + 1)

    # 1c. Total squad value ratio
    val_h = hp['value'].sum()
    val_a = ap['value'].sum()
    signals['squad_value_ratio'] = val_h / (val_h + val_a + 1)

    # ========== 2. FLOW SIGNALS (active decisions) ==========

    # 2a. Transfer ratio — EXISTING
    tms_h = hp['transfers_balance'].sum()
    tms_a = ap['transfers_balance'].sum()
    signals['transfer_ratio'] = tms_h / (abs(tms_h) + abs(tms_a) + 1)

    # 2b. Transfer-in conviction (transfers_in weighted by value — expensive buys)
    conv_h = (hp['transfers_in'] * hp['value']).sum()
    conv_a = (ap['transfers_in'] * ap['value']).sum()
    signals['transfer_conviction_ratio'] = conv_h / (conv_h + conv_a + 1)

    # 2c. Transfer intensity (transfers_in / selected — how actively being bought)
    intensity_h = (hp['transfers_in'] / (hp['selected'] + 1)).mean()
    intensity_a = (ap['transfers_in'] / (ap['selected'] + 1)).mean()
    signals['transfer_intensity_delta'] = intensity_h - intensity_a

    # 2d. Sell pressure (transfers_out ratio)
    tout_h = hp['transfers_out'].sum()
    tout_a = ap['transfers_out'].sum()
    signals['sell_pressure_ratio'] = tout_h / (tout_h + tout_a + 1)

    # 2e. Captain proxy (transfer velocity) — EXISTING
    tv_h = tms_h / (sel_h + 1)
    tv_a = tms_a / (sel_a + 1)
    signals['captain_proxy'] = tv_h - tv_a

    # 2f. Ownership velocity (would need prev GW — computed at match level later)

    # ========== 3. QUALITY SIGNALS (underlying performance from prev GW) ==========

    if len(hp_prev) > 0 and len(ap_prev) > 0:
        # 3a. ICT Threat — attacking danger
        threat_h = (hp_prev['threat'] * hp_prev['selected']).sum() if 'threat' in hp_prev.columns else 0
        threat_a = (ap_prev['threat'] * ap_prev['selected']).sum() if 'threat' in ap_prev.columns else 0
        signals['threat_ratio'] = threat_h / (threat_h + threat_a + 1)

        # 3b. ICT Creativity — chance creation
        creat_h = (hp_prev['creativity'] * hp_prev['selected']).sum() if 'creativity' in hp_prev.columns else 0
        creat_a = (ap_prev['creativity'] * ap_prev['selected']).sum() if 'creativity' in ap_prev.columns else 0
        signals['creativity_ratio'] = creat_h / (creat_h + creat_a + 1)

        # 3c. ICT Influence — overall match impact
        infl_h = (hp_prev['influence'] * hp_prev['selected']).sum() if 'influence' in hp_prev.columns else 0
        infl_a = (ap_prev['influence'] * ap_prev['selected']).sum() if 'influence' in ap_prev.columns else 0
        signals['influence_ratio'] = infl_h / (infl_h + infl_a + 1)

        # 3d. BPS quality — detailed performance metric
        bps_h = (hp_prev['bps'] * hp_prev['selected']).sum() if 'bps' in hp_prev.columns else 0
        bps_a = (ap_prev['bps'] * ap_prev['selected']).sum() if 'bps' in ap_prev.columns else 0
        signals['bps_quality_ratio'] = bps_h / (bps_h + bps_a + 1)

        # 3e. Points momentum — recent performance weighted by current ownership
        pts_h = (hp_prev['total_points'] * hp['selected']).sum() if 'total_points' in hp_prev.columns else 0
        pts_a = (ap_prev['total_points'] * ap['selected']).sum() if 'total_points' in ap_prev.columns else 0
        signals['points_momentum_ratio'] = pts_h / (pts_h + pts_a + 1)

        # 3f. Minutes ratio — who played more (fitness/rotation)
        min_h = hp_prev['minutes'].sum() if 'minutes' in hp_prev.columns else 0
        min_a = ap_prev['minutes'].sum() if 'minutes' in ap_prev.columns else 0
        signals['minutes_ratio'] = min_h / (min_h + min_a + 1)

        # 3g. Clean sheet signal from prev GW
        cs_h = hp_prev['clean_sheets'].sum() if 'clean_sheets' in hp_prev.columns else 0
        cs_a = ap_prev['clean_sheets'].sum() if 'clean_sheets' in ap_prev.columns else 0
        signals['prev_clean_sheet_ratio'] = cs_h / (cs_h + cs_a + 1)

        # 3h. Goals conceded ratio (inverted — lower is better defensively)
        gc_h = hp_prev['goals_conceded'].sum() if 'goals_conceded' in hp_prev.columns else 0
        gc_a = ap_prev['goals_conceded'].sum() if 'goals_conceded' in ap_prev.columns else 0
        # Invert: fewer goals conceded = stronger
        signals['defensive_strength_ratio'] = gc_a / (gc_h + gc_a + 1)
    else:
        # GW1 or missing previous data — set quality signals to 0.5 (neutral)
        for key in ['threat_ratio', 'creativity_ratio', 'influence_ratio',
                     'bps_quality_ratio', 'points_momentum_ratio', 'minutes_ratio',
                     'prev_clean_sheet_ratio', 'defensive_strength_ratio']:
            signals[key] = 0.5

    # ========== 4. STRUCTURE SIGNALS (tactical composition) ==========

    # 4a. DCS ratio — EXISTING
    def_gk_h = hp[hp['element_type'].isin([1, 2])]['selected'].sum()
    def_gk_a = ap[ap['element_type'].isin([1, 2])]['selected'].sum()
    signals['dcs_ratio'] = def_gk_h / (def_gk_h + def_gk_a + 1)

    # 4b. Forward ownership ratio (attacking intent)
    fwd_h = hp[hp['element_type'] == 4]['selected'].sum()
    fwd_a = ap[ap['element_type'] == 4]['selected'].sum()
    signals['fwd_ownership_ratio'] = fwd_h / (fwd_h + fwd_a + 1)

    # 4c. Midfielder ownership ratio
    mid_h = hp[hp['element_type'] == 3]['selected'].sum()
    mid_a = ap[ap['element_type'] == 3]['selected'].sum()
    signals['mid_ownership_ratio'] = mid_h / (mid_h + mid_a + 1)

    # 4d. Premium player ratio (value >= 80, i.e. 8.0M+)
    prem_h = hp[hp['value'] >= 80]['selected'].sum()
    prem_a = ap[ap['value'] >= 80]['selected'].sum()
    signals['premium_ratio'] = prem_h / (prem_h + prem_a + 1)

    # 4e. Budget player ratio (value < 50, i.e. <5.0M)
    budget_h = hp[hp['value'] < 50]['selected'].sum()
    budget_a = ap[ap['value'] < 50]['selected'].sum()
    signals['budget_ratio'] = budget_h / (budget_h + budget_a + 1)

    # 4f. Selection concentration (HHI) — how spread vs concentrated ownership is
    def hhi(selected_series):
        total = selected_series.sum()
        if total == 0:
            return 0
        shares = selected_series / total
        return (shares ** 2).sum()

    hhi_h = hhi(hp['selected'])
    hhi_a = hhi(ap['selected'])
    signals['concentration_delta'] = hhi_h - hhi_a

    # 4g. Star player dependence — top 3 players' share of total selected
    def top_n_share(selected_series, n=3):
        total = selected_series.sum()
        if total == 0:
            return 0
        return selected_series.nlargest(n).sum() / total

    star_h = top_n_share(hp['selected'])
    star_a = top_n_share(ap['selected'])
    signals['star_dependence_delta'] = star_h - star_a

    # ========== 5. FORWARD-LOOKING SIGNALS (where available) ==========

    # 5a. Expected points ratio (xP) — 2020-21+
    if 'xP' in hp.columns and hp['xP'].notna().any():
        xp_h = hp['xP'].sum()
        xp_a = ap['xP'].sum()
        signals['xp_ratio'] = xp_h / (xp_h + xp_a + 1)
    else:
        signals['xp_ratio'] = np.nan

    # 5b. Expected goals (xG) — 2023-24+
    if 'expected_goals' in hp.columns and hp['expected_goals'].notna().any():
        xg_h = hp['expected_goals'].sum()
        xg_a = ap['expected_goals'].sum()
        signals['xg_ratio'] = xg_h / (xg_h + xg_a + 1)
    else:
        signals['xg_ratio'] = np.nan

    # 5c. Expected goals conceded (xGC) — 2023-24+
    if 'expected_goals_conceded' in hp.columns and hp['expected_goals_conceded'].notna().any():
        xgc_h = hp['expected_goals_conceded'].sum()
        xgc_a = ap['expected_goals_conceded'].sum()
        # Invert: fewer expected goals conceded = stronger
        signals['xgc_strength_ratio'] = xgc_a / (xgc_h + xgc_a + 1)
    else:
        signals['xgc_strength_ratio'] = np.nan

    return signals


# ================================================================
# Main Pipeline
# ================================================================

def build_expanded_signals() -> pd.DataFrame:
    """Build expanded signal set for all matches across all seasons."""
    matches = load_parquet('matches.parquet')
    all_signals = []

    for season in SEASONS:
        print(f"  Processing {season}...")
        df = load_player_gw(season)
        df = df[df['team_id'] > 0]

        season_matches = matches[matches['season'] == season]
        count = 0

        for _, match in season_matches.iterrows():
            gw = match['gw']
            th_id = match['team_h_id']
            ta_id = match['team_a_id']

            # Current GW players
            hp = df[(df['gw'] == gw) & (df['team_id'] == th_id)]
            ap = df[(df['gw'] == gw) & (df['team_id'] == ta_id)]

            if len(hp) == 0 or len(ap) == 0:
                continue

            # Previous GW players (for quality/momentum signals)
            prev_gw = gw - 1
            hp_prev = df[(df['gw'] == prev_gw) & (df['team_id'] == th_id)]
            ap_prev = df[(df['gw'] == prev_gw) & (df['team_id'] == ta_id)]

            # For prev GW quality signals, we need to align player IDs
            # so we weight prev performance by CURRENT ownership
            if len(hp_prev) > 0:
                hp_prev = hp_prev.copy()
                current_sel = dict(zip(hp['element'], hp['selected']))
                hp_prev['selected'] = hp_prev['element'].map(current_sel).fillna(0)

            if len(ap_prev) > 0:
                ap_prev = ap_prev.copy()
                current_sel = dict(zip(ap['element'], ap['selected']))
                ap_prev['selected'] = ap_prev['element'].map(current_sel).fillna(0)

            sigs = compute_match_signals(hp, ap, hp_prev, ap_prev)
            sigs['season'] = season
            sigs['gw'] = gw
            sigs['fixture_id'] = match['fixture_id']
            all_signals.append(sigs)
            count += 1

        print(f"    → {count} matches")

    sig_df = pd.DataFrame(all_signals)
    return sig_df


def compute_velocity_signals(sig_df: pd.DataFrame) -> pd.DataFrame:
    """Add ownership velocity (needs team-GW panel, computed post-hoc)."""
    matches = load_parquet('matches.parquet')
    merged = sig_df.merge(
        matches[['season', 'gw', 'fixture_id', 'total_selected_h', 'total_selected_a',
                 'team_h_name', 'team_a_name']],
        on=['season', 'gw', 'fixture_id'], how='left'
    )

    # Build team-GW panel
    home = merged[['season', 'gw', 'team_h_name', 'total_selected_h']].rename(
        columns={'team_h_name': 'team', 'total_selected_h': 'total_selected'})
    away = merged[['season', 'gw', 'team_a_name', 'total_selected_a']].rename(
        columns={'team_a_name': 'team', 'total_selected_a': 'total_selected'})

    team_gw = pd.concat([home, away]).groupby(['season', 'team', 'gw'])['total_selected'].first().reset_index()
    team_gw = team_gw.sort_values(['season', 'team', 'gw'])
    team_gw['prev_sel'] = team_gw.groupby(['season', 'team'])['total_selected'].shift(1)
    team_gw['velocity'] = (team_gw['total_selected'] - team_gw['prev_sel']).fillna(0)

    # Join back
    ov_h = team_gw.rename(columns={'team': 'team_h_name', 'velocity': 'ov_h'})[['season', 'team_h_name', 'gw', 'ov_h']]
    ov_a = team_gw.rename(columns={'team': 'team_a_name', 'velocity': 'ov_a'})[['season', 'team_a_name', 'gw', 'ov_a']]

    merged = merged.merge(ov_h, on=['season', 'team_h_name', 'gw'], how='left')
    merged = merged.merge(ov_a, on=['season', 'team_a_name', 'gw'], how='left')
    merged['velocity_delta'] = merged['ov_h'].fillna(0) - merged['ov_a'].fillna(0)

    # Drop temp columns
    merged = merged.drop(columns=['total_selected_h', 'total_selected_a',
                                   'team_h_name', 'team_a_name', 'ov_h', 'ov_a'], errors='ignore')

    return merged


# ================================================================
# Model Training
# ================================================================

def multiclass_brier(y_true, y_proba, n_classes=3):
    return np.mean([
        brier_score_loss((y_true == c).astype(int), y_proba[:, c])
        for c in range(n_classes)
    ])


def train_expanded_models(sig_df: pd.DataFrame):
    """Train models with various feature subsets and report results."""
    matches = load_parquet('matches.parquet')
    merged = sig_df.merge(
        matches[['season', 'gw', 'fixture_id', 'result',
                 'implied_prob_h', 'implied_prob_d', 'implied_prob_a']],
        on=['season', 'gw', 'fixture_id'], how='left'
    )

    ODDS = ['implied_prob_h', 'implied_prob_d', 'implied_prob_a']

    # Original 5 signals
    ORIGINAL_5 = ['transfer_ratio', 'ownership_ratio', 'captain_proxy', 'dcs_ratio', 'velocity_delta']

    # All signals available across ALL 9 seasons
    UNIVERSAL = [
        'ownership_ratio', 'raw_ownership_ratio', 'squad_value_ratio',
        'transfer_ratio', 'transfer_conviction_ratio', 'transfer_intensity_delta',
        'sell_pressure_ratio', 'captain_proxy',
        'threat_ratio', 'creativity_ratio', 'influence_ratio',
        'bps_quality_ratio', 'points_momentum_ratio', 'minutes_ratio',
        'prev_clean_sheet_ratio', 'defensive_strength_ratio',
        'dcs_ratio', 'fwd_ownership_ratio', 'mid_ownership_ratio',
        'premium_ratio', 'budget_ratio',
        'concentration_delta', 'star_dependence_delta',
        'velocity_delta',
    ]

    # Group by category
    STOCK = ['ownership_ratio', 'raw_ownership_ratio', 'squad_value_ratio']
    FLOW = ['transfer_ratio', 'transfer_conviction_ratio', 'transfer_intensity_delta',
            'sell_pressure_ratio', 'captain_proxy', 'velocity_delta']
    QUALITY = ['threat_ratio', 'creativity_ratio', 'influence_ratio',
               'bps_quality_ratio', 'points_momentum_ratio', 'minutes_ratio',
               'prev_clean_sheet_ratio', 'defensive_strength_ratio']
    STRUCTURE = ['dcs_ratio', 'fwd_ownership_ratio', 'mid_ownership_ratio',
                 'premium_ratio', 'budget_ratio',
                 'concentration_delta', 'star_dependence_delta']

    feature_sets = {
        'odds_only':           ODDS,
        'original_5':          ORIGINAL_5,
        'original_5+odds':     ODDS + ORIGINAL_5,
        'stock':               STOCK,
        'flow':                FLOW,
        'quality':             QUALITY,
        'structure':           STRUCTURE,
        'all_fpl':             UNIVERSAL,
        'stock+flow':          STOCK + FLOW,
        'quality+structure':   QUALITY + STRUCTURE,
        'all_fpl+odds':        ODDS + UNIVERSAL,
        'quality+odds':        ODDS + QUALITY,
        'structure+odds':      ODDS + STRUCTURE,
        'flow+quality+odds':   ODDS + FLOW + QUALITY,
    }

    # Prepare data — use UNIVERSAL signals (available across all seasons)
    all_features = list(set(ODDS + UNIVERSAL))
    clean = merged.dropna(subset=all_features + ['result'])

    train = clean[clean['season'].isin(TRAIN_SEASONS)]
    val = clean[clean['season'].isin(VAL_SEASONS)]
    test = clean[clean['season'].isin(TEST_SEASONS)]

    print(f"\n{'='*80}")
    print("EXPANDED MODEL COMPARISON")
    print(f"{'='*80}")
    print(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")
    print(f"Total features available: {len(UNIVERSAL)} FPL + 3 odds = {len(UNIVERSAL) + 3}")

    RESULT_MAP = {'A': 0, 'D': 1, 'H': 2}
    y_train = train['result'].map(RESULT_MAP).values
    y_val = val['result'].map(RESULT_MAP).values
    y_test = test['result'].map(RESULT_MAP).values

    scaler = StandardScaler()
    X_train_full = pd.DataFrame(
        scaler.fit_transform(train[all_features]),
        columns=all_features, index=train.index)
    X_val_full = pd.DataFrame(
        scaler.transform(val[all_features]),
        columns=all_features, index=val.index)
    X_test_full = pd.DataFrame(
        scaler.transform(test[all_features]),
        columns=all_features, index=test.index)

    print(f"\n{'Model':<25} {'#Feat':>5} {'Val Brier':>10} {'Test Brier':>11} "
          f"{'Val Acc':>8} {'Test Acc':>9} {'BSS vs Odds':>12}")
    print("-" * 85)

    results = {}
    for name, features in feature_sets.items():
        # Filter to features that exist in the scaler
        feats = [f for f in features if f in all_features]
        if len(feats) == 0:
            continue

        base = LogisticRegression(solver='lbfgs', max_iter=1000, C=0.01)
        model = CalibratedClassifierCV(base, cv=5, method='isotonic')
        model.fit(X_train_full[feats].values, y_train)

        val_proba = model.predict_proba(X_val_full[feats].values)
        test_proba = model.predict_proba(X_test_full[feats].values)

        val_brier = multiclass_brier(y_val, val_proba)
        test_brier = multiclass_brier(y_test, test_proba)
        val_acc = accuracy_score(y_val, model.predict(X_val_full[feats].values))
        test_acc = accuracy_score(y_test, model.predict(X_test_full[feats].values))

        results[name] = {
            'val_brier': val_brier, 'test_brier': test_brier,
            'val_acc': val_acc, 'test_acc': test_acc,
            'n_features': len(feats),
        }

    # Print with BSS
    odds_test = results['odds_only']['test_brier']
    for name, r in results.items():
        bss = 1 - (r['test_brier'] / odds_test) if name != 'odds_only' else 0
        print(f"{name:<25} {r['n_features']:>5} {r['val_brier']:>10.4f} {r['test_brier']:>11.4f} "
              f"{r['val_acc']:>8.3f} {r['test_acc']:>9.3f} {bss:>+12.4f}")

    # Naive baseline
    naive_brier = multiclass_brier(y_test, np.tile(
        [(y_train == 0).mean(), (y_train == 1).mean(), (y_train == 2).mean()],
        (len(y_test), 1)
    ))

    print(f"\nNaive baseline test Brier: {naive_brier:.4f}")
    print(f"\nSkill vs Naive (% of gap closed):")
    odds_skill = 1 - odds_test / naive_brier
    for name, r in results.items():
        skill = 1 - r['test_brier'] / naive_brier
        pct = (skill / odds_skill * 100) if odds_skill > 0 else 0
        marker = " <<<" if pct > 100 else ""
        print(f"  {name:<25} skill={skill*100:>5.1f}%  ({pct:>5.1f}% of odds skill){marker}")

    # Signal correlations with odds
    print(f"\n--- All Signal Correlations with Implied P(Home) ---")
    for col in sorted(UNIVERSAL):
        valid_data = merged[['implied_prob_h', col]].dropna()
        if len(valid_data) > 100:
            r = valid_data['implied_prob_h'].corr(valid_data[col])
            print(f"  {col:<35} r = {r:>7.4f}")

    # Feature importance from all_fpl+odds model
    print(f"\n--- Feature Importance (all_fpl+odds model coefficients) ---")
    feats = [f for f in (ODDS + UNIVERSAL) if f in all_features]
    base = LogisticRegression(solver='lbfgs', max_iter=1000, C=0.01)
    base.fit(X_train_full[feats].values, y_train)
    importances = pd.DataFrame({
        'feature': feats,
        'mean_abs_coef': np.abs(base.coef_).mean(axis=0),
    }).sort_values('mean_abs_coef', ascending=False)
    for _, row in importances.iterrows():
        print(f"  {row['feature']:<35} {row['mean_abs_coef']:>8.4f}")

    return results, merged


def plot_signal_overview(merged: pd.DataFrame, results: dict):
    """Plot expanded signal analysis."""
    setup_chart_style()

    # Bar chart: all models ranked by test Brier
    fig, ax = plt.subplots(figsize=(14, 8))

    sorted_results = sorted(results.items(), key=lambda x: x[1]['test_brier'])
    names = [r[0] for r in sorted_results]
    briers = [r[1]['test_brier'] for r in sorted_results]
    colors = [CHART_COLORS['coral'] if 'odds' in n else CHART_COLORS['teal'] for n in names]
    # Highlight odds_only
    for i, n in enumerate(names):
        if n == 'odds_only':
            colors[i] = CHART_COLORS['purple']

    bars = ax.barh(range(len(names)), briers, color=colors, alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('Test Brier Score (lower = better)')
    ax.set_title('Expanded Signal Analysis: All Model Variants')
    ax.axvline(results['odds_only']['test_brier'], color=CHART_COLORS['white'],
               linestyle='--', alpha=0.4, label='Odds-only baseline')
    ax.legend(loc='lower right')
    ax.invert_yaxis()

    plt.tight_layout()
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(CHARTS_DIR / 'expanded_model_comparison.png')
    plt.close()
    print(f"Saved: {CHARTS_DIR / 'expanded_model_comparison.png'}")


# ================================================================
# Main
# ================================================================

def main():
    print("=" * 80)
    print("EXPANDED SIGNAL EXTRACTION")
    print("Mining every dimension of FPL manager intuition")
    print("=" * 80)

    print("\n[1/4] Building expanded signals from raw player data...")
    sig_df = build_expanded_signals()

    print(f"\n[2/4] Adding velocity signals...")
    sig_df = compute_velocity_signals(sig_df)

    save_parquet(sig_df, 'signals_expanded.parquet')

    print(f"\n[3/4] Training models...")
    results, merged = train_expanded_models(sig_df)

    print(f"\n[4/4] Plotting...")
    plot_signal_overview(merged, results)

    # Save results
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    import json
    with open(MODEL_DIR / 'expanded_comparison.json', 'w') as f:
        json.dump({k: {kk: float(vv) for kk, vv in v.items()}
                   for k, v in results.items()}, f, indent=2)

    print(f"\nResults saved to {MODEL_DIR / 'expanded_comparison.json'}")
    print("Done!")


if __name__ == '__main__':
    main()
