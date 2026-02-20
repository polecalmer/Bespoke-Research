"""Elite Noise Coefficient: estimate top-manager ownership from crowd data.

Uses archived top-50K ownership data from fplAnalytics (2018-19 season, GW1-23)
to learn the systematic distortion between crowd ownership and elite ownership.
Applies that mapping retroactively to all 9 seasons and tests whether
elite-adjusted signals carry stronger predictive signal for match outcomes.

Data source: https://github.com/gpoudel/FPL-Analytics/tree/master/fplBoardLive/data
"""

import sys
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEASONS, TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS,
    DATA_RAW_FPL, DATA_PROCESSED, MODEL_DIR, CHARTS_DIR,
    load_parquet, save_parquet, setup_chart_style, CHART_COLORS,
)

# --- Config ---
ELITE_DATA_DIR = DATA_RAW_FPL.parent / 'fpl_elite'
ELITE_BASE_URL = 'https://raw.githubusercontent.com/gpoudel/FPL-Analytics/master/fplBoardLive/data'
CALIBRATION_GWS = list(range(1, 24))  # GW1-23 available


# ================================================================
# STEP 9a: Download & Analyze Calibration Data
# ================================================================

def download_elite_csvs():
    """Download top50K-GW{n}.csv files from fplAnalytics GitHub."""
    import requests

    ELITE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = 0

    for gw in CALIBRATION_GWS:
        fname = f'top50K-GW{gw}.csv'
        dest = ELITE_DATA_DIR / fname
        if dest.exists():
            continue

        url = f'{ELITE_BASE_URL}/{fname}'
        for attempt in range(4):
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                dest.write_text(resp.text)
                downloaded += 1
                break
            except Exception as e:
                if attempt < 3:
                    wait = 2 ** (attempt + 1)
                    print(f"  Retry {attempt+1} for GW{gw} (waiting {wait}s): {e}")
                    time.sleep(wait)
                else:
                    print(f"  FAILED to download GW{gw}: {e}")

    print(f"Downloaded {downloaded} new files ({len(CALIBRATION_GWS)} total GWs)")


def load_calibration_data() -> pd.DataFrame:
    """Load all top-50K CSVs and compute paired (crowd_pct, elite_pct)."""
    frames = []
    for gw in CALIBRATION_GWS:
        path = ELITE_DATA_DIR / f'top50K-GW{gw}.csv'
        if not path.exists():
            continue
        df = pd.read_csv(path, sep=';')
        df.columns = [c.strip().strip('"') for c in df.columns]
        # Strip quotes from string columns
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].astype(str).str.strip('"')
        df['gw'] = gw
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No calibration CSVs found. Run download first.")

    cal = pd.concat(frames, ignore_index=True)

    # Determine sample size per GW: sum of max(selected_by) players or from manager file
    # Heuristic: sample_size = max plausible count. For a proper estimate,
    # the top player's selected_by should be close to the total sample.
    # We'll use the manager count files if available, otherwise estimate.
    gw_sample_sizes = {}
    for gw in cal['gw'].unique():
        mgr_path = ELITE_DATA_DIR.parent / 'fpl_elite' / f'top50k_managers_df-GW{gw}.csv'
        if mgr_path.exists():
            mgr_df = pd.read_csv(mgr_path, sep=';', on_bad_lines='skip')
            gw_sample_sizes[gw] = len(mgr_df)
        else:
            # Estimate: each manager has 15 players, so total player-selections
            # divided by 15 gives approximate manager count
            gw_data = cal[cal['gw'] == gw]
            total_selections = gw_data['selected_by'].sum()
            # Each manager picks 15 players
            estimated_n = total_selections / 15
            gw_sample_sizes[gw] = estimated_n

    cal['sample_size'] = cal['gw'].map(gw_sample_sizes)

    # Ensure numeric types
    for col in ['selected_by_percent', 'selected_by', 'started_by', 'captained_by',
                'vice_captained_by', 'triple_captained_by', 'benched_by', 'now_cost']:
        if col in cal.columns:
            cal[col] = pd.to_numeric(cal[col], errors='coerce')

    # Compute percentages
    cal['crowd_pct'] = cal['selected_by_percent'] / 100.0
    cal['elite_pct'] = cal['selected_by'] / cal['sample_size']

    # Also compute elite captain and started fractions
    cal['elite_started_pct'] = cal['started_by'] / cal['sample_size']
    cal['elite_captained_pct'] = cal['captained_by'].fillna(0) / cal['sample_size']

    # Sanity checks
    print(f"Calibration data: {len(cal)} player-GW observations across {cal['gw'].nunique()} GWs")
    print(f"Sample sizes per GW: min={min(gw_sample_sizes.values()):.0f}, "
          f"max={max(gw_sample_sizes.values()):.0f}")
    print(f"Crowd pct range: [{cal['crowd_pct'].min():.3f}, {cal['crowd_pct'].max():.3f}]")
    print(f"Elite pct range: [{cal['elite_pct'].min():.4f}, {cal['elite_pct'].max():.4f}]")

    # Flag potential issues
    outliers = cal[cal['elite_pct'] > 1.0]
    if len(outliers) > 0:
        print(f"WARNING: {len(outliers)} rows have elite_pct > 1.0 (capping at 1.0)")
        cal['elite_pct'] = cal['elite_pct'].clip(0, 1)

    return cal


def analyze_distortion(cal: pd.DataFrame) -> dict:
    """Analyze the crowd-to-elite ownership distortion."""
    # Filter to meaningful ownership (exclude near-zero)
    valid = cal[(cal['crowd_pct'] > 0.001) & (cal['elite_pct'] > 0)].copy()
    valid['ratio'] = valid['elite_pct'] / valid['crowd_pct']

    print(f"\n{'='*60}")
    print("CROWD → ELITE DISTORTION ANALYSIS")
    print(f"{'='*60}")
    print(f"Valid observations: {len(valid)}")

    # Binned analysis
    bins = [0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0]
    labels = ['0-1%', '1-2%', '2-5%', '5-10%', '10-20%', '20-30%', '30-50%', '50%+']
    valid['crowd_bin'] = pd.cut(valid['crowd_pct'], bins=bins, labels=labels)

    bin_stats = valid.groupby('crowd_bin', observed=True).agg(
        count=('ratio', 'size'),
        mean_ratio=('ratio', 'mean'),
        median_ratio=('ratio', 'median'),
        mean_crowd=('crowd_pct', 'mean'),
        mean_elite=('elite_pct', 'mean'),
    ).round(3)

    print(f"\n{'Crowd Bin':<12} {'Count':>6} {'Mean Ratio':>11} {'Med Ratio':>10} {'Avg Crowd':>10} {'Avg Elite':>10}")
    print("-" * 65)
    for idx, row in bin_stats.iterrows():
        print(f"{str(idx):<12} {row['count']:>6.0f} {row['mean_ratio']:>11.3f} "
              f"{row['median_ratio']:>10.3f} {row['mean_crowd']:>10.3f} {row['mean_elite']:>10.3f}")

    # Overall correlation
    r = valid['crowd_pct'].corr(valid['elite_pct'])
    print(f"\nCorrelation (crowd_pct, elite_pct): r = {r:.4f}")

    return {
        'n_obs': len(valid),
        'correlation': r,
        'bin_stats': bin_stats.to_dict(),
    }


# ================================================================
# STEP 9b: Fit the Noise Coefficient Mapping
# ================================================================

def fit_elite_mapping(cal: pd.DataFrame) -> dict:
    """Fit crowd→elite mapping using isotonic regression and validate stability."""
    valid = cal[(cal['crowd_pct'] > 0.001)].copy()

    # Split: GW1-12 for fitting, GW13-23 for validation
    fit_data = valid[valid['gw'] <= 12]
    val_data = valid[valid['gw'] > 12]

    print(f"\n{'='*60}")
    print("FITTING ELITE MAPPING")
    print(f"{'='*60}")
    print(f"Fit data: {len(fit_data)} obs (GW1-12)")
    print(f"Val data: {len(val_data)} obs (GW13-23)")

    # Method 1: Isotonic regression (monotonic, non-parametric)
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    iso.fit(fit_data['crowd_pct'].values, fit_data['elite_pct'].values)

    # Evaluate on validation set
    val_pred = iso.predict(val_data['crowd_pct'].values)
    val_mae = np.mean(np.abs(val_pred - val_data['elite_pct'].values))
    val_rmse = np.sqrt(np.mean((val_pred - val_data['elite_pct'].values) ** 2))

    # In-sample fit
    fit_pred = iso.predict(fit_data['crowd_pct'].values)
    fit_mae = np.mean(np.abs(fit_pred - fit_data['elite_pct'].values))

    print(f"\nIsotonic Regression:")
    print(f"  Fit MAE:  {fit_mae:.4f}")
    print(f"  Val MAE:  {val_mae:.4f}")
    print(f"  Val RMSE: {val_rmse:.4f}")

    # Method 2: Binned quantile mapping (simpler, more robust)
    # Create 20 equal-frequency bins and map crowd_pct centroid → median elite_pct
    fit_data = fit_data.copy()
    fit_data['q_bin'] = pd.qcut(fit_data['crowd_pct'], q=20, duplicates='drop')
    bin_map = fit_data.groupby('q_bin', observed=True).agg(
        crowd_mid=('crowd_pct', 'median'),
        elite_mid=('elite_pct', 'median'),
    ).sort_values('crowd_mid')

    # Linear interpolation between bin midpoints
    interp_fn = interp1d(
        bin_map['crowd_mid'].values,
        bin_map['elite_mid'].values,
        kind='linear',
        bounds_error=False,
        fill_value=(bin_map['elite_mid'].iloc[0], bin_map['elite_mid'].iloc[-1]),
    )

    val_pred_interp = interp_fn(val_data['crowd_pct'].values)
    val_mae_interp = np.mean(np.abs(val_pred_interp - val_data['elite_pct'].values))

    print(f"\nBinned Interpolation:")
    print(f"  Val MAE:  {val_mae_interp:.4f}")

    # Method 3: Polynomial fit (degree 3)
    coeffs = np.polyfit(fit_data['crowd_pct'].values, fit_data['elite_pct'].values, deg=3)
    poly_fn = np.poly1d(coeffs)
    val_pred_poly = np.clip(poly_fn(val_data['crowd_pct'].values), 0, 1)
    val_mae_poly = np.mean(np.abs(val_pred_poly - val_data['elite_pct'].values))

    print(f"\nPolynomial (deg=3):")
    print(f"  Val MAE:  {val_mae_poly:.4f}")
    print(f"  Coefficients: {coeffs}")

    # Pick best method
    methods = {
        'isotonic': val_mae,
        'binned_interp': val_mae_interp,
        'polynomial': val_mae_poly,
    }
    best_method = min(methods, key=methods.get)
    print(f"\nBest method: {best_method} (Val MAE = {methods[best_method]:.4f})")

    # Refit on ALL data for final model
    iso_full = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    iso_full.fit(valid['crowd_pct'].values, valid['elite_pct'].values)

    # Also refit polynomial on all data
    coeffs_full = np.polyfit(valid['crowd_pct'].values, valid['elite_pct'].values, deg=3)

    # Stability check: compare GW1-12 vs GW13-23 mapping shapes
    print(f"\n--- Stability Check ---")
    test_points = np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.50])
    print(f"{'Crowd %':<10} {'GW1-12':>10} {'GW13-23':>10} {'All':>10} {'Diff':>10}")
    print("-" * 55)

    iso_early = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    iso_early.fit(fit_data['crowd_pct'].values, fit_data['elite_pct'].values)

    iso_late = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    late_data = valid[valid['gw'] > 12]
    iso_late.fit(late_data['crowd_pct'].values, late_data['elite_pct'].values)

    for p in test_points:
        early_pred = iso_early.predict([p])[0]
        late_pred = iso_late.predict([p])[0]
        full_pred = iso_full.predict([p])[0]
        diff = abs(early_pred - late_pred)
        print(f"{p*100:>8.0f}% {early_pred:>10.3f} {late_pred:>10.3f} {full_pred:>10.3f} {diff:>10.3f}")

    return {
        'iso_model': iso_full,
        'poly_coeffs': coeffs_full,
        'best_method': best_method,
        'val_metrics': methods,
    }


def plot_distortion_curve(cal: pd.DataFrame, mapping: dict):
    """Plot the crowd→elite distortion curve."""
    setup_chart_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    valid = cal[(cal['crowd_pct'] > 0.001)].copy()

    # Left: scatter + fitted curve
    ax = axes[0]
    ax.scatter(valid['crowd_pct'] * 100, valid['elite_pct'] * 100,
               alpha=0.05, s=3, color=CHART_COLORS['teal'], label='Player-GW obs')

    x_range = np.linspace(0, valid['crowd_pct'].max(), 200)
    iso_pred = mapping['iso_model'].predict(x_range)
    ax.plot(x_range * 100, iso_pred * 100, color=CHART_COLORS['coral'],
            linewidth=2.5, label='Isotonic fit')
    ax.plot([0, 100], [0, 100], '--', color=CHART_COLORS['white'],
            alpha=0.3, label='Identity (no distortion)')

    ax.set_xlabel('Crowd Ownership %')
    ax.set_ylabel('Elite (Top 50K) Ownership %')
    ax.set_title('Crowd → Elite Ownership Mapping')
    ax.legend(loc='upper left')
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 100)

    # Right: ratio by crowd ownership bin
    ax = axes[1]
    valid['ratio'] = valid['elite_pct'] / (valid['crowd_pct'] + 1e-6)
    bins = np.arange(0, 0.65, 0.05)
    valid['bin'] = pd.cut(valid['crowd_pct'], bins=bins)
    bin_stats = valid.groupby('bin', observed=True)['ratio'].agg(['mean', 'median', 'std'])

    bin_centers = [(b.left + b.right) / 2 * 100 for b in bin_stats.index]
    ax.bar(bin_centers, bin_stats['median'], width=4, color=CHART_COLORS['purple'],
           alpha=0.8, label='Median ratio')
    ax.errorbar(bin_centers, bin_stats['median'], yerr=bin_stats['std'],
                fmt='none', color=CHART_COLORS['white'], alpha=0.3)
    ax.axhline(1.0, color=CHART_COLORS['white'], linestyle='--', alpha=0.3, label='No distortion')

    ax.set_xlabel('Crowd Ownership %')
    ax.set_ylabel('Elite / Crowd Ratio')
    ax.set_title('Elite Amplification by Ownership Level')
    ax.legend()
    ax.set_ylim(0, max(bin_stats['median'].max() * 1.3, 3))

    plt.tight_layout()
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(CHARTS_DIR / 'elite_distortion_curve.png')
    plt.close()
    print(f"Saved: {CHARTS_DIR / 'elite_distortion_curve.png'}")


# ================================================================
# STEP 9c: Apply Mapping to Historical Data
# ================================================================

def _estimate_total_managers(season: str, merged_gw: pd.DataFrame, pr_path: Path) -> float:
    """Estimate total FPL managers for a season by cross-referencing selected counts with percentages.

    Uses players_raw.csv (end-of-season snapshot with selected_by_percent) to calibrate
    the raw 'selected' counts from merged_gw.csv.
    """
    if not pr_path.exists():
        return 0

    pr = pd.read_csv(pr_path, encoding='latin-1', low_memory=False)
    pr.columns = [c.strip().strip('"') for c in pr.columns]

    if 'selected_by_percent' not in pr.columns:
        return 0

    pr['sbp'] = pd.to_numeric(pr['selected_by_percent'], errors='coerce')
    pr['id_int'] = pd.to_numeric(pr['id'], errors='coerce').astype('Int64')

    # Use the last GW to cross-reference (closest to end-of-season snapshot)
    last_gw = merged_gw['gw'].max()
    last_gw_data = merged_gw[merged_gw['gw'] == last_gw].copy()

    # Merge on player ID
    cross = last_gw_data.merge(pr[['id_int', 'sbp']], left_on='element', right_on='id_int', how='inner')
    valid = cross[(cross['selected'] > 1000) & (cross['sbp'] > 0.1)]

    if len(valid) < 10:
        return 0

    # N = selected / (sbp / 100)
    valid = valid.copy()
    valid['est_N'] = valid['selected'] / (valid['sbp'] / 100)
    total_managers = valid['est_N'].median()

    print(f"    Estimated total managers for {season}: {total_managers:,.0f} "
          f"(from {len(valid)} players)")
    return total_managers


def apply_elite_adjustment(mapping: dict) -> pd.DataFrame:
    """Apply the elite mapping to all 9 seasons of player data and rebuild match signals."""
    iso = mapping['iso_model']

    matches = load_parquet('matches.parquet')
    signals = load_parquet('signals.parquet')

    all_elite_signals = []

    for season in SEASONS:
        print(f"  Adjusting {season}...")
        path = DATA_RAW_FPL / f'{season}_merged_gw.csv'
        if not path.exists():
            print(f"    SKIP: no merged_gw for {season}")
            continue

        df = pd.read_csv(path, encoding='latin-1', low_memory=False)
        df.columns = [c.strip().strip('"') for c in df.columns]

        # Numeric coercion
        for col in ['element', 'GW', 'value', 'selected', 'transfers_in',
                     'transfers_out', 'transfers_balance', 'fixture',
                     'opponent_team', 'team_h_score', 'team_a_score']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'transfers_balance' not in df.columns:
            df['transfers_balance'] = df['transfers_in'] - df['transfers_out']

        if df['was_home'].dtype == object:
            df['was_home'] = df['was_home'].astype(str).str.strip().str.lower() == 'true'

        df['gw'] = df['GW'].astype('Int64')

        # Compute crowd_pct (overall ownership %) for each player-GW
        # merged_gw has 'selected' (raw count), not 'selected_by_percent'
        # We estimate total managers per-GW to convert count → percentage
        if 'selected_by_percent' in df.columns:
            df['crowd_pct'] = pd.to_numeric(df['selected_by_percent'], errors='coerce') / 100.0
        else:
            # Estimate total managers from players_raw cross-reference
            pr_path = DATA_RAW_FPL / f'{season}_players_raw.csv'
            total_managers = _estimate_total_managers(season, df, pr_path)
            if total_managers > 0:
                df['crowd_pct'] = df['selected'] / total_managers
            else:
                print(f"    WARNING: cannot estimate total managers for {season}")
                df['crowd_pct'] = np.nan

        # Apply the mapping: crowd_pct → elite_pct
        valid_mask = df['crowd_pct'].notna() & (df['crowd_pct'] > 0)
        df['elite_pct'] = np.nan
        if valid_mask.any():
            df.loc[valid_mask, 'elite_pct'] = iso.predict(
                df.loc[valid_mask, 'crowd_pct'].values
            )

        # Reconstruct "elite_selected" count by scaling
        # elite_selected = elite_pct * (selected / crowd_pct)
        # This preserves the scale relative to original data
        df['elite_selected'] = np.where(
            (df['crowd_pct'] > 0.001) & df['elite_pct'].notna(),
            df['elite_pct'] * (df['selected'] / (df['crowd_pct'] + 1e-6)),
            df['selected']  # fallback to original
        )

        # Assign team IDs (reuse logic from original script)
        season_matches = matches[matches['season'] == season]
        if 'team' in df.columns and df['team'].notna().any():
            team_name_to_id = {}
            for _, row in season_matches.iterrows():
                team_name_to_id[row['team_h_name']] = int(row['team_h_id'])
                team_name_to_id[row['team_a_name']] = int(row['team_a_id'])
            df['team_id'] = df['team'].map(team_name_to_id)
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
        df = df[df['team_id'] > 0]

        # Get element_type for DEF/GK filtering
        pr_path = DATA_RAW_FPL / f'{season}_players_raw.csv'
        if pr_path.exists() and 'element_type' not in df.columns:
            pr = pd.read_csv(pr_path, encoding='latin-1', low_memory=False)
            pr.columns = [c.strip().strip('"') for c in pr.columns]
            if 'element_type' in pr.columns:
                et_map = dict(zip(pr['id'].astype(int), pr['element_type'].astype(int)))
                df['element_type'] = df['element'].map(et_map)

        # Build match-level elite signals
        for _, match in season_matches.iterrows():
            gw = match['gw']
            th_id, ta_id = match['team_h_id'], match['team_a_id']
            fid = match['fixture_id']

            hp = df[(df['gw'] == gw) & (df['team_id'] == th_id)]
            ap = df[(df['gw'] == gw) & (df['team_id'] == ta_id)]

            if len(hp) == 0 or len(ap) == 0:
                all_elite_signals.append({
                    'season': season, 'gw': gw, 'fixture_id': fid,
                    'elite_ownership_ratio': np.nan,
                    'elite_transfer_ratio': np.nan,
                    'elite_dcs_ratio': np.nan,
                })
                continue

            # Elite Ownership Ratio (price-weighted)
            eown_h = (hp['elite_selected'] * hp['value']).sum()
            eown_a = (ap['elite_selected'] * ap['value']).sum()
            elite_own_ratio = eown_h / (eown_h + eown_a + 1)

            # Elite Transfer Ratio (weight transfers by elite adjustment factor)
            # The adjustment factor = elite_pct / crowd_pct
            hp = hp.copy()
            ap = ap.copy()
            hp['elite_factor'] = np.where(
                hp['crowd_pct'] > 0.001,
                hp['elite_pct'] / (hp['crowd_pct'] + 1e-6),
                1.0
            )
            ap['elite_factor'] = np.where(
                ap['crowd_pct'] > 0.001,
                ap['elite_pct'] / (ap['crowd_pct'] + 1e-6),
                1.0
            )
            # Weight each player's net transfers by how much elites over/under-weight them
            adj_trf_h = (hp['transfers_balance'] * hp['elite_factor']).sum()
            adj_trf_a = (ap['transfers_balance'] * ap['elite_factor']).sum()
            elite_trf_ratio = adj_trf_h / (abs(adj_trf_h) + abs(adj_trf_a) + 1)

            # Elite DCS (DEF+GK) Ratio
            if 'element_type' in hp.columns:
                hp_def = hp[hp['element_type'].isin([1, 2])]
                ap_def = ap[ap['element_type'].isin([1, 2])]
                edcs_h = hp_def['elite_selected'].sum()
                edcs_a = ap_def['elite_selected'].sum()
                elite_dcs = edcs_h / (edcs_h + edcs_a + 1)
            else:
                elite_dcs = np.nan

            all_elite_signals.append({
                'season': season, 'gw': gw, 'fixture_id': fid,
                'elite_ownership_ratio': elite_own_ratio,
                'elite_transfer_ratio': elite_trf_ratio,
                'elite_dcs_ratio': elite_dcs,
            })

    elite_df = pd.DataFrame(all_elite_signals)
    print(f"\nElite signals computed for {len(elite_df)} matches")

    # Merge with existing signals
    merged = signals.merge(elite_df, on=['season', 'gw', 'fixture_id'], how='left')
    return merged


# ================================================================
# STEP 9d: Train & Compare Models
# ================================================================

def multiclass_brier(y_true, y_proba, n_classes=3):
    return np.mean([
        brier_score_loss((y_true == c).astype(int), y_proba[:, c])
        for c in range(n_classes)
    ])


def train_and_compare(merged: pd.DataFrame):
    """Train models with elite-adjusted signals and compare to baselines."""
    ODDS_FEATURES = ['implied_prob_h', 'implied_prob_d', 'implied_prob_a']
    CROWD_FPL = ['transfer_ratio', 'ownership_ratio', 'captain_proxy', 'dcs_ratio', 'velocity_delta']
    ELITE_FPL = ['elite_ownership_ratio', 'elite_transfer_ratio', 'elite_dcs_ratio']

    feature_sets = {
        'odds_only': ODDS_FEATURES,
        'crowd_fpl': CROWD_FPL,
        'elite_fpl': ELITE_FPL,
        'crowd+odds': ODDS_FEATURES + CROWD_FPL,
        'elite+odds': ODDS_FEATURES + ELITE_FPL,
        'elite+crowd': CROWD_FPL + ELITE_FPL,
        'all_features': ODDS_FEATURES + CROWD_FPL + ELITE_FPL,
    }

    # Prepare data
    all_features = list(set(ODDS_FEATURES + CROWD_FPL + ELITE_FPL))
    clean = merged.dropna(subset=all_features + ['result'])

    train = clean[clean['season'].isin(TRAIN_SEASONS)]
    val = clean[clean['season'].isin(VAL_SEASONS)]
    test = clean[clean['season'].isin(TEST_SEASONS)]

    print(f"\n{'='*70}")
    print("MODEL COMPARISON: ELITE vs CROWD vs ODDS")
    print(f"{'='*70}")
    print(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")

    RESULT_MAP = {'A': 0, 'D': 1, 'H': 2}
    y_train = train['result'].map(RESULT_MAP).values
    y_val = val['result'].map(RESULT_MAP).values
    y_test = test['result'].map(RESULT_MAP).values

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(train[all_features]),
                           columns=all_features, index=train.index)
    X_val = pd.DataFrame(scaler.transform(val[all_features]),
                         columns=all_features, index=val.index)
    X_test = pd.DataFrame(scaler.transform(test[all_features]),
                          columns=all_features, index=test.index)

    print(f"\n{'Model':<25} {'Val Brier':>10} {'Test Brier':>11} {'Val Acc':>8} "
          f"{'Test Acc':>9} {'BSS vs Odds':>12}")
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

    # Print with BSS
    odds_test_brier = results['odds_only']['test_brier']
    for name, r in results.items():
        bss = 1 - (r['test_brier'] / odds_test_brier) if name != 'odds_only' else 0
        print(f"{name:<25} {r['val_brier']:>10.4f} {r['test_brier']:>11.4f} "
              f"{r['val_acc']:>8.3f} {r['test_acc']:>9.3f} {bss:>+12.4f}")

    # Naive baseline
    naive_brier = multiclass_brier(y_test, np.tile(
        [(y_train == 0).mean(), (y_train == 1).mean(), (y_train == 2).mean()],
        (len(y_test), 1)
    ))

    print(f"\nNaive baseline test Brier: {naive_brier:.4f}")
    print(f"\nSkill vs Naive (% of gap closed):")
    odds_skill = 1 - odds_test_brier / naive_brier
    for name, r in results.items():
        skill = 1 - r['test_brier'] / naive_brier
        pct_of_odds = (skill / odds_skill * 100) if odds_skill > 0 else 0
        marker = " <<<" if pct_of_odds > 100 else ""
        print(f"  {name:<25} skill={skill*100:>5.1f}%  ({pct_of_odds:>5.1f}% of odds skill){marker}")

    # Signal correlations
    print(f"\n--- Signal Correlations with Implied Prob (Home Win) ---")
    all_sig_cols = CROWD_FPL + ELITE_FPL
    for col in all_sig_cols:
        valid = merged[['implied_prob_h', col]].dropna()
        r = valid['implied_prob_h'].corr(valid[col])
        print(f"  {col:<30} r = {r:>7.4f}")

    # Cross-signal correlations (elite vs crowd)
    print(f"\n--- Elite vs Crowd Signal Correlations ---")
    pairs = [
        ('ownership_ratio', 'elite_ownership_ratio'),
        ('transfer_ratio', 'elite_transfer_ratio'),
        ('dcs_ratio', 'elite_dcs_ratio'),
    ]
    for crowd_col, elite_col in pairs:
        valid = merged[[crowd_col, elite_col]].dropna()
        r = valid[crowd_col].corr(valid[elite_col])
        print(f"  {crowd_col:<25} ↔ {elite_col:<25} r = {r:>7.4f}")

    return results


# ================================================================
# Main
# ================================================================

def main():
    print("=" * 70)
    print("ELITE NOISE COEFFICIENT ANALYSIS")
    print("Estimating top-manager ownership from crowd data")
    print("=" * 70)

    # Step 9a: Download calibration data
    print("\n[1/5] Downloading calibration data from fplAnalytics...")
    download_elite_csvs()

    # Step 9a: Load and analyze
    print("\n[2/5] Loading calibration data...")
    cal = load_calibration_data()
    distortion_stats = analyze_distortion(cal)

    # Step 9b: Fit mapping
    print("\n[3/5] Fitting elite mapping...")
    mapping = fit_elite_mapping(cal)

    # Plot
    plot_distortion_curve(cal, mapping)

    # Step 9c: Apply to historical data
    print("\n[4/5] Applying elite adjustment to all seasons...")
    merged = apply_elite_adjustment(mapping)

    # Save
    save_parquet(merged, 'signals_with_elite.parquet')

    # Step 9d: Train and compare
    print("\n[5/5] Training models...")
    results = train_and_compare(merged)

    # Save results
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / 'elite_comparison.json', 'w') as f:
        json.dump({k: {kk: float(vv) for kk, vv in v.items()}
                   for k, v in results.items()}, f, indent=2)

    print(f"\nResults saved to {MODEL_DIR / 'elite_comparison.json'}")
    print("Done!")


if __name__ == '__main__':
    main()
