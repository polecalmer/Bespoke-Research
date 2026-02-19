"""Compute FPL signals per match from the match-level dataset.

Signals:
1. Transfer Momentum Score (TMS) → transfer_ratio
2. Ownership Conviction Score (OCS) → ownership_ratio
3. Captaincy proxy via transfer velocity → captain_proxy
4. Defensive Confidence Score (DCS) → dcs_ratio
5. Ownership Velocity (OV) → velocity_delta

Outputs: data/processed/signals.parquet
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_parquet, save_parquet


def compute_transfer_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Signal 1: Transfer Momentum Score — normalized net transfers ratio."""
    tms_h = df['net_transfers_h']
    tms_a = df['net_transfers_a']

    df['transfer_delta'] = tms_h - tms_a
    df['transfer_ratio'] = tms_h / (tms_h.abs() + tms_a.abs() + 1)
    return df


def compute_ownership_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Signal 2: Ownership Conviction Score — price-weighted ownership ratio."""
    ocs_h = df['price_weighted_selected_h']
    ocs_a = df['price_weighted_selected_a']

    df['ownership_ratio'] = ocs_h / (ocs_h + ocs_a + 1)
    return df


def compute_captain_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Signal 3: Captaincy proxy — transfer velocity differential."""
    tv_h = df['net_transfers_h'] / (df['total_selected_h'] + 1)
    tv_a = df['net_transfers_a'] / (df['total_selected_a'] + 1)

    df['captain_proxy'] = tv_h - tv_a
    return df


def compute_dcs(df: pd.DataFrame) -> pd.DataFrame:
    """Signal 4: Defensive Confidence Score — DEF+GK ownership ratio."""
    dcs_h = df['def_gk_selected_h']
    dcs_a = df['def_gk_selected_a']

    df['dcs_ratio'] = dcs_h / (dcs_h + dcs_a + 1)
    return df


def compute_ownership_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Signal 5: Ownership Velocity — week-over-week change in team ownership.

    Build a team-gameweek panel, compute lag, then join back to matches.
    """
    # Unpivot: create team-gw records from both home and away perspectives
    home = df[['season', 'gw', 'team_h_name', 'total_selected_h']].rename(
        columns={'team_h_name': 'team', 'total_selected_h': 'total_selected'})
    away = df[['season', 'gw', 'team_a_name', 'total_selected_a']].rename(
        columns={'team_a_name': 'team', 'total_selected_a': 'total_selected'})

    team_gw = pd.concat([home, away], ignore_index=True)

    # Deduplicate: a team plays once per GW normally, but double GWs exist
    team_gw = team_gw.groupby(['season', 'team', 'gw'])['total_selected'].first().reset_index()
    team_gw = team_gw.sort_values(['season', 'team', 'gw'])

    # Compute lagged value (previous gameweek for same team within same season)
    team_gw['prev_selected'] = team_gw.groupby(['season', 'team'])['total_selected'].shift(1)
    team_gw['ownership_velocity'] = team_gw['total_selected'] - team_gw['prev_selected']

    # GW1: no previous → set velocity to 0
    team_gw['ownership_velocity'] = team_gw['ownership_velocity'].fillna(0)

    # Join back: home team velocity
    ov_h = team_gw[['season', 'team', 'gw', 'ownership_velocity']].rename(
        columns={'team': 'team_h_name', 'ownership_velocity': 'ov_h'})
    df = df.merge(ov_h, on=['season', 'team_h_name', 'gw'], how='left')

    # Away team velocity
    ov_a = team_gw[['season', 'team', 'gw', 'ownership_velocity']].rename(
        columns={'team': 'team_a_name', 'ownership_velocity': 'ov_a'})
    df = df.merge(ov_a, on=['season', 'team_a_name', 'gw'], how='left')

    df['ov_h'] = df['ov_h'].fillna(0)
    df['ov_a'] = df['ov_a'].fillna(0)
    df['velocity_delta'] = df['ov_h'] - df['ov_a']

    return df


def main():
    print("Engineering FPL signals...")

    df = load_parquet('matches.parquet')
    print(f"Loaded {len(df)} matches")

    # Compute all 5 signals
    df = compute_transfer_signals(df)
    df = compute_ownership_signals(df)
    df = compute_captain_proxy(df)
    df = compute_dcs(df)
    df = compute_ownership_velocity(df)

    # Report signal statistics
    signal_cols = ['transfer_ratio', 'ownership_ratio', 'captain_proxy', 'dcs_ratio', 'velocity_delta']
    print("\nSignal statistics:")
    for col in signal_cols:
        s = df[col]
        print(f"  {col:<20} mean={s.mean():>10.4f}  std={s.std():>10.4f}  "
              f"min={s.min():>12.4f}  max={s.max():>12.4f}  NaN={s.isna().sum()}")

    # Check correlation with implied odds
    print("\nCorrelation of signals with implied_prob_h:")
    for col in signal_cols:
        valid = df[['implied_prob_h', col]].dropna()
        corr = valid['implied_prob_h'].corr(valid[col])
        print(f"  {col:<20} r={corr:>7.4f}")

    save_parquet(df, 'signals.parquet')


if __name__ == '__main__':
    main()
