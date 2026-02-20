"""Build match-level dataset: one row per EPL match with FPL aggregates + betting odds.

Handles schema differences across 9 seasons (2016-17 to 2024-25).
Outputs: data/processed/matches.parquet
"""

import sys
import json
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEASONS, DATA_RAW_FPL, DATA_RAW_ODDS, DATA_PROCESSED, DATA_REFERENCE,
    FPL_TO_ODDS, ODDS_TO_FPL, odds_to_implied_probs, save_parquet,
)

warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

# Seasons where merged_gw.csv has 'team' column
SEASONS_WITH_TEAM_COL = ['2020-21', '2021-22', '2022-23', '2023-24', '2024-25']
# Seasons with fixtures.csv
SEASONS_WITH_FIXTURES = ['2018-19', '2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25']


def load_team_mappings():
    """Load season_team_ids from team_mappings.json."""
    path = DATA_REFERENCE / 'team_mappings.json'
    with open(path) as f:
        return json.load(f)['season_team_ids']


def load_merged_gw(season: str) -> pd.DataFrame:
    """Load and normalize merged_gw.csv for a season."""
    path = DATA_RAW_FPL / f'{season}_merged_gw.csv'
    df = pd.read_csv(path, encoding='latin-1', low_memory=False)

    # Strip quotes from column names
    df.columns = [c.strip().strip('"') for c in df.columns]

    # Ensure GW column is int
    if 'GW' in df.columns:
        df['gw'] = pd.to_numeric(df['GW'], errors='coerce').astype('Int64')
    elif 'round' in df.columns:
        df['gw'] = pd.to_numeric(df['round'], errors='coerce').astype('Int64')

    # Ensure key numeric columns
    for col in ['element', 'fixture', 'opponent_team', 'selected',
                'transfers_in', 'transfers_out', 'value',
                'team_h_score', 'team_a_score', 'total_points']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Ensure was_home is bool
    if df['was_home'].dtype == object:
        df['was_home'] = df['was_home'].astype(str).str.strip().str.lower() == 'true'

    # Compute transfers_balance if missing
    if 'transfers_balance' not in df.columns:
        df['transfers_balance'] = df['transfers_in'] - df['transfers_out']

    df['season'] = season
    return df


def load_players_raw(season: str) -> pd.DataFrame:
    """Load players_raw.csv: element -> (team_id, element_type, now_cost)."""
    path = DATA_RAW_FPL / f'{season}_players_raw.csv'
    if not path.exists():
        return pd.DataFrame(columns=['element', 'team_id', 'element_type', 'now_cost'])

    df = pd.read_csv(path, encoding='latin-1', low_memory=False)
    df.columns = [c.strip().strip('"') for c in df.columns]

    result = pd.DataFrame()
    result['element'] = df['id'].astype(int)
    result['team_id'] = df['team'].astype(int) if 'team' in df.columns else -1
    result['element_type'] = df['element_type'].astype(int) if 'element_type' in df.columns else 0
    result['now_cost'] = pd.to_numeric(df.get('now_cost', 50), errors='coerce').fillna(50)

    return result


def load_fixtures(season: str) -> pd.DataFrame | None:
    """Load fixtures.csv for a season."""
    path = DATA_RAW_FPL / f'{season}_fixtures.csv'
    if not path.exists():
        return None

    # Read only needed columns to avoid issues with embedded JSON in stats column
    try:
        df = pd.read_csv(path, encoding='latin-1', low_memory=False,
                         usecols=lambda c: c in ['id', 'event', 'team_h', 'team_a',
                                                  'team_h_score', 'team_a_score', 'kickoff_time'])
    except ValueError:
        df = pd.read_csv(path, encoding='latin-1', low_memory=False)
        cols_needed = ['id', 'event', 'team_h', 'team_a', 'team_h_score', 'team_a_score', 'kickoff_time']
        cols_available = [c for c in cols_needed if c in df.columns]
        df = df[cols_available]

    for col in ['id', 'event', 'team_h', 'team_a', 'team_h_score', 'team_a_score']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def assign_team_from_fixtures(gw_df: pd.DataFrame, fixtures_df: pd.DataFrame) -> pd.DataFrame:
    """Assign team_id using fixtures.csv: join on fixture ID, derive from was_home."""
    fix = fixtures_df[['id', 'team_h', 'team_a']].rename(columns={'id': 'fixture'})
    merged = gw_df.merge(fix, on='fixture', how='left')

    merged['team_id'] = np.where(
        merged['was_home'],
        merged['team_h'],
        merged['team_a']
    )
    merged = merged.drop(columns=['team_h', 'team_a'], errors='ignore')
    return merged


def assign_team_from_grouping(gw_df: pd.DataFrame) -> pd.DataFrame:
    """Assign team_id by grouping players by fixture — for seasons without fixtures.csv."""
    # For each (fixture, gw) group:
    # - Home players' opponent_team = away team ID
    # - Away players' opponent_team = home team ID
    # - So: home player's team = away player's opponent_team (and vice versa)

    fixture_teams = []
    for (fix_id, gw), group in gw_df.groupby(['fixture', 'gw']):
        home = group[group['was_home'] == True]
        away = group[group['was_home'] == False]

        if len(home) > 0 and len(away) > 0:
            home_team_id = away['opponent_team'].iloc[0]
            away_team_id = home['opponent_team'].iloc[0]
        elif len(home) > 0:
            # Only home players available — use opponent_team as away team
            away_team_id = home['opponent_team'].iloc[0]
            home_team_id = -1  # will try to recover from players_raw
        elif len(away) > 0:
            home_team_id = away['opponent_team'].iloc[0]
            away_team_id = -1
        else:
            continue

        fixture_teams.append({
            'fixture': fix_id, 'gw': gw,
            'home_team_id': home_team_id, 'away_team_id': away_team_id,
        })

    fix_df = pd.DataFrame(fixture_teams)
    merged = gw_df.merge(fix_df, on=['fixture', 'gw'], how='left')
    merged['team_id'] = np.where(
        merged['was_home'],
        merged['home_team_id'],
        merged['away_team_id']
    )
    merged = merged.drop(columns=['home_team_id', 'away_team_id'], errors='ignore')
    return merged


def assign_team_and_position(gw_df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Assign team_id and element_type to each player-row."""
    players = load_players_raw(season)

    # --- Team assignment ---
    if season in SEASONS_WITH_TEAM_COL:
        # merged_gw has 'team' column (string name) — map to int ID
        team_mapping = load_team_mappings().get(season, {})
        name_to_id = {v: int(k) for k, v in team_mapping.items()}
        if 'team' in gw_df.columns:
            gw_df['team_id'] = gw_df['team'].map(name_to_id)
        else:
            gw_df['team_id'] = np.nan
    elif season in SEASONS_WITH_FIXTURES:
        fixtures = load_fixtures(season)
        if fixtures is not None:
            gw_df = assign_team_from_fixtures(gw_df, fixtures)
        else:
            gw_df = assign_team_from_grouping(gw_df)
    else:
        # 2016-17, 2017-18: no fixtures.csv, no team column
        gw_df = assign_team_from_grouping(gw_df)

    # Fill missing team_ids from players_raw (fallback)
    if 'team_id' in gw_df.columns:
        missing = gw_df['team_id'].isna()
        if missing.any() and len(players) > 0:
            pr_map = dict(zip(players['element'], players['team_id']))
            gw_df.loc[missing, 'team_id'] = gw_df.loc[missing, 'element'].map(pr_map)

    # --- Position (element_type) assignment ---
    if season in SEASONS_WITH_TEAM_COL and 'position' in gw_df.columns:
        pos_map = {'GK': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}
        gw_df['element_type'] = gw_df['position'].map(pos_map)
    else:
        # From players_raw
        if len(players) > 0:
            et_map = dict(zip(players['element'], players['element_type']))
            gw_df['element_type'] = gw_df['element'].map(et_map)
        else:
            gw_df['element_type'] = 0

    # Fill missing element_type
    gw_df['element_type'] = gw_df['element_type'].fillna(0).astype(int)
    gw_df['team_id'] = gw_df['team_id'].fillna(-1).astype(int)

    return gw_df


def build_fixture_info(gw_df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Build fixture-level info: (fixture, gw, team_h_id, team_a_id, team_h_score, team_a_score, kickoff_time)."""
    # Group by fixture and gw
    fixtures = []
    for (fix_id, gw), group in gw_df.groupby(['fixture', 'gw']):
        home = group[group['was_home'] == True]
        away = group[group['was_home'] == False]

        if len(home) == 0 or len(away) == 0:
            continue

        team_h_id = home['team_id'].iloc[0]
        team_a_id = away['team_id'].iloc[0]
        team_h_score = home['team_h_score'].iloc[0] if 'team_h_score' in home.columns else np.nan
        team_a_score = home['team_a_score'].iloc[0] if 'team_a_score' in home.columns else np.nan
        kickoff_time = home['kickoff_time'].iloc[0] if 'kickoff_time' in home.columns else ''

        fixtures.append({
            'season': season,
            'gw': gw,
            'fixture_id': fix_id,
            'team_h_id': int(team_h_id),
            'team_a_id': int(team_a_id),
            'team_h_score': team_h_score,
            'team_a_score': team_a_score,
            'kickoff_time': kickoff_time,
        })

    return pd.DataFrame(fixtures)


def aggregate_fpl_per_team(gw_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player-level data to (season, gw, fixture, team_id) level."""
    # Basic aggregates
    agg_df = gw_df.groupby(['season', 'gw', 'fixture', 'team_id']).agg(
        total_selected=('selected', 'sum'),
        total_transfers_in=('transfers_in', 'sum'),
        total_transfers_out=('transfers_out', 'sum'),
        net_transfers=('transfers_balance', 'sum'),
        num_players=('element', 'count'),
    ).reset_index()

    # Price-weighted selected: sum(selected * value)
    gw_df['_price_weighted'] = gw_df['selected'] * gw_df['value']
    pw = gw_df.groupby(['season', 'gw', 'fixture', 'team_id'])['_price_weighted'].sum().reset_index()
    pw.columns = ['season', 'gw', 'fixture', 'team_id', 'price_weighted_selected']
    agg_df = agg_df.merge(pw, on=['season', 'gw', 'fixture', 'team_id'], how='left')

    # DEF + GK selected (element_type 1 or 2)
    def_gk = gw_df[gw_df['element_type'].isin([1, 2])].groupby(
        ['season', 'gw', 'fixture', 'team_id']
    )['selected'].sum().reset_index()
    def_gk.columns = ['season', 'gw', 'fixture', 'team_id', 'def_gk_selected']
    agg_df = agg_df.merge(def_gk, on=['season', 'gw', 'fixture', 'team_id'], how='left')
    agg_df['def_gk_selected'] = agg_df['def_gk_selected'].fillna(0)

    # Clean up temp column
    gw_df.drop(columns=['_price_weighted'], inplace=True, errors='ignore')

    return agg_df


def build_match_rows(fix_info: pd.DataFrame, team_agg: pd.DataFrame,
                     team_names: dict) -> pd.DataFrame:
    """Combine fixture info with team aggregates into one row per match."""
    # Merge home team aggregates
    home = team_agg.rename(columns={
        'total_selected': 'total_selected_h',
        'total_transfers_in': 'total_transfers_in_h',
        'total_transfers_out': 'total_transfers_out_h',
        'net_transfers': 'net_transfers_h',
        'num_players': 'num_players_h',
        'price_weighted_selected': 'price_weighted_selected_h',
        'def_gk_selected': 'def_gk_selected_h',
    })
    home = home.rename(columns={'team_id': 'team_h_id'})

    matches = fix_info.merge(
        home[['season', 'gw', 'fixture', 'team_h_id',
              'total_selected_h', 'total_transfers_in_h', 'total_transfers_out_h',
              'net_transfers_h', 'num_players_h', 'price_weighted_selected_h',
              'def_gk_selected_h']],
        left_on=['season', 'gw', 'fixture_id', 'team_h_id'],
        right_on=['season', 'gw', 'fixture', 'team_h_id'],
        how='left'
    ).drop(columns=['fixture'], errors='ignore')

    # Merge away team aggregates
    away = team_agg.rename(columns={
        'total_selected': 'total_selected_a',
        'total_transfers_in': 'total_transfers_in_a',
        'total_transfers_out': 'total_transfers_out_a',
        'net_transfers': 'net_transfers_a',
        'num_players': 'num_players_a',
        'price_weighted_selected': 'price_weighted_selected_a',
        'def_gk_selected': 'def_gk_selected_a',
    })
    away = away.rename(columns={'team_id': 'team_a_id'})

    matches = matches.merge(
        away[['season', 'gw', 'fixture', 'team_a_id',
              'total_selected_a', 'total_transfers_in_a', 'total_transfers_out_a',
              'net_transfers_a', 'num_players_a', 'price_weighted_selected_a',
              'def_gk_selected_a']],
        left_on=['season', 'gw', 'fixture_id', 'team_a_id'],
        right_on=['season', 'gw', 'fixture', 'team_a_id'],
        how='left'
    ).drop(columns=['fixture'], errors='ignore')

    # Add team names
    matches['team_h_name'] = matches['team_h_id'].astype(str).map(team_names)
    matches['team_a_name'] = matches['team_a_id'].astype(str).map(team_names)

    # Derive result
    matches['result'] = np.where(
        matches['team_h_score'] > matches['team_a_score'], 'H',
        np.where(matches['team_h_score'] == matches['team_a_score'], 'D', 'A')
    )

    return matches


def load_odds(season: str) -> pd.DataFrame | None:
    """Load odds CSV for a season."""
    path = DATA_RAW_ODDS / f'{season}.csv'
    if not path.exists():
        return None

    df = pd.read_csv(path)
    # Parse date
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

    # Ensure numeric odds
    for col in ['B365H', 'B365D', 'B365A', 'FTHG', 'FTAG']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def join_odds(matches: pd.DataFrame, odds: pd.DataFrame, season: str) -> pd.DataFrame:
    """Join FPL match rows with betting odds."""
    if odds is None or len(odds) == 0:
        for col in ['B365H', 'B365D', 'B365A', 'implied_prob_h', 'implied_prob_d', 'implied_prob_a']:
            matches[col] = np.nan
        return matches

    # Map FPL team names to odds team names
    matches['odds_home'] = matches['team_h_name'].map(FPL_TO_ODDS)
    matches['odds_away'] = matches['team_a_name'].map(FPL_TO_ODDS)

    # Parse kickoff_time for date matching
    matches['match_date'] = pd.to_datetime(matches['kickoff_time'], errors='coerce').dt.normalize()

    # Try exact match first: (HomeTeam, AwayTeam) within season
    # Since there's exactly one home fixture between any pair per season, match on teams alone
    merged = matches.merge(
        odds[['HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A', 'FTHG', 'FTAG']],
        left_on=['odds_home', 'odds_away'],
        right_on=['HomeTeam', 'AwayTeam'],
        how='left'
    )

    merged = merged.drop(columns=['HomeTeam', 'AwayTeam', 'odds_home', 'odds_away', 'match_date'],
                         errors='ignore')

    # Compute implied probabilities
    has_odds = merged['B365H'].notna() & merged['B365D'].notna() & merged['B365A'].notna()
    merged['implied_prob_h'] = np.nan
    merged['implied_prob_d'] = np.nan
    merged['implied_prob_a'] = np.nan

    if has_odds.any():
        probs = merged.loc[has_odds].apply(
            lambda r: odds_to_implied_probs(r['B365H'], r['B365D'], r['B365A']),
            axis=1, result_type='expand'
        )
        probs.columns = ['implied_prob_h', 'implied_prob_d', 'implied_prob_a']
        merged.loc[has_odds, ['implied_prob_h', 'implied_prob_d', 'implied_prob_a']] = probs.values

    return merged


def process_season(season: str, team_id_map: dict) -> pd.DataFrame:
    """Process one season: load, assign teams, aggregate, build match rows."""
    print(f"\n--- Processing {season} ---")

    # Load data
    gw_df = load_merged_gw(season)
    print(f"  Loaded {len(gw_df)} player-gameweek rows")

    # Drop rows with missing key columns
    gw_df = gw_df.dropna(subset=['gw', 'fixture', 'selected'])
    gw_df['gw'] = gw_df['gw'].astype(int)

    # Assign team_id and element_type
    gw_df = assign_team_and_position(gw_df, season)

    # Check assignment quality
    invalid = (gw_df['team_id'] < 1).sum()
    if invalid > 0:
        print(f"  WARNING: {invalid} rows with invalid team_id (will be excluded)")
        gw_df = gw_df[gw_df['team_id'] >= 1]

    # Build fixture info
    fix_info = build_fixture_info(gw_df, season)
    print(f"  Found {len(fix_info)} fixtures")

    # Aggregate to team level
    team_agg = aggregate_fpl_per_team(gw_df)

    # Build match rows
    team_names = team_id_map.get(season, {})
    matches = build_match_rows(fix_info, team_agg, team_names)

    # Join odds
    odds = load_odds(season)
    matches = join_odds(matches, odds, season)

    # Report
    odds_matched = matches['B365H'].notna().sum()
    print(f"  Built {len(matches)} match rows, {odds_matched} with odds")

    return matches


def validate_matches(df: pd.DataFrame):
    """Print validation summary."""
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    print(f"\nTotal matches: {len(df)}")
    print(f"Matches with odds: {df['B365H'].notna().sum()}")
    print(f"Matches with valid result: {df['result'].isin(['H', 'D', 'A']).sum()}")

    print(f"\nPer season:")
    print(f"{'Season':<12} {'Matches':<10} {'With Odds':<12} {'H/D/A Split'}")
    print("-" * 55)
    for season in SEASONS:
        s = df[df['season'] == season]
        with_odds = s['B365H'].notna().sum()
        h = (s['result'] == 'H').sum()
        d = (s['result'] == 'D').sum()
        a = (s['result'] == 'A').sum()
        print(f"{season:<12} {len(s):<10} {with_odds:<12} {h}/{d}/{a}")

    # Spot check: verify a few scores against odds data
    sample = df[df['B365H'].notna()].sample(min(5, len(df)), random_state=42)
    print(f"\nSpot check (5 random matches):")
    for _, row in sample.iterrows():
        print(f"  {row['season']} GW{row['gw']}: {row['team_h_name']} {row['team_h_score']:.0f}-{row['team_a_score']:.0f} {row['team_a_name']} "
              f"(result={row['result']}, odds H={row['B365H']:.2f} D={row['B365D']:.2f} A={row['B365A']:.2f})")


def main():
    print("Building match dataset...")
    team_id_maps = load_team_mappings()

    all_matches = []
    for season in SEASONS:
        matches = process_season(season, team_id_maps)
        all_matches.append(matches)

    df = pd.concat(all_matches, ignore_index=True)

    # Clean up
    df = df.dropna(subset=['team_h_name', 'team_a_name'])

    # Select and order columns
    cols = [
        'season', 'gw', 'fixture_id', 'kickoff_time',
        'team_h_id', 'team_h_name', 'team_a_id', 'team_a_name',
        'team_h_score', 'team_a_score', 'result',
        'total_selected_h', 'net_transfers_h', 'price_weighted_selected_h', 'def_gk_selected_h', 'num_players_h',
        'total_selected_a', 'net_transfers_a', 'price_weighted_selected_a', 'def_gk_selected_a', 'num_players_a',
        'B365H', 'B365D', 'B365A', 'implied_prob_h', 'implied_prob_d', 'implied_prob_a',
    ]
    df = df[[c for c in cols if c in df.columns]]

    save_parquet(df, 'matches.parquet')
    validate_matches(df)


if __name__ == '__main__':
    main()
