"""Shared constants, team mappings, and helper functions for FPL betting signal research."""

from pathlib import Path
import json
import difflib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Seasons ---
SEASONS = [
    '2016-17', '2017-18', '2018-19', '2019-20', '2020-21',
    '2021-22', '2022-23', '2023-24', '2024-25',
]

SEASON_CODES = {
    '2016-17': '1617', '2017-18': '1718', '2018-19': '1819',
    '2019-20': '1920', '2020-21': '2021', '2021-22': '2122',
    '2022-23': '2223', '2023-24': '2324', '2024-25': '2425',
}

TRAIN_SEASONS = ['2016-17', '2017-18', '2018-19', '2019-20', '2020-21', '2021-22']
VAL_SEASONS = ['2022-23']
TEST_SEASONS = ['2023-24', '2024-25']

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_FPL = PROJECT_ROOT / 'data' / 'raw' / 'fpl'
DATA_RAW_ODDS = PROJECT_ROOT / 'data' / 'raw' / 'odds'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
DATA_REFERENCE = PROJECT_ROOT / 'data' / 'reference'
OUTPUTS = PROJECT_ROOT / 'outputs'
CHARTS_DIR = OUTPUTS / 'charts'
MODEL_DIR = OUTPUTS / 'model'

# --- Element types ---
ELEMENT_TYPE_MAP = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
POSITION_STR_MAP = {'GK': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}

# --- FPL team name to football-data.co.uk team name ---
FPL_TO_ODDS = {
    'Arsenal': 'Arsenal',
    'Aston Villa': 'Aston Villa',
    'Bournemouth': 'Bournemouth',
    'Brentford': 'Brentford',
    'Brighton': 'Brighton',
    'Burnley': 'Burnley',
    'Cardiff': 'Cardiff',
    'Chelsea': 'Chelsea',
    'Crystal Palace': 'Crystal Palace',
    'Everton': 'Everton',
    'Fulham': 'Fulham',
    'Huddersfield': 'Huddersfield',
    'Hull': 'Hull',
    'Ipswich': 'Ipswich',
    'Leeds': 'Leeds',
    'Leicester': 'Leicester',
    'Liverpool': 'Liverpool',
    'Luton': 'Luton',
    'Man City': 'Man City',
    'Man Utd': 'Man United',
    'Middlesbrough': 'Middlesbrough',
    'Newcastle': 'Newcastle',
    'Norwich': 'Norwich',
    "Nott'm Forest": "Nott'm Forest",
    'Sheffield Utd': 'Sheffield United',
    'Southampton': 'Southampton',
    'Spurs': 'Tottenham',
    'Stoke': 'Stoke',
    'Sunderland': 'Sunderland',
    'Swansea': 'Swansea',
    'Watford': 'Watford',
    'West Brom': 'West Brom',
    'West Ham': 'West Ham',
    'Wolves': 'Wolves',
}

# Reverse mapping: odds name -> FPL name
ODDS_TO_FPL = {v: k for k, v in FPL_TO_ODDS.items()}

# --- Chart styling ---
CHART_COLORS = {
    'navy': '#1e3a5f',
    'teal': '#2dd4bf',
    'purple': '#8b5cf6',
    'coral': '#f97316',
    'white': '#e5e5e5',
    'bg': '#0a0a0a',
    'grid': '#1a1a1a',
    'edge': '#333333',
}


def setup_chart_style():
    """Configure matplotlib for dark-theme charts."""
    plt.rcParams.update({
        'figure.facecolor': CHART_COLORS['bg'],
        'axes.facecolor': CHART_COLORS['bg'],
        'axes.edgecolor': CHART_COLORS['edge'],
        'axes.labelcolor': CHART_COLORS['white'],
        'text.color': CHART_COLORS['white'],
        'xtick.color': CHART_COLORS['white'],
        'ytick.color': CHART_COLORS['white'],
        'grid.color': CHART_COLORS['grid'],
        'grid.alpha': 0.3,
        'font.family': 'sans-serif',
        'font.size': 11,
        'figure.figsize': (12, 7),
        'savefig.dpi': 300,
        'savefig.facecolor': CHART_COLORS['bg'],
        'savefig.bbox': 'tight',
        'legend.facecolor': '#111111',
        'legend.edgecolor': CHART_COLORS['edge'],
    })


def get_fpl_team_mapping(season: str) -> dict:
    """Return {team_id (int): team_name (str)} for a given season.

    Uses master_team_list.csv for 2016-17 through 2023-24.
    Uses {season}/teams.csv for 2024-25.
    Falls back to players_raw.csv + teams.csv when available.
    """
    # Try master_team_list first
    master_path = DATA_RAW_FPL / 'master_team_list.csv'
    if master_path.exists():
        master = pd.read_csv(master_path)
        # Columns: season_name (or season), team_id (or id), team_name (or name)
        # Normalize column names
        cols = master.columns.str.lower().str.strip()
        master.columns = cols
        season_col = [c for c in cols if 'season' in c]
        id_col = [c for c in cols if 'id' in c or c == 'team']
        name_col = [c for c in cols if 'name' in c]

        if season_col and id_col and name_col:
            filtered = master[master[season_col[0]].astype(str) == season]
            if len(filtered) > 0:
                return dict(zip(filtered[id_col[0]].astype(int), filtered[name_col[0]].astype(str)))

    # Fallback: per-season teams.csv
    teams_path = DATA_RAW_FPL / f'{season}_teams.csv'
    if teams_path.exists():
        teams = pd.read_csv(teams_path)
        cols = teams.columns.str.lower().str.strip()
        teams.columns = cols
        id_col = 'id' if 'id' in cols else cols[0]
        name_col = 'name' if 'name' in cols else cols[1]
        return dict(zip(teams[id_col].astype(int), teams[name_col].astype(str)))

    return {}


def get_player_metadata(season: str) -> pd.DataFrame:
    """Load players_raw.csv for a season and return (element_id, team_id, element_type, now_cost).

    Returns DataFrame with columns: element, team_id, element_type, now_cost
    """
    path = DATA_RAW_FPL / f'{season}_players_raw.csv'
    if not path.exists():
        return pd.DataFrame(columns=['element', 'team_id', 'element_type', 'now_cost'])

    df = pd.read_csv(path)
    cols = df.columns.str.strip().str.strip('"')
    df.columns = cols

    result = pd.DataFrame()
    result['element'] = df['id'].astype(int) if 'id' in df.columns else df.iloc[:, 0].astype(int)

    if 'team' in df.columns:
        result['team_id'] = df['team'].astype(int)
    elif 'team_code' in df.columns:
        result['team_id'] = df['team_code'].astype(int)
    else:
        result['team_id'] = -1

    if 'element_type' in df.columns:
        result['element_type'] = df['element_type'].astype(int)
    else:
        result['element_type'] = 0  # unknown

    if 'now_cost' in df.columns:
        result['now_cost'] = df['now_cost'].astype(float)
    else:
        result['now_cost'] = 50.0  # default ~5.0M

    return result


def fuzzy_match_team(name: str, candidates: list, cutoff: float = 0.6) -> str | None:
    """Fuzzy-match a team name against a list of candidates."""
    matches = difflib.get_close_matches(name, candidates, n=1, cutoff=cutoff)
    return matches[0] if matches else None


def odds_to_implied_probs(h_odds: float, d_odds: float, a_odds: float) -> tuple:
    """Convert decimal odds to normalized implied probabilities (removing overround)."""
    raw_h = 1.0 / h_odds
    raw_d = 1.0 / d_odds
    raw_a = 1.0 / a_odds
    overround = raw_h + raw_d + raw_a
    return (raw_h / overround, raw_d / overround, raw_a / overround)


def load_parquet(name: str) -> pd.DataFrame:
    """Load a parquet file from the processed data directory."""
    return pd.read_parquet(DATA_PROCESSED / name)


def save_parquet(df: pd.DataFrame, name: str):
    """Save a DataFrame to the processed data directory as parquet."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_PROCESSED / name, index=False)
    print(f"Saved {name}: {len(df)} rows, {len(df.columns)} columns")
