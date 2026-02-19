"""Download all raw data: FPL from vaastav GitHub, odds from football-data.co.uk."""

import sys
import time
import json
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    SEASONS, SEASON_CODES, DATA_RAW_FPL, DATA_RAW_ODDS,
    DATA_REFERENCE, FPL_TO_ODDS,
)

BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master"
ODDS_BASE = "https://www.football-data.co.uk/mmz4281"


def download_file(url: str, dest: Path, retries: int = 3, timeout: int = 60) -> bool:
    """Download a file with retries and exponential backoff. Returns True if successful."""
    if dest.exists() and dest.stat().st_size > 0:
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                dest.write_bytes(resp.content)
                return True
            elif resp.status_code == 404:
                return False
            else:
                print(f"  HTTP {resp.status_code} for {url}")
        except requests.RequestException as e:
            print(f"  Attempt {attempt + 1}/{retries} failed: {e}")

        if attempt < retries - 1:
            wait = 2 ** (attempt + 1)
            time.sleep(wait)

    return False


def fetch_fpl_data():
    """Download FPL CSVs from vaastav GitHub (individual files, not full clone)."""
    print("=" * 60)
    print("Downloading FPL data from vaastav/Fantasy-Premier-League...")
    print("=" * 60)

    DATA_RAW_FPL.mkdir(parents=True, exist_ok=True)

    # Download master_team_list.csv (once)
    master_url = f"{BASE_URL}/data/master_team_list.csv"
    master_dest = DATA_RAW_FPL / "master_team_list.csv"
    ok = download_file(master_url, master_dest)
    print(f"  master_team_list.csv: {'OK' if ok else 'FAILED'}")

    results = {}
    for season in tqdm(SEASONS, desc="Seasons"):
        results[season] = {}

        # merged_gw.csv
        url = f"{BASE_URL}/data/{season}/gws/merged_gw.csv"
        dest = DATA_RAW_FPL / f"{season}_merged_gw.csv"
        ok = download_file(url, dest)
        results[season]['merged_gw'] = ok

        # fixtures.csv
        url = f"{BASE_URL}/data/{season}/fixtures.csv"
        dest = DATA_RAW_FPL / f"{season}_fixtures.csv"
        ok = download_file(url, dest)
        results[season]['fixtures'] = ok

        # players_raw.csv
        url = f"{BASE_URL}/data/{season}/players_raw.csv"
        dest = DATA_RAW_FPL / f"{season}_players_raw.csv"
        ok = download_file(url, dest)
        results[season]['players_raw'] = ok

        # teams.csv
        url = f"{BASE_URL}/data/{season}/teams.csv"
        dest = DATA_RAW_FPL / f"{season}_teams.csv"
        ok = download_file(url, dest)
        results[season]['teams'] = ok

    print("\nFPL Download Summary:")
    print(f"{'Season':<12} {'merged_gw':<12} {'fixtures':<12} {'players_raw':<14} {'teams':<10}")
    print("-" * 60)
    for season, res in results.items():
        row = [season]
        for key in ['merged_gw', 'fixtures', 'players_raw', 'teams']:
            row.append('OK' if res.get(key) else 'MISSING')
        print(f"{row[0]:<12} {row[1]:<12} {row[2]:<12} {row[3]:<14} {row[4]:<10}")

    return results


def fetch_odds_data():
    """Download EPL odds CSVs from football-data.co.uk."""
    print("\n" + "=" * 60)
    print("Downloading odds data from football-data.co.uk...")
    print("=" * 60)

    DATA_RAW_ODDS.mkdir(parents=True, exist_ok=True)

    results = {}
    for season, code in tqdm(SEASON_CODES.items(), desc="Odds"):
        url = f"{ODDS_BASE}/{code}/E0.csv"
        dest = DATA_RAW_ODDS / f"{season}.csv"
        ok = download_file(url, dest, timeout=30)
        results[season] = ok
        if not ok:
            print(f"\n  WARNING: Could not download odds for {season}.")
            print(f"  Please manually download from: {url}")
            print(f"  And save to: {dest}")

    print("\nOdds Download Summary:")
    for season, ok in results.items():
        print(f"  {season}: {'OK' if ok else 'MISSING'}")

    return results


def build_team_mappings():
    """Build and save team_mappings.json."""
    print("\n" + "=" * 60)
    print("Building team mappings...")
    print("=" * 60)

    import pandas as pd

    mappings = {
        'fpl_to_odds': FPL_TO_ODDS,
        'season_team_ids': {},
    }

    # From master_team_list.csv
    master_path = DATA_RAW_FPL / 'master_team_list.csv'
    if master_path.exists():
        master = pd.read_csv(master_path)
        cols_lower = {c: c.lower().strip() for c in master.columns}
        master = master.rename(columns=cols_lower)
        cols = master.columns.tolist()

        # Find the right columns
        season_col = next((c for c in cols if 'season' in c), None)
        id_col = next((c for c in cols if c in ('team', 'id', 'team_id')), None)
        name_col = next((c for c in cols if 'name' in c), None)

        if season_col and id_col and name_col:
            for season in SEASONS:
                filtered = master[master[season_col].astype(str) == season]
                if len(filtered) > 0:
                    mapping = {str(int(row[id_col])): str(row[name_col])
                               for _, row in filtered.iterrows()}
                    mappings['season_team_ids'][season] = mapping

    # For seasons not in master (e.g. 2024-25), try teams.csv
    for season in SEASONS:
        if season not in mappings['season_team_ids']:
            teams_path = DATA_RAW_FPL / f'{season}_teams.csv'
            if teams_path.exists():
                teams = pd.read_csv(teams_path)
                cols_lower = {c: c.lower().strip() for c in teams.columns}
                teams = teams.rename(columns=cols_lower)
                id_col = 'id' if 'id' in teams.columns else teams.columns[0]
                name_col = 'name' if 'name' in teams.columns else teams.columns[1]
                mapping = {str(int(row[id_col])): str(row[name_col])
                           for _, row in teams.iterrows()}
                mappings['season_team_ids'][season] = mapping

    DATA_REFERENCE.mkdir(parents=True, exist_ok=True)
    out_path = DATA_REFERENCE / 'team_mappings.json'
    with open(out_path, 'w') as f:
        json.dump(mappings, f, indent=2)
    print(f"Saved team mappings to {out_path}")

    # Print summary
    for season, teams in sorted(mappings['season_team_ids'].items()):
        print(f"  {season}: {len(teams)} teams")


def validate_data():
    """Print row counts for all downloaded files."""
    import pandas as pd

    print("\n" + "=" * 60)
    print("Validation: Row counts")
    print("=" * 60)
    print(f"{'Season':<12} {'merged_gw':<14} {'fixtures':<12} {'players_raw':<14} {'odds':<10}")
    print("-" * 62)

    for season in SEASONS:
        row = [season]

        for fname in [f'{season}_merged_gw.csv', f'{season}_fixtures.csv', f'{season}_players_raw.csv']:
            path = DATA_RAW_FPL / fname
            if path.exists():
                try:
                    df = pd.read_csv(path, low_memory=False)
                    row.append(str(len(df)))
                except Exception:
                    row.append('ERR')
            else:
                row.append('N/A')

        odds_path = DATA_RAW_ODDS / f'{season}.csv'
        if odds_path.exists():
            try:
                df = pd.read_csv(odds_path, encoding='utf-8', on_bad_lines='skip')
                row.append(str(len(df)))
            except Exception:
                try:
                    df = pd.read_csv(odds_path, encoding='latin-1', on_bad_lines='skip')
                    row.append(str(len(df)))
                except Exception:
                    row.append('ERR')
        else:
            row.append('N/A')

        print(f"{row[0]:<12} {row[1]:<14} {row[2]:<12} {row[3]:<14} {row[4]:<10}")


def main():
    fetch_fpl_data()
    fetch_odds_data()
    build_team_mappings()
    validate_data()
    print("\nData fetch complete.")


if __name__ == '__main__':
    main()
