# FPL Intelligence → EPL Betting Signal: Research Plan

## Objective

Test whether Fantasy Premier League aggregate behavior (ownership, captaincy, transfers) contains predictive signal for EPL match outcomes that is **not already priced into betting odds**. If signal exists, quantify edge and build a live pipeline.

---

## Project Structure

```
fpl-betting-signal/
├── data/
│   ├── raw/
│   │   ├── fpl/                  # vaastav GitHub data (2016-17 → 2024-25)
│   │   ├── odds/                 # football-data.co.uk CSVs
│   │   └── elite/                # top 10k manager data (if available)
│   ├── processed/
│   │   ├── matches.parquet       # unified match-level dataset
│   │   ├── signals.parquet       # engineered features per match
│   │   └── backtest.parquet      # results with P&L
│   └── reference/
│       └── team_mappings.json    # FPL team names ↔ odds team names
├── src/
│   ├── 01_fetch_data.py
│   ├── 02_build_match_dataset.py
│   ├── 03_engineer_signals.py
│   ├── 04_model_train.py
│   ├── 05_backtest.py
│   └── utils.py
├── notebooks/
│   └── eda.ipynb                 # exploratory analysis
├── outputs/
│   ├── results.md                # final findings
│   ├── charts/                   # all visualizations
│   └── model/                    # saved model artifacts
├── requirements.txt
└── README.md
```

---

## Data Sources

### 1. FPL Historical Data
- **Source:** `https://github.com/vaastav/Fantasy-Premier-League`
- **Clone:** `git clone https://github.com/vaastav/Fantasy-Premier-League.git data/raw/fpl`
- **Key files per season:**
  - `data/{season}/gws/merged_gw.csv` — player-level stats per gameweek (ownership %, transfers in/out, selected_by_percent, minutes, goals, assists, clean_sheets, etc.)
  - `data/{season}/cleaned_players.csv` — player metadata (team, position, price)
  - `data/{season}/fixtures.csv` — fixture list with teams + GW mapping
- **Seasons to use:** `2016-17`, `2017-18`, `2018-19`, `2019-20`, `2020-21`, `2021-22`, `2022-23`, `2023-24`, `2024-25`
- **Critical columns from merged_gw.csv:** `name`, `team`, `GW` (or `round`), `selected`, `transfers_in`, `transfers_out`, `was_home`, `opponent_team`, `total_points`, `goals_scored`, `assists`, `clean_sheets`, `minutes`
- Note: Column names may vary by season. Inspect each season's CSV headers first.

### 2. Historical Betting Odds
- **Source:** `https://www.football-data.co.uk/englandm.php`
- **Download:** EPL CSV for each season (e.g., `https://www.football-data.co.uk/mmz4281/2324/E0.csv`)
- **URL pattern:** `https://www.football-data.co.uk/mmz4281/{YYNN}/E0.csv` where YYNN = e.g. `2324` for 2023-24
- **Key columns:** `Date`, `HomeTeam`, `AwayTeam`, `FTHG` (home goals), `FTAG` (away goals), `FTR` (result: H/D/A), `B365H`, `B365D`, `B365A` (Bet365 odds), `Avg>2.5`, `Avg<2.5` (over/under odds where available)
- Save to `data/raw/odds/`

### 3. Team Name Mapping
FPL and football-data.co.uk use different team names. Build a mapping JSON:
```json
{
  "Man City": ["Manchester City", "Man City"],
  "Man Utd": ["Manchester United", "Man United", "Man Utd"],
  "Spurs": ["Tottenham", "Tottenham Hotspur", "Spurs"],
  "Wolves": ["Wolverhampton", "Wolverhampton Wanderers", "Wolves"],
  "Nott'm Forest": ["Nottingham Forest", "Nott'm Forest"],
  "Sheffield Utd": ["Sheffield United", "Sheffield Utd"],
  "West Ham": ["West Ham United", "West Ham"],
  "Brighton": ["Brighton and Hove Albion", "Brighton"],
  "Newcastle": ["Newcastle United", "Newcastle Utd", "Newcastle"],
  "Leicester": ["Leicester City", "Leicester"],
  "Leeds": ["Leeds United", "Leeds"],
  "West Brom": ["West Bromwich Albion", "West Brom"]
}
```
Extend this as needed. Use fuzzy matching as fallback. The FPL API uses integer team IDs — map these to names via the `teams` endpoint or `cleaned_players.csv`.

---

## Signal Definitions

For each EPL match (Team_H vs Team_A) in gameweek GW, compute the following signals using **pre-match** FPL data (i.e., data available before the GW deadline):

### Signal 1: Transfer Momentum Score (TMS)
```python
# Net transfers for all players belonging to each team in the GW
tms_h = sum(transfers_in - transfers_out) for all Team_H players in GW
tms_a = sum(transfers_in - transfers_out) for all Team_A players in GW
transfer_delta = tms_h - tms_a
transfer_ratio = tms_h / (tms_h + tms_a + 1)  # +1 to avoid div/0
```

### Signal 2: Ownership Conviction Score (OCS)
```python
# Price-weighted ownership captures conviction intensity
# A 60% owned £13M Haaland signals more than a 60% owned £4.5M bench fodder
ocs_h = sum(selected_by_percent * now_cost) for Team_H players
ocs_a = sum(selected_by_percent * now_cost) for Team_A players
ownership_ratio = ocs_h / (ocs_h + ocs_a)
```

### Signal 3: Captaincy Concentration (CC)
```python
# Requires captaincy data — available via top manager picks or API
# If captaincy % not in vaastav data, approximate using:
# - Player with highest selected% on each team as proxy
# - Or use transfers_in as conviction proxy
# If available:
cc_h = sum(captain_pct) for Team_H players
cc_a = sum(captain_pct) for Team_A players
captain_ratio = cc_h / (cc_h + cc_a + 0.001)
```
**Note:** Captaincy data may not be in vaastav's historical dataset. If unavailable, skip this signal and focus on transfers + ownership. The FPL API `/api/event/{gw}/live/` endpoint has some aggregate data but not historical captaincy splits. Use transfer velocity as the proxy — it's the next-best conviction indicator.

### Signal 4: Defensive Confidence Score (DCS)
```python
# Ownership of DEF (element_type=2) and GK (element_type=1) players
# High DEF/GK ownership → crowd expects clean sheet → maps to Under market
dcs_h = sum(selected_by_percent) for Team_H DEF + GK
dcs_a = sum(selected_by_percent) for Team_A DEF + GK
# Normalize by number of DEF/GK available per team
```

### Signal 5: Ownership Velocity (OV)
```python
# Week-over-week change in total team ownership
# Captures momentum / trend strength
ov_h = sum(selected_by_percent[GW]) - sum(selected_by_percent[GW-1]) for Team_H
ov_a = sum(selected_by_percent[GW]) - sum(selected_by_percent[GW-1]) for Team_A
velocity_delta = ov_h - ov_a
```

### Odds Conversion
```python
# Convert decimal odds to implied probability (removing overround)
raw_h = 1 / B365H
raw_d = 1 / B365D
raw_a = 1 / B365A
overround = raw_h + raw_d + raw_a
implied_prob_h = raw_h / overround
implied_prob_d = raw_d / overround
implied_prob_a = raw_a / overround
```

---

## Model Specification

### Primary Model: Logistic Regression (Multinomial)
```python
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

# Features (per match)
X = [
    transfer_ratio,
    ownership_ratio,
    velocity_delta,
    dcs_h, dcs_a,
    implied_prob_h, implied_prob_d, implied_prob_a,  # include odds as features
    is_home,  # home advantage control
]

# Target
y = match_result  # 0=Away, 1=Draw, 2=Home

# Train with calibration
base_model = LogisticRegression(multi_class='multinomial', max_iter=1000)
model = CalibratedClassifierCV(base_model, cv=5)
model.fit(X_train, y_train)
```

### Benchmark Models
1. **Odds-only:** `X = [implied_prob_h, implied_prob_d, implied_prob_a]` → This is the market to beat
2. **FPL-only:** `X = [transfer_ratio, ownership_ratio, velocity_delta, dcs_h, dcs_a]` → Pure FPL signal
3. **FPL + Odds:** All features → The hypothesis model

### Evaluation Metrics
```python
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

# For each model:
# 1. Brier Score (lower = better calibrated probabilities)
# 2. Log Loss (lower = better probability estimates)
# 3. Accuracy (% correct predictions)
# 4. Brier Skill Score vs odds-only baseline:
#    BSS = 1 - (brier_fpl_odds / brier_odds_only)
#    BSS > 0 means FPL adds value over odds alone
```

### ROI Backtest
```python
# Flat-stake betting simulation
# For each match where FPL+Odds model diverges from odds-only by > threshold:
# - If P_model(H) - P_odds(H) > threshold: bet on Home
# - If P_model(A) - P_odds(A) > threshold: bet on Away
# Track: total bets, wins, losses, ROI, max drawdown

# Test thresholds: [0.03, 0.05, 0.07, 0.10]
# Report ROI and number of qualifying bets for each threshold
```

---

## Visualization Requirements

Generate all charts with dark background, consistent with 1kx style:
- **Background:** `#0a0a0a` or `#111111`
- **Colors:** Navy `#1e3a5f`, Teal `#2dd4bf`, Purple `#8b5cf6`, Coral `#f97316`, White `#e5e5e5`
- **Font:** Sans-serif (use matplotlib's `'sans-serif'`)
- **Minimal gridlines**, left-aligned titles, generous whitespace

### Required Charts
1. **Signal Distribution:** Histogram of each FPL signal, faceted by match outcome (H/D/A)
2. **Divergence vs ROI:** Scatter plot showing divergence threshold (x) vs cumulative ROI (y) 
3. **Calibration Plot:** Reliability diagram — predicted probability (x) vs observed frequency (y) for each model
4. **Brier Score Comparison:** Bar chart comparing Brier scores across all 3 models
5. **Cumulative P&L:** Line chart showing cumulative returns over time for divergence-based betting strategy
6. **Signal Correlation Matrix:** Heatmap of FPL signals + odds correlations
7. **Season Breakdown:** ROI by season (bar chart) to check consistency
8. **Quintile Analysis:** For each signal, bucket matches into quintiles, show actual win rate vs implied odds win rate per quintile

---

## 1-Week Execution Plan

### Day 1 (Mon): Data Assembly
- [ ] Clone vaastav repo, download football-data.co.uk CSVs for all available seasons
- [ ] Inspect column headers across seasons — document any naming inconsistencies
- [ ] Build team name mapping JSON (FPL names ↔ odds names)
- [ ] Create `01_fetch_data.py` that automates all downloads
- [ ] Create `02_build_match_dataset.py`:
  - Join FPL GW data → fixtures → match results + odds
  - Output: `matches.parquet` with one row per match, containing both FPL aggregates and odds
- [ ] Validate: spot-check 10 random matches to confirm joins are correct

### Day 2 (Tue): Signal Engineering + EDA
- [ ] Create `03_engineer_signals.py`:
  - Compute all 5 signals per match
  - Handle edge cases (GW1 has no prior week for velocity, newly promoted teams, etc.)
  - Output: `signals.parquet`
- [ ] Run EDA notebook:
  - Distribution of each signal
  - Correlation matrix (signals vs each other, signals vs odds)
  - Quick check: are signals just proxies for odds? If correlation with implied_prob > 0.85, signal may not add value
  - Identify which signals have independent variance from odds

### Day 3 (Wed): Model Training
- [ ] Create `04_model_train.py`:
  - Train/val/test split: 2016-17 → 2021-22 train, 2022-23 val, 2023-24 + 2024-25 test
  - Train all 3 models (odds-only, FPL-only, FPL+odds)
  - Calibrate probabilities
  - Compute Brier Score, Log Loss, Accuracy on validation set
  - Compute Brier Skill Score (FPL+odds vs odds-only)
  - Generate calibration plots and Brier comparison chart
- [ ] Hyperparameter sensitivity: test with/without regularization, with/without individual signals
- [ ] Feature importance: which FPL signals contribute most?

### Day 4 (Thu): Backtesting
- [ ] Create `05_backtest.py`:
  - Walk-forward backtest on test set (2023-24, 2024-25)
  - For each divergence threshold [0.03, 0.05, 0.07, 0.10]:
    - Count qualifying bets
    - Compute flat-stake ROI
    - Compute max drawdown
    - Compute Sharpe ratio of daily returns
  - Generate cumulative P&L chart, season breakdown, quintile analysis
- [ ] Statistical significance: bootstrap 95% CI on ROI, binomial test on hit rate

### Day 5 (Fri): Season-Level Signals + Report
- [ ] Test season-level hypothesis:
  - At GW10, GW19, GW28: take team with highest aggregate FPL ownership
  - Compare to bookmaker ante-post odds for title/top-4
  - Small sample (8 seasons) so treat as supplementary, not primary
- [ ] Compile `outputs/results.md`:
  - Executive summary: does FPL add predictive value over odds?
  - Key numbers: Brier Skill Score, best ROI threshold, hit rate
  - All charts
  - Signal-by-signal breakdown
  - Recommendations: which signals to track live, which markets to target
- [ ] If results are positive: draft architecture for live pipeline (FPL API → signal computation → Polymarket/Betfair comparison → alert system)

---

## Key Decision Points

After Day 3 (model training), evaluate:
- **If Brier Skill Score > 0.02:** Proceed to full backtest — meaningful signal exists
- **If Brier Skill Score 0.00–0.02:** Signal exists but is weak — focus on specific subsets (e.g., only matches where elite managers diverge from crowd)
- **If Brier Skill Score < 0.00:** FPL signals don't add value over odds. Pivot to testing individual signals in isolation or testing against specific market types only (over/under, Asian handicap) rather than 1X2

---

## Dependencies

```
# requirements.txt
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
pyarrow>=12.0      # for parquet
requests>=2.31
tqdm>=4.65
scipy>=1.11
```

---

## Notes for Execution

1. **Start with the simplest version.** Get `matches.parquet` built correctly before worrying about advanced signals. A correct, simple analysis beats a broken complex one.

2. **The team name mapping is the trickiest part.** FPL uses integer team IDs that change each season. football-data.co.uk uses full team names that also vary. Build the mapping carefully and validate with spot checks.

3. **Missing data is expected.** Some seasons may not have all columns (e.g., `selected_by_percent` might be named differently, or captaincy data might be absent). Handle gracefully with fallbacks.

4. **The null hypothesis is that odds are efficient.** We need to prove FPL adds *incremental* information. Always benchmark against odds-only. If FPL+Odds doesn't beat Odds-Only, the hypothesis fails — and that's a valid finding.

5. **Chart style reminder:** Dark bg (`#0a0a0a`), colors: navy/teal/purple/coral/white, minimal grids, clean sans-serif, left-aligned titles, no frames.
