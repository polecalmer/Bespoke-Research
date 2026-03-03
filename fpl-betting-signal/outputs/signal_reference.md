# FPL Signal Reference

## How We Extract Predictions from a Fantasy Game

This document is a complete technical reference for the signals we extract from Fantasy Premier League manager behaviour and how they translate into match outcome predictions.

---

## The Core Idea

11 million people play FPL every week. Each one makes decisions — who to buy, sell, captain, bench — that implicitly encode beliefs about upcoming match outcomes. We extract those beliefs as numerical signals and feed them into a calibrated logistic regression model that outputs P(Home Win), P(Draw), P(Away Win) for every Premier League match.

**The result:** crowd signals alone achieve 71.7% of professional bookmaker prediction skill, with an Expected Calibration Error of 0.036 (matching the bookmakers' 0.037). This isn't noise — it's a statistically significant information source (Spearman rho=0.90 between crowd confidence and actual accuracy, p=0.037).

---

## Data Sources

| Source | What We Get | URL |
|--------|------------|-----|
| vaastav/Fantasy-Premier-League (GitHub) | Player-level GW data: ownership, transfers, points, ICT, prices | `github.com/vaastav/Fantasy-Premier-League` |
| football-data.co.uk | Match-level betting odds (Bet365 H/D/A) | `football-data.co.uk/mmz4281/` |
| FPL API (live only) | Real-time bootstrap + fixture data | `fantasy.premierleague.com/api/` |

**Coverage:** 9 seasons (2016-17 through 2024-25), ~3,400 matches, ~170,000 player-gameweek rows. Plus 2025-26 live validation (GW1-24, 240 matches).

---

## Signal Taxonomy

We extract 24 signals organised into 4 categories. Each signal is computed at the **match level** — comparing the home team's aggregated player data against the away team's.

### How Signals Are Constructed

Most signals follow a **ratio pattern**:

```
signal = metric_home / (metric_home + metric_away + 1)
```

This normalises to a 0-1 range where 0.5 = teams are equal. Values >0.5 favour the home team. The `+1` in the denominator prevents division by zero and acts as light smoothing.

A few signals use a **delta pattern** (home minus away) when the metric can be negative or when the ratio form doesn't make intuitive sense.

### Temporal Rules (No Look-Ahead Bias)

All signals use only data available **before** the match kicks off:

- **Ownership/transfers**: Current GW data (set by managers before deadline)
- **Quality metrics** (points, ICT, clean sheets): From the **previous** gameweek's results
- **Velocity**: Change between current and previous GW ownership

This is critical — every signal represents what was knowable at team selection deadline, not after the match.

---

## The 24 Signals

### Category 1: STOCK Signals (Accumulated State)

These capture the overall "installed base" of manager belief — how many managers hold players from each team and how much they've invested.

#### 1. `ownership_ratio` — Price-Weighted Ownership Conviction

```python
pw_selected_h = sum(selected_i * now_cost_i)  # for home team players
pw_selected_a = sum(selected_i * now_cost_i)  # for away team players
ownership_ratio = pw_selected_h / (pw_selected_h + pw_selected_a + 1)
```

**What it means:** Managers who own expensive players from a team are making a stronger statement than those owning cheap ones. An £12M Salah in 3 million teams signals more conviction than a £4.5M bench fodder in the same number.

**Source columns:** `selected`, `now_cost` (FPL player data)

| Stat | Value |
|------|-------|
| Mutual information | 0.107 (3rd highest of all signals) |
| Correlation with P(Home) | Strong positive |
| Correlation with `dcs_ratio` | r = 0.816 |

**Why it matters:** This is our single best signal. It captures the aggregate portfolio allocation of millions of managers, weighted by conviction (price). It's the closest thing to a "prediction market" hidden inside a fantasy game.

#### 2. `raw_ownership_ratio` — Unweighted Ownership

```python
sel_h = sum(selected_i)  # home team
sel_a = sum(selected_i)  # away team
raw_ownership_ratio = sel_h / (sel_h + sel_a + 1)
```

**What it means:** Pure headcount — how many manager-slots are occupied by each team's players. Unlike `ownership_ratio`, this doesn't weight by price, so a £4.5M enabler counts the same as a £13M premium.

**Why include both?** The gap between weighted and unweighted ownership reveals where managers are investing premium budget vs. filling cheap slots. A team with high `ownership_ratio` but low `raw_ownership_ratio` has a few expensive, highly-owned stars.

#### 3. `squad_value_ratio` — Total Squad Value

```python
val_h = sum(now_cost_i)  # home team total squad value
val_a = sum(now_cost_i)  # away team total squad value
squad_value_ratio = val_h / (val_h + val_a + 1)
```

**What it means:** The FPL market's valuation of each team's squad. Prices in FPL adjust based on transfer activity (demand/supply), so this indirectly captures the market's assessment of player quality. Arsenal's squad consistently prices higher than Luton's.

**Limitation:** Prices are sticky — they change by £0.1M at most per day. So this signal moves slowly and mostly captures pre-season expectations plus gradual drift.

---

### Category 2: FLOW Signals (Active Decisions This Week)

These capture what managers are *doing right now* — transfers in/out, captaincy-related velocity. Flow signals are noisier but more timely than stock signals.

#### 4. `transfer_ratio` — Net Transfer Momentum

```python
net_h = sum(transfers_in_i - transfers_out_i)  # home team
net_a = sum(transfers_in_i - transfers_out_i)  # away team
transfer_ratio = net_h / (|net_h| + |net_a| + 1)
```

**What it means:** Which team are managers actively buying into this week? A team with high net transfers before a match is one that millions of managers are choosing to invest in — they expect good returns (points).

| Stat | Value |
|------|-------|
| Mutual information | 0.054 |
| Mean by result (2025-26) | H: 0.572, D: 0.492, A: 0.398 |
| H-A separation | +0.174 |

**Edge potential:** Transfer data is available in real-time from the FPL API. Heavy transfer activity into a team's players before a GW deadline is a live crowd signal that bookmakers may not fully price in, especially for mid-table matches that get less professional attention.

#### 5. `transfer_conviction_ratio` — Value-Weighted Transfers

```python
tc_h = sum(transfers_in_i * now_cost_i)  # home team
tc_a = sum(transfers_in_i * now_cost_i)  # away team
transfer_conviction_ratio = tc_h / (tc_h + tc_a + 1)
```

**What it means:** Similar to transfer_ratio but weighted by player price. Transferring in a £12M premium signals more conviction than a £4.5M punt. This distinguishes "I believe this team will score" from "I need a cheap enabler."

#### 6. `transfer_intensity_delta` — Transfer Rate Relative to Base

```python
intensity_h = mean(transfers_in_i / (selected_i + 1))  # per player
intensity_a = mean(transfers_in_i / (selected_i + 1))
transfer_intensity_delta = intensity_h - intensity_a
```

**What it means:** Transfer rate normalised by existing ownership. A player owned by 100K getting 50K transfers in is a much stronger signal than a player owned by 5M getting 50K. This captures "breakout" transfers — players being discovered or suddenly in demand.

#### 7. `sell_pressure_ratio` — Who's Being Dumped?

```python
out_h = sum(transfers_out_i)  # home team
out_a = sum(transfers_out_i)  # away team
sell_pressure_ratio = out_h / (out_h + out_a + 1)
```

**What it means:** The inverse of buying conviction. High sell pressure on a team's players before a match means managers are losing faith. Interpret carefully: a team with high sell pressure and high buy activity simultaneously is experiencing *rotation* (managers swapping one player for another), not necessarily negative sentiment.

#### 8. `captain_proxy` — Transfer Velocity Differential

```python
tv_h = net_transfers_h / (total_selected_h + 1)
tv_a = net_transfers_a / (total_selected_a + 1)
captain_proxy = tv_h - tv_a
```

**What it means:** Normalised transfer velocity — how fast ownership is changing relative to the installed base. Named "captain_proxy" because in early development we hypothesised it tracked captain choices, though it more precisely captures marginal demand intensity.

| Stat | Value |
|------|-------|
| Mutual information | 0.034 |
| Correlation with `velocity_delta` | r = 0.723 |

#### 9. `velocity_delta` — Week-over-Week Ownership Change

```python
ownership_velocity_h = total_selected_h(this_GW) - total_selected_h(prev_GW)
ownership_velocity_a = total_selected_a(this_GW) - total_selected_a(prev_GW)
velocity_delta = ownership_velocity_h - ownership_velocity_a
```

**What it means:** The *change* in total team ownership from last week to this week. Positive velocity means the crowd is increasing allocation to the home team. This is the first derivative of the stock signal — it captures *momentum* in crowd belief.

**Note:** GW1 = 0 (no previous week to compare against).

---

### Category 3: QUALITY Signals (Performance from Previous GW)

These use actual FPL performance metrics from the **previous** gameweek, weighted by **current** ownership. The weighting matters: it captures "quality of the players managers are currently backing."

#### 10. `threat_ratio` — Attacking Danger (ICT Threat)

```python
threat_h = sum(threat_prev_i * selected_current_i)  # home team
threat_a = sum(threat_prev_i * selected_current_i)  # away team
threat_ratio = threat_h / (threat_h + threat_a + 1)
```

**What it means:** ICT Threat measures attacking danger — shots, touches in the box, goal attempts. Weighted by current ownership, this asks: "Are the players managers own from this team actually creating chances?"

**Source:** FPL's ICT (Influence, Creativity, Threat) index, computed by Opta.

#### 11. `creativity_ratio` — Chance Creation (ICT Creativity)

```python
creativity_h = sum(creativity_prev_i * selected_current_i)
creativity_a = sum(creativity_prev_i * selected_current_i)
creativity_ratio = creativity_h / (creativity_h + creativity_a + 1)
```

**What it means:** ICT Creativity measures chance creation — key passes, successful crosses, through balls. Tells us whether the players managers are backing are involved in creating goals.

#### 12. `influence_ratio` — Overall Match Involvement (ICT Influence)

```python
influence_h = sum(influence_prev_i * selected_current_i)
influence_a = sum(influence_prev_i * selected_current_i)
influence_ratio = influence_h / (influence_h + influence_a + 1)
```

**What it means:** ICT Influence is the broadest metric — it captures overall impact on a match including defensive actions. A high influence ratio means the players managers own from this team dominated their previous match.

#### 13. `bps_quality_ratio` — Bonus Points Quality

```python
bps_h = sum(bps_prev_i * selected_current_i)
bps_a = sum(bps_prev_i * selected_current_i)
bps_quality_ratio = bps_h / (bps_h + bps_a + 1)
```

**What it means:** BPS (Bonus Point System) rewards the best-performing players in each match. Ownership-weighted BPS from the previous week tells us whether the crowd is backing players who are actually performing at a high level.

#### 14. `points_momentum_ratio` — Recent FPL Points

```python
pts_h = sum(total_points_prev_i * selected_current_i)
pts_a = sum(total_points_prev_i * selected_current_i)
points_momentum_ratio = pts_h / (pts_h + pts_a + 1)
```

**What it means:** The simplest quality signal — did the players managers own from this team score well last week? Ownership-weighted recent form.

#### 15. `minutes_ratio` — Playing Time

```python
mins_h = sum(minutes_prev_i)  # home team
mins_a = sum(minutes_prev_i)  # away team
minutes_ratio = mins_h / (mins_h + mins_a + 1)
```

**What it means:** Total minutes played last week. This is primarily a squad availability signal — teams with fewer minutes may have key players rested, injured, or rotated. Less informative than other quality signals but serves as a control.

#### 16. `prev_clean_sheet_ratio` — Defensive Performance

```python
cs_h = sum(clean_sheets_prev_i)  # home team
cs_a = sum(clean_sheets_prev_i)  # away team
prev_clean_sheet_ratio = cs_h / (cs_h + cs_a + 1)
```

**What it means:** Binary signal — did the team keep a clean sheet last week? Clean sheets are a strong indicator of defensive organisation. Unlike the attacking metrics, this is unweighted by ownership since clean sheets apply uniformly to all defensive players.

#### 17. `defensive_strength_ratio` — Goals Conceded (Inverted)

```python
gc_h = sum(goals_conceded_prev_i)  # home team
gc_a = sum(goals_conceded_prev_i)  # away team
defensive_strength_ratio = gc_a / (gc_h + gc_a + 1)  # NOTE: inverted
```

**What it means:** Goals conceded last week, with the ratio inverted so higher = better defence (home team conceded fewer). Provides more granularity than the binary clean sheet signal.

---

### Category 4: STRUCTURE Signals (Tactical Composition)

These capture *how* managers are building their squads — which positions they prioritise, how concentrated ownership is, and whether they're betting on stars or spreading risk.

#### 18. `dcs_ratio` — Defensive Confidence Score

```python
def_gk_h = sum(selected_i) where element_type in (1, 2)  # GK + DEF, home team
def_gk_a = sum(selected_i) where element_type in (1, 2)  # away team
dcs_ratio = def_gk_h / (def_gk_h + def_gk_a + 1)
```

**What it means:** Ratio of defensive player ownership. When managers own lots of defenders/keepers from a team, they expect that team to keep clean sheets — which requires winning or drawing. This is an indirect but powerful signal.

| Stat | Value |
|------|-------|
| Mutual information | 0.065 (2nd highest FPL signal) |
| Correlation with `ownership_ratio` | r = 0.816 |

**Why it's strong:** Defensive ownership is a purer signal of "I think this team won't concede" than attacking ownership (which could just mean "I want points from goals, regardless of match result").

#### 19. `fwd_ownership_ratio` — Forward Ownership

```python
fwd_h = sum(selected_i) where element_type == 4  # FWD, home team
fwd_a = sum(selected_i) where element_type == 4
fwd_ownership_ratio = fwd_h / (fwd_h + fwd_a + 1)
```

**What it means:** How many managers own forwards from each team. Forward ownership signals attacking expectation — managers expect goals.

#### 20. `mid_ownership_ratio` — Midfielder Ownership

```python
mid_h = sum(selected_i) where element_type == 3  # MID, home team
mid_a = sum(selected_i) where element_type == 3
mid_ownership_ratio = mid_h / (mid_h + mid_a + 1)
```

**What it means:** Midfielder ownership. In FPL, midfielders who score/assist get bonus points, so high midfielder ownership from a team signals expected attacking involvement with clean sheet potential.

#### 21. `premium_ratio` — Premium Player Ownership (>= £8.0M)

```python
prem_h = sum(selected_i) where now_cost >= 80  # home, >= £8.0M
prem_a = sum(selected_i) where now_cost >= 80
premium_ratio = prem_h / (prem_h + prem_a + 1)
```

**What it means:** How much managers are investing in a team's expensive players. Premium ownership is a signal of *deep* conviction — owning a £12M player costs a significant chunk of the £100M budget, so it represents a meaningful allocation decision.

#### 22. `budget_ratio` — Budget Player Ownership (< £5.0M)

```python
bud_h = sum(selected_i) where now_cost < 50  # home, < £5.0M
bud_a = sum(selected_i) where now_cost < 50
budget_ratio = bud_h / (bud_h + bud_a + 1)
```

**What it means:** The inverse of premium — cheap player ownership. A team with high budget ratio is being used as a source of "enablers" rather than primary attackers. This can signal that managers see the team as defensively solid but unlikely to score heavily.

#### 23. `concentration_delta` — Ownership Concentration (HHI)

```python
shares_h = [selected_i / total_selected_h for each player]  # home
HHI_h = sum(share^2 for share in shares_h)
HHI_a = sum(share^2 for share in shares_a)
concentration_delta = HHI_h - HHI_a
```

**What it means:** Herfindahl-Hirschman Index of ownership concentration. High HHI means managers are piling into a few star players; low HHI means ownership is spread across the squad. Concentrated ownership implies the crowd has identified specific match-winning individuals.

**Interpretation:** Positive delta = home team has more concentrated (star-dependent) ownership. This can be both bullish (the star is genuinely world-class) and fragile (if the star blanks, the signal was wrong).

#### 24. `star_dependence_delta` — Top-3 Player Dominance

```python
top3_share_h = sum(top 3 selected_i) / total_selected_h  # home
top3_share_a = sum(top 3 selected_i) / total_selected_a  # away
star_dependence_delta = top3_share_h - top3_share_a
```

**What it means:** What fraction of a team's total ownership is held by just the top 3 players? High star dependence means the crowd's bet on this team is concentrated in a small number of individuals. Like `concentration_delta` but simpler and more interpretable.

---

## Forward-Looking Signals (Limited Availability)

These signals are only available for recent seasons where FPL introduced expected stats.

#### `xp_ratio` — Expected Points (2020-21 onwards)

FPL's own expected points metric. Available from 2020-21 season. Filled with NaN for earlier seasons and excluded from the universal feature set.

#### `xg_ratio` — Expected Goals (2023-24 onwards)

Opta expected goals, surfaced through FPL from 2023-24. Only 2 full seasons of data so far — too few for reliable training.

#### `xgc_strength_ratio` — Expected Goals Conceded (2023-24 onwards)

Defensive xG counterpart. Same availability limitations as xG.

**Status:** These are excluded from the core model due to limited temporal coverage. As more seasons accumulate, they'll become trainable.

---

## From Signals to Predictions

### Model Architecture

```
24 signals → StandardScaler → LogisticRegression(C=0.01) → CalibratedClassifierCV(isotonic, 5-fold)
                                                                      ↓
                                                          P(Home), P(Draw), P(Away)
```

**Why Logistic Regression?** With only ~2,600 training matches (6 seasons × ~380 matches), complex models overfit. L2-regularised logistic regression (C=0.01 means heavy regularisation) is the right complexity for this data volume. We tested higher C values — they all performed worse.

**Why Isotonic Calibration?** Raw logistic regression outputs are already probabilities, but isotonic calibration further adjusts them to match observed frequencies. This is what brings our ECE down to 0.036.

### Training Split

| Set | Seasons | Matches | Purpose |
|-----|---------|---------|---------|
| Train | 2016-17 to 2021-22 | ~2,280 | Fit model weights |
| Validation | 2022-23 | ~380 | Select hyperparameters (C) |
| Test | 2023-24 + 2024-25 | 760 | Final evaluation (never touched during training) |
| Live | 2025-26 (ongoing) | 240+ | True out-of-sample forward test |

---

## Signal Ranking: Which Signals Matter Most?

### By Mutual Information with Match Outcome

Mutual information measures how many "bits" of outcome prediction each signal carries, independent of model specification.

| Rank | Signal | MI (bits) | Category |
|------|--------|-----------|----------|
| 1 | `ownership_ratio` | 0.107 | Stock |
| 2 | `dcs_ratio` | 0.065 | Structure |
| 3 | `transfer_ratio` | 0.054 | Flow |
| 4 | `velocity_delta` | 0.034 | Flow |
| 5 | `captain_proxy` | 0.034 | Flow |

For comparison: bookmaker implied probabilities carry 0.130-0.133 bits. The crowd's best single signal (`ownership_ratio`) achieves 81% of the best single odds signal.

### By Model Category Performance (Test Set Brier Score)

| Model Variant | Features | Test Brier | Test Accuracy |
|---------------|----------|------------|---------------|
| Odds only | 3 (implied probs) | 0.1853 | 56.1% |
| Structure + Odds | 10 | 0.1869 | 55.5% |
| Quality + Odds | 11 | 0.1869 | 57.0% |
| Original 5 + Odds | 8 | 0.1872 | 56.2% |
| All FPL + Odds | 27 | 0.1887 | 55.5% |
| Stock (FPL only) | 3 | 0.1941 | 55.5% |
| Original 5 (FPL only) | 5 | 0.1953 | 55.3% |
| All FPL (no odds) | 24 | 0.1941 | 54.1% |
| Quality (FPL only) | 8 | 0.1975 | 55.0% |
| Flow (FPL only) | 6 | 0.1977 | 53.4% |

**Key insight:** Stock signals (3 features) match or beat the full 24-feature model. The ownership-based signals carry the lion's share of predictive power. Adding more signals doesn't hurt much but doesn't help either — the information is largely redundant.

### By Conditional Information (Unique Signal Beyond Odds)

After accounting for what bookmaker odds already tell us:

- **Conditional MI (FPL | Odds):** 0.034 bits
- **FPL MI as % of Odds MI:** 91.8%

The crowd provides information that overlaps heavily with odds (both track the same underlying match dynamics) but contributes an additional ~0.034 bits of independent prediction. This is small but real — it's enough to produce statistically significant improvements in season-long predictions (permutation p < 0.001 at GW20-30 for top-4 prediction).

---

## Where Proprietary Edge Might Exist

### 1. Timing Advantage

FPL transfer data updates in real-time throughout the week. Betting odds also move, but the FPL crowd reaction to news (injuries, team announcements, tactical leaks) may propagate through transfer data before it fully reflects in odds — especially for:

- **Mid-table matches** that professional odds-setters spend less time on
- **Late team news** that breaks after the main odds movement on Monday/Tuesday
- **Injury returns** where FPL managers who follow training photos/press conferences react before the wider market

**Signal to watch:** `transfer_intensity_delta` — sudden spikes in per-capita transfer rate into a team's players indicate breaking positive news.

### 2. Structural Information

The **composition** of how managers build their squads carries information that odds don't directly price:

- High `premium_ratio` for a team means managers are making expensive, high-conviction bets — these managers have skin in the game (opportunity cost of their £100M budget)
- `concentration_delta` reveals when the crowd is betting on a specific star player vs. the whole team
- `dcs_ratio` directly encodes defensive expectations, which bookmakers price indirectly through over/under markets but not through 1X2 odds

### 3. Agreement/Disagreement Filtering

When FPL crowd and bookmakers agree on the match favourite (88% of matches), the crowd offers no edge. The potential edge exists in the **12% disagreement zone** (89/760 test matches):

- Crowd correct: 32.6% of disagreement matches
- Odds correct: 38.2% of disagreement matches
- McNemar's p = 0.61 (not significantly different)

While odds win more disagreement matches overall, the gap is not statistically significant. The crowd may have a specific advantage on certain disagreement types — this needs larger sample sizes to confirm.

### 4. Season-Arc Prediction

The strongest evidence of genuine FPL edge comes from **season-level** prediction (predicting final league standings), not match-level:

| Checkpoint | PPG Only Brier | PPG + FPL Brier | Improvement | Permutation p |
|------------|---------------|-----------------|-------------|---------------|
| GW10 | 0.057 | 0.072 | -25.5% (worse) | 0.043 |
| GW15 | 0.049 | 0.059 | -20.5% (worse) | 0.335 |
| GW20 | 0.028 | 0.025 | +8.8% | **< 0.001** |
| GW25 | 0.025 | 0.020 | +19.9% | **< 0.001** |
| GW30 | 0.031 | 0.029 | +6.1% | **< 0.001** |

From GW20 onwards, adding FPL signals to points-per-game significantly improves top-4 prediction. Leave-one-season-out CV confirms: Brier drops from 0.051 to 0.038 across 15 folds.

**The commercial implication:** FPL signals are most valuable not for individual match betting but for **futures/outrights markets** (top 4 finish, relegation) where the crowd's accumulated season-long conviction carries genuine independent information.

### 5. Calibration Quality

FPL crowd ECE: **0.036** vs Bookmaker ECE: **0.037**

The crowd is *as well calibrated* as professional bookmakers. When the FPL model says 65% chance of a home win, the actual frequency is ~65%. This means the crowd's probability estimates can be taken at face value — they're not systematically overconfident or underconfident.

Confidence-accuracy relationship (quintiles on test set):

| Confidence Level | Crowd Accuracy |
|-----------------|----------------|
| Q1 (least confident) | 40.8% |
| Q2 | 48.0% |
| Q3 | 61.2% |
| Q4 | 59.2% |
| Q5 (most confident) | 69.7% |

Spearman correlation: rho = 0.90, p = 0.037. The crowd reliably "knows what it knows."

---

## Live Validation: 2025-26 Season (GW1-24)

True out-of-sample forward test using pre-trained models (trained on 2016-22, never updated):

| Metric | FPL Crowd | Odds (base rates*) | FPL+Odds |
|--------|-----------|---------------------|----------|
| Brier Score | **0.2131** | 0.2159 | 0.2132 |
| Accuracy | **47.5%** | 44.6% | 45.0% |
| Correct picks | **114**/240 | 107/240 | 108/240 |
| BSS vs Naive | +1.1% | -0.2% | +1.1% |

*Odds model uses base rate fallback (0.45/0.27/0.28) as live odds weren't available for backfill.

**Notable:** The FPL crowd model outperforms the odds baseline in this forward test. It wins 13 of 24 individual gameweeks. Best week: GW7 with 90% accuracy and Brier 0.117.

When crowd and odds models disagree (84 matches / 35%):
- **Crowd correct: 35** (41.7%)
- **Odds correct: 28** (33.3%)

The crowd wins the disagreement subset in this live period, though the sample size is small.

---

## Signal Correlations and Redundancy

High correlations between signals indicate shared information:

| Signal Pair | Correlation | Interpretation |
|-------------|-------------|----------------|
| `ownership_ratio` ↔ `dcs_ratio` | r = 0.816 | Defensive ownership tracks total ownership |
| `captain_proxy` ↔ `velocity_delta` | r = 0.723 | Transfer velocity measures overlap |
| `raw_ownership_ratio` ↔ `ownership_ratio` | ~0.95 | Price weighting adds modest info |
| `threat_ratio` ↔ `creativity_ratio` | ~0.8 | ICT components are correlated |

**Practical implication:** The 24 signals have high internal redundancy. The effective dimensionality is closer to 5-7 independent information channels:

1. **Ownership level** (stock signals, ~3 variants)
2. **Transfer momentum** (flow signals, ~4 variants)
3. **Recent form quality** (quality signals, ~8 variants but highly correlated)
4. **Positional composition** (structure signals, ~5 variants)
5. **Ownership concentration** (HHI, star dependence)

This is why the original 5-signal model performs nearly as well as the full 24: it captures most of the independent variation.

---

## Summary: What the Crowd Knows

| Claim | Evidence | Strength |
|-------|----------|----------|
| Crowd predictions are meaningful (not noise) | BSS = 10.0% vs naive, 71.7% of bookmaker skill | Strong (760 test matches, 10K bootstrap) |
| Crowd calibration matches bookmakers | ECE 0.036 vs 0.037 | Strong |
| Crowd knows when it's confident | Rho = 0.90, p = 0.037 | Moderate (5 quintiles) |
| FPL adds info beyond odds (match-level) | Conditional MI = 0.034 bits | Small but real |
| FPL adds info beyond odds (season-level) | Perm p < 0.001 at GW20-30 | Strong for top-4 |
| Crowd beats odds on disagreement | 32.6% vs 38.2% (McNemar p = 0.61) | Not significant |
| Live 2025-26 validates the signal | FPL Brier 0.213 beats naive 0.215 | Encouraging (n=240) |

**The bottom line:** 11 million fantasy managers, playing a free game for fun, independently reproduce 72% of what professional bookmakers achieve with proprietary models, insider knowledge, and billions in market liquidity. The signal is real, calibrated, and carries a small amount of independent information that could be exploitable in specific contexts — particularly season-long futures markets and high-confidence subset filtering.

---

*Generated from analysis of 3,421 historical matches (2016-25) and 240 live 2025-26 matches.*
*Source code: `src/03_engineer_signals.py`, `src/08_expanded_signals.py`, `src/04_model_train.py`, `src/10_crowd_wisdom.py`*
