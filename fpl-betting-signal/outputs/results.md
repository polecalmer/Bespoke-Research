# FPL Betting Signal Research: Results

## Executive Summary

**Does FPL crowd behavior add predictive value over betting odds for EPL match outcomes?**

**No.** After testing 5 FPL-derived signals across 3,421 EPL matches over 9 seasons (2016-17 to 2024-25), we find no statistically significant evidence that Fantasy Premier League aggregate data (ownership, transfers, defensive selections) improves upon the predictive power already embedded in Bet365 match odds.

The FPL+Odds model achieves a Brier Skill Score of **-0.008** on the test set (2023-24, 2024-25), meaning it performs *worse* than the Odds-Only baseline. The divergence-based betting strategy produces negative ROI at most thresholds, with no statistically significant positive results.

**The null hypothesis holds: betting odds are efficient with respect to FPL crowd signals.**

---

## Data

| Item | Count |
|------|-------|
| Seasons | 9 (2016-17 to 2024-25) |
| Total matches | 3,421 |
| Matches with odds | 3,421 (100%) |
| Training set | 2,281 matches (2016-17 to 2021-22) |
| Validation set | 380 matches (2022-23) |
| Test set | 760 matches (2023-24, 2024-25) |

FPL data sourced from [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League). Betting odds from football-data.co.uk (via [xgabora mirror](https://github.com/xgabora/Club-Football-Match-Data-2000-2025)).

---

## Signals

Five FPL-derived features computed per match:

| Signal | Description | Corr with Odds |
|--------|-------------|----------------|
| Transfer Ratio | Net transfers, normalized home vs away | 0.326 |
| Ownership Ratio | Price-weighted ownership, home/(home+away) | 0.876 |
| Captain Proxy | Transfer velocity differential (captaincy unavailable) | 0.360 |
| Defensive Confidence | DEF+GK ownership ratio | 0.680 |
| Ownership Velocity | Week-over-week ownership change delta | 0.362 |

**Key observation:** Ownership Ratio correlates at r=0.876 with implied home win probability, meaning it largely duplicates the information already in odds. The remaining signals (Transfer Ratio, Captain Proxy, Velocity) have moderate correlations (0.33-0.36), suggesting some independent variance, but not enough to improve predictions.

![Signal Correlation Matrix](charts/06_correlation_matrix.png)

---

## Model Performance

### Validation Set (2022-23)

| Model | Brier Score | Log Loss | Accuracy |
|-------|-------------|----------|----------|
| **Odds-Only** | **0.1927** | **0.9754** | **0.561** |
| FPL-Only | 0.1960 | 1.3442 | 0.545 |
| FPL+Odds | 0.1928 | 0.9878 | 0.547 |

### Test Set (2023-24, 2024-25)

| Model | Brier Score | Log Loss | Accuracy |
|-------|-------------|----------|----------|
| **Odds-Only** | **0.1856** | **0.9440** | 0.559 |
| FPL-Only | 0.1938 | 1.0732 | 0.558 |
| FPL+Odds | 0.1872 | 0.9530 | **0.561** |

### Brier Skill Score

| Metric | Value | Interpretation |
|--------|-------|----------------|
| BSS (Validation) | -0.0006 | FPL+Odds no better than Odds-Only |
| BSS (Test) | -0.0082 | FPL+Odds slightly worse |

A BSS > 0 would indicate FPL adds value. Both validation and test BSS are negative.

![Brier Score Comparison](charts/04_brier_comparison.png)

![Calibration Plot](charts/03_calibration_plot.png)

---

## Feature Importance

Model coefficients for the FPL+Odds combined model (mean absolute coefficient):

| Feature | Importance | Type |
|---------|-----------|------|
| Implied P(Away) | 0.1927 | Odds |
| Implied P(Home) | 0.1700 | Odds |
| Implied P(Draw) | 0.0897 | Odds |
| Defensive Confidence | 0.0387 | FPL |
| Captain Proxy | 0.0286 | FPL |
| Ownership Velocity | 0.0230 | FPL |
| Ownership Ratio | 0.0150 | FPL |
| Transfer Ratio | 0.0109 | FPL |

Odds features dominate. Among FPL signals, Defensive Confidence (DEF/GK ownership) has the largest coefficient, suggesting it captures *some* information about clean sheet expectations, but not enough to overcome the noise.

---

## Backtest Results

Flat-stake betting simulation on test set (760 matches):

| Threshold | Bets | Wins | Hit% | ROI% | Max DD | Sharpe | p-value | 95% CI |
|-----------|------|------|------|------|--------|--------|---------|--------|
| 0.03 | 471 | 223 | 47.3 | -7.2 | 43.6 | -1.33 | 0.437 | [-17.7%, 3.6%] |
| 0.05 | 218 | 108 | 49.5 | -7.7 | 24.2 | -1.00 | 0.485 | [-22.5%, 8.1%] |
| 0.07 | 112 | 68 | 60.7 | -2.3 | 12.5 | -0.27 | 0.197 | [-19.0%, 15.0%] |
| 0.10 | 49 | 34 | 69.4 | +5.9 | 3.5 | 0.48 | 0.169 | [-17.1%, 30.1%] |

The only positive ROI (5.9% at threshold=0.10) comes from just 49 bets with very wide confidence intervals crossing zero. **Not statistically significant at any threshold** (all p-values > 0.15).

### Season Breakdown (threshold=0.05)

| Season | Bets | Wins | ROI% |
|--------|------|------|------|
| 2023-24 | 116 | 60 | +3.6 |
| 2024-25 | 102 | 48 | -20.6 |

Inconsistent across seasons — another sign of no true edge.

### Outcome Breakdown (threshold=0.05)

| Outcome | Bets | Wins | ROI% |
|---------|------|------|------|
| Home | 85 | 31 | -25.1 |
| Draw | 29 | 7 | +3.0 |
| Away | 104 | 70 | +3.5 |

![Cumulative P&L](charts/05_cumulative_pnl.png)

![Divergence vs ROI](charts/02_divergence_vs_roi.png)

---

## Signal Analysis

### Signal Distributions

![Signal Distributions](charts/01_signal_distributions.png)

The signal distributions overlap heavily across outcomes (Home/Draw/Away), confirming low discriminative power.

### Quintile Analysis

![Quintile Analysis](charts/08_quintile_analysis.png)

When matches are bucketed into quintiles by each signal, the actual home win rates closely track the implied odds rates. There are no systematic deviations where the signal "knows" something the odds don't.

### Season Breakdown

![Season Breakdown](charts/07_season_breakdown.png)

---

## Why FPL Signals Don't Beat Odds

1. **Ownership mirrors odds.** The correlation between Ownership Ratio and implied probability is 0.876 — FPL managers and bookmakers are using the same public information (form, injuries, fixtures).

2. **Transfers are reactive, not predictive.** FPL transfer windows close before the gameweek deadline, and managers tend to transfer *after* news breaks, meaning transfer momentum reflects already-known information.

3. **Bookmakers aggregate more information.** Odds incorporate insider knowledge, algorithmic models, market flows, team news, weather, and much more. FPL crowd wisdom is a strict subset.

4. **Small independent signal drowned by noise.** The FPL signals with lower odds correlation (Transfer Ratio ~0.33, Captain Proxy ~0.36) do carry *some* independent variance, but the signal-to-noise ratio is too low to be exploitable after accounting for the bookmaker's margin (typically ~5% overround).

---

## Conclusions

1. **FPL crowd signals do not add predictive value over betting odds** for EPL match outcome prediction. The Brier Skill Score is negative on both validation (-0.001) and test (-0.008) sets.

2. **No profitable betting strategy exists** based on FPL-odds divergence. All tested thresholds produce either negative or statistically insignificant ROI.

3. **The strongest FPL signal is Defensive Confidence** (DEF/GK ownership), which captures some clean sheet expectation signal. This could potentially be tested against the Over/Under or Clean Sheet betting markets specifically, though this was outside the scope of this study.

4. **FPL ownership is highly correlated with odds** (r=0.876), confirming that the FPL crowd and bookmakers are processing the same underlying information.

5. **A valid negative result.** The null hypothesis that odds are efficient with respect to FPL crowd signals cannot be rejected. This is a useful finding — it means FPL signals can be safely ignored when building EPL betting models.

---

## Potential Next Steps (if pursuing further)

- Test DCS signal specifically against Over/Under and Clean Sheet markets (different odds, different efficiency)
- Investigate Top 10K manager data (elite subset may contain stronger signal than the crowd)
- Test captaincy concentration if real-time API data becomes available
- Explore in-play FPL data (live point updates during matches) as a signal for live betting markets
- Test Asian Handicap markets which may be less efficient than 1X2
