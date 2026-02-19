# FPL Betting Signal Research

Testing whether Fantasy Premier League aggregate behavior (ownership, captaincy, transfers) contains predictive signal for EPL match outcomes that is **not already priced into betting odds**.

## Result

**No actionable signal found.** FPL crowd data does not improve upon betting odds for EPL match prediction. See [outputs/results.md](outputs/results.md) for the full analysis.

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline

Run scripts in order:

```bash
python src/01_fetch_data.py         # Download FPL + odds data
python src/02_build_match_dataset.py # Build match-level dataset
python src/03_engineer_signals.py    # Compute FPL signals
python src/04_model_train.py         # Train & evaluate models
python src/05_backtest.py            # Run betting simulation
python src/06_visualize.py           # Generate charts
```

## Data

- **FPL**: 9 seasons (2016-17 to 2024-25) from [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League)
- **Odds**: Bet365 match odds from [football-data.co.uk](https://www.football-data.co.uk/englandm.php)
- **3,421 matches** total, all matched with odds

## Key Metrics

| Model | Brier Score (Test) | BSS vs Odds |
|-------|-------------------|-------------|
| Odds-Only | 0.1856 | baseline |
| FPL-Only | 0.1938 | — |
| FPL+Odds | 0.1872 | -0.008 |
