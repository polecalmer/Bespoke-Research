"""Generate all 8 required charts with dark theme styling.

1. Signal distributions by match outcome
2. Divergence vs ROI
3. Calibration plot
4. Brier Score comparison
5. Cumulative P&L
6. Signal correlation heatmap
7. Season breakdown (ROI by season)
8. Quintile analysis

Outputs: outputs/charts/*.png
"""

import sys
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.calibration import calibration_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    setup_chart_style, CHART_COLORS, CHARTS_DIR, MODEL_DIR,
    load_parquet, TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS,
)

NAVY = CHART_COLORS['navy']
TEAL = CHART_COLORS['teal']
PURPLE = CHART_COLORS['purple']
CORAL = CHART_COLORS['coral']
WHITE = CHART_COLORS['white']

SIGNAL_COLS = ['transfer_ratio', 'ownership_ratio', 'captain_proxy', 'dcs_ratio', 'velocity_delta']
SIGNAL_LABELS = {
    'transfer_ratio': 'Transfer Ratio',
    'ownership_ratio': 'Ownership Ratio',
    'captain_proxy': 'Captain Proxy',
    'dcs_ratio': 'Defensive Confidence',
    'velocity_delta': 'Ownership Velocity',
}


def chart1_signal_distributions(signals: pd.DataFrame):
    """Chart 1: Histogram of each FPL signal, faceted by match outcome (H/D/A)."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, col in enumerate(SIGNAL_COLS):
        ax = axes[i]
        for outcome, color, label in [('H', TEAL, 'Home Win'), ('D', PURPLE, 'Draw'), ('A', CORAL, 'Away Win')]:
            data = signals[signals['result'] == outcome][col].dropna()
            ax.hist(data, bins=40, alpha=0.55, color=color, label=label, density=True)
        ax.set_title(SIGNAL_LABELS[col], fontsize=12, loc='left')
        ax.legend(fontsize=8, framealpha=0.3)
        ax.grid(True, alpha=0.2)

    axes[-1].set_visible(False)
    fig.suptitle('FPL Signal Distributions by Match Outcome', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(CHARTS_DIR / '01_signal_distributions.png')
    plt.close()
    print("  Chart 1: Signal distributions saved")


def chart2_divergence_vs_roi(backtest: pd.DataFrame):
    """Chart 2: Scatter/line showing divergence threshold vs cumulative ROI."""
    fig, ax = plt.subplots(figsize=(10, 6))

    thresholds = np.arange(0.01, 0.16, 0.01)
    for outcome, color, label in [('H', TEAL, 'Home'), ('D', PURPLE, 'Draw'), ('A', CORAL, 'Away')]:
        rois = []
        counts = []
        for t in thresholds:
            bets = backtest[(backtest['bet_outcome'] == outcome) & (backtest['divergence'] > t)]
            if len(bets) >= 5:
                roi = 100.0 * bets['profit'].sum() / len(bets)
                rois.append(roi)
                counts.append(len(bets))
            else:
                rois.append(np.nan)
                counts.append(0)
        ax.plot(thresholds, rois, color=color, label=label, linewidth=2, marker='o', markersize=4)

    # Overall
    rois_all = []
    for t in thresholds:
        bets = backtest[backtest['divergence'] > t]
        if len(bets) >= 5:
            rois_all.append(100.0 * bets['profit'].sum() / len(bets))
        else:
            rois_all.append(np.nan)
    ax.plot(thresholds, rois_all, color=WHITE, label='All', linewidth=2.5, linestyle='--')

    ax.axhline(y=0, color=WHITE, alpha=0.3, linestyle='-')
    ax.set_xlabel('Divergence Threshold')
    ax.set_ylabel('ROI (%)')
    ax.set_title('ROI by Divergence Threshold', fontsize=14, loc='left')
    ax.legend(framealpha=0.3)
    ax.grid(True, alpha=0.2)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    plt.tight_layout()
    fig.savefig(CHARTS_DIR / '02_divergence_vs_roi.png')
    plt.close()
    print("  Chart 2: Divergence vs ROI saved")


def chart3_calibration(test_preds: pd.DataFrame):
    """Chart 3: Reliability diagram for each model."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # For home win probability (class=2=Home)
    y_true = (test_preds['result'] == 'H').astype(int)

    models = [
        ('odds_only', NAVY, 'Odds-Only'),
        ('fpl_only', PURPLE, 'FPL-Only'),
        ('fpl_odds', TEAL, 'FPL+Odds'),
    ]

    for model_name, color, label in models:
        y_prob = test_preds[f'p_{model_name}_H']
        if y_prob.isna().all():
            continue
        valid = y_prob.notna()
        try:
            prob_true, prob_pred = calibration_curve(
                y_true[valid], y_prob[valid], n_bins=10, strategy='uniform'
            )
            ax.plot(prob_pred, prob_true, color=color, label=label, linewidth=2, marker='s', markersize=5)
        except Exception:
            pass

    # Also add market implied probability
    y_market = test_preds['implied_prob_h']
    valid_market = y_market.notna()
    try:
        prob_true_m, prob_pred_m = calibration_curve(
            y_true[valid_market], y_market[valid_market], n_bins=10, strategy='uniform'
        )
        ax.plot(prob_pred_m, prob_true_m, color=CORAL, label='Market Odds', linewidth=2,
                marker='D', markersize=5, linestyle='--')
    except Exception:
        pass

    ax.plot([0, 1], [0, 1], color=WHITE, alpha=0.4, linestyle='--', label='Perfect')
    ax.set_xlabel('Predicted Probability (Home Win)')
    ax.set_ylabel('Observed Frequency')
    ax.set_title('Calibration Plot — Home Win Probability', fontsize=14, loc='left')
    ax.legend(framealpha=0.3)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(CHARTS_DIR / '03_calibration_plot.png')
    plt.close()
    print("  Chart 3: Calibration plot saved")


def chart4_brier_comparison(metrics: dict):
    """Chart 4: Bar chart comparing Brier scores across models."""
    fig, ax = plt.subplots(figsize=(8, 5))

    models = ['odds_only', 'fpl_only', 'fpl_odds']
    labels = ['Odds-Only', 'FPL-Only', 'FPL+Odds']
    colors = [NAVY, PURPLE, TEAL]

    val_briers = [metrics[m]['brier'] for m in models]
    test_briers = [metrics[m]['test_brier'] for m in models]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width / 2, val_briers, width, label='Validation', color=colors, alpha=0.7)
    bars2 = ax.bar(x + width / 2, test_briers, width, label='Test', color=colors, alpha=1.0,
                   edgecolor=WHITE, linewidth=1)

    ax.set_ylabel('Brier Score (lower is better)')
    ax.set_title('Model Comparison — Brier Score', fontsize=14, loc='left')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(framealpha=0.3)
    ax.grid(True, alpha=0.2, axis='y')

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.001,
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=8, color=WHITE)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.001,
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=8, color=WHITE)

    plt.tight_layout()
    fig.savefig(CHARTS_DIR / '04_brier_comparison.png')
    plt.close()
    print("  Chart 4: Brier comparison saved")


def chart5_cumulative_pnl(backtest: pd.DataFrame):
    """Chart 5: Cumulative P&L line chart per threshold."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Use threshold=0.03 backtest (most bets) — recompute from raw data at each threshold
    colors_map = {0.03: TEAL, 0.05: PURPLE, 0.07: CORAL, 0.10: NAVY}

    for threshold in [0.03, 0.05, 0.07, 0.10]:
        subset = backtest[backtest['threshold'] == threshold].sort_values(['season', 'gw'])
        if len(subset) == 0:
            continue
        cumulative = np.cumsum(subset['profit'].values)
        ax.plot(range(len(cumulative)), cumulative, color=colors_map[threshold],
                label=f'Threshold={threshold}', linewidth=1.5)

    ax.axhline(y=0, color=WHITE, alpha=0.3, linestyle='-')
    ax.set_xlabel('Bet Number')
    ax.set_ylabel('Cumulative P&L (units)')
    ax.set_title('Cumulative P&L — Divergence Strategy', fontsize=14, loc='left')
    ax.legend(framealpha=0.3)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(CHARTS_DIR / '05_cumulative_pnl.png')
    plt.close()
    print("  Chart 5: Cumulative P&L saved")


def chart6_correlation_matrix(signals: pd.DataFrame):
    """Chart 6: Heatmap of FPL signals + odds correlations."""
    cols = SIGNAL_COLS + ['implied_prob_h', 'implied_prob_d', 'implied_prob_a']
    labels = ['Transfer\nRatio', 'Ownership\nRatio', 'Captain\nProxy', 'Defensive\nConfidence',
              'Ownership\nVelocity', 'Implied\nP(Home)', 'Implied\nP(Draw)', 'Implied\nP(Away)']

    corr = signals[cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.zeros_like(corr, dtype=bool)

    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr, annot=True, fmt='.2f', cmap=cmap, center=0,
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, linecolor=CHART_COLORS['edge'],
                ax=ax, cbar_kws={'shrink': 0.8},
                annot_kws={'size': 9, 'color': WHITE})
    ax.set_title('Signal Correlation Matrix', fontsize=14, loc='left')

    plt.tight_layout()
    fig.savefig(CHARTS_DIR / '06_correlation_matrix.png')
    plt.close()
    print("  Chart 6: Correlation matrix saved")


def chart7_season_breakdown(backtest: pd.DataFrame, signals: pd.DataFrame):
    """Chart 7: ROI by season (bar chart) at threshold=0.05."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Use ALL data through all thresholds — compute at 0.05
    subset = backtest[backtest['threshold'] == 0.05]

    all_seasons = sorted(signals['season'].unique())
    rois = []
    counts = []
    bar_colors = []

    for s in all_seasons:
        ss = subset[subset['season'] == s]
        if len(ss) > 0:
            roi = 100.0 * ss['profit'].sum() / len(ss)
            rois.append(roi)
            counts.append(len(ss))
        else:
            rois.append(0)
            counts.append(0)

        if s in TRAIN_SEASONS:
            bar_colors.append(NAVY)
        elif s in VAL_SEASONS:
            bar_colors.append(PURPLE)
        else:
            bar_colors.append(TEAL)

    x = range(len(all_seasons))
    bars = ax.bar(x, rois, color=bar_colors, alpha=0.8, edgecolor=WHITE, linewidth=0.5)

    # Add bet count labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                    f'n={count}', ha='center', va='bottom', fontsize=8, color=WHITE)

    ax.axhline(y=0, color=WHITE, alpha=0.3, linestyle='-')
    ax.set_xticks(list(x))
    ax.set_xticklabels(all_seasons, rotation=45)
    ax.set_ylabel('ROI (%)')
    ax.set_title('ROI by Season (threshold=0.05, test set only)', fontsize=14, loc='left')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(True, alpha=0.2, axis='y')

    # Legend for splits
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=NAVY, label='Train'),
        Patch(facecolor=PURPLE, label='Validation'),
        Patch(facecolor=TEAL, label='Test'),
    ]
    ax.legend(handles=legend_elements, framealpha=0.3)

    plt.tight_layout()
    fig.savefig(CHARTS_DIR / '07_season_breakdown.png')
    plt.close()
    print("  Chart 7: Season breakdown saved")


def chart8_quintile_analysis(signals: pd.DataFrame):
    """Chart 8: For each signal, bucket into quintiles, show actual vs implied home win rate."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, col in enumerate(SIGNAL_COLS):
        ax = axes[i]
        valid = signals[[col, 'result', 'implied_prob_h']].dropna()
        if len(valid) == 0:
            continue

        valid['quintile'] = pd.qcut(valid[col], 5, labels=['Q1\n(Low)', 'Q2', 'Q3', 'Q4', 'Q5\n(High)'],
                                    duplicates='drop')

        actual = valid.groupby('quintile', observed=True).apply(
            lambda g: (g['result'] == 'H').mean() * 100, include_groups=False)
        implied = valid.groupby('quintile', observed=True)['implied_prob_h'].mean() * 100

        x = range(len(actual))
        width = 0.35
        ax.bar([xi - width / 2 for xi in x], actual.values, width, color=TEAL, alpha=0.8, label='Actual')
        ax.bar([xi + width / 2 for xi in x], implied.values, width, color=CORAL, alpha=0.8, label='Implied')

        ax.set_xticks(list(x))
        ax.set_xticklabels(actual.index, fontsize=8)
        ax.set_ylabel('Home Win %')
        ax.set_title(SIGNAL_LABELS[col], fontsize=11, loc='left')
        ax.legend(fontsize=8, framealpha=0.3)
        ax.grid(True, alpha=0.2, axis='y')

    axes[-1].set_visible(False)
    fig.suptitle('Quintile Analysis — Actual vs Implied Home Win Rate', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(CHARTS_DIR / '08_quintile_analysis.png')
    plt.close()
    print("  Chart 8: Quintile analysis saved")


def main():
    setup_chart_style()
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating charts...")

    signals = load_parquet('signals.parquet')
    backtest = load_parquet('backtest.parquet')
    test_preds = pd.read_parquet(MODEL_DIR / 'test_predictions.parquet')

    with open(MODEL_DIR / 'metrics.json') as f:
        metrics = json.load(f)

    chart1_signal_distributions(signals)
    chart2_divergence_vs_roi(backtest)
    chart3_calibration(test_preds)
    chart4_brier_comparison(metrics)
    chart5_cumulative_pnl(backtest)
    chart6_correlation_matrix(signals)
    chart7_season_breakdown(backtest, signals)
    chart8_quintile_analysis(signals)

    print(f"\nAll 8 charts saved to {CHARTS_DIR}")


if __name__ == '__main__':
    main()
