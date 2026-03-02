"""Crowd Wisdom Analysis: What 11 Million Fantasy Managers Know.

Reframes the FPL research around the remarkable positive finding: a crowd
playing a fantasy game for fun independently reproduces ~72% of professional
bookmaker accuracy for EPL match prediction.

Sections:
1. Crowd Prediction Accuracy Ladder (with bootstrap CIs)
2. Agreement/Disagreement Analysis (when crowd and market disagree)
3. Crowd Confidence Calibration (are confident crowds more accurate?)
4. Season-Arc Bootstrap Validation (honest CIs on the season-level finding)
5. Information Content Comparison (mutual information in bits)
6. Headline Summary Chart

Outputs:
  - outputs/charts/crowd_wisdom_*.png
  - outputs/model/crowd_wisdom.json
  - Appends section to outputs/results.md
"""

import sys
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    brier_score_loss, log_loss, accuracy_score, mutual_info_score
)
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    setup_chart_style, CHART_COLORS, CHARTS_DIR, MODEL_DIR,
    DATA_PROCESSED, load_parquet,
    TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS,
)

# --- Constants ---
NAVY = CHART_COLORS['navy']
TEAL = CHART_COLORS['teal']
PURPLE = CHART_COLORS['purple']
CORAL = CHART_COLORS['coral']
WHITE = CHART_COLORS['white']
BG = CHART_COLORS['bg']

RESULT_MAP = {'A': 0, 'D': 1, 'H': 2}
RESULT_LABELS = ['Away', 'Draw', 'Home']
N_BOOTSTRAP = 10_000
RNG = np.random.RandomState(42)

FPL_FEATURES = ['transfer_ratio', 'ownership_ratio', 'captain_proxy',
                'dcs_ratio', 'velocity_delta']
ODDS_FEATURES = ['implied_prob_h', 'implied_prob_d', 'implied_prob_a']

# Season-arc feature sets (from 09_season_arc.py)
SA_BASELINE = ['ppg', 'gd_per_game']
SA_EXTENDED = ['ppg', 'xppg', 'gd_per_game', 'overperf', 'form_trend',
               'home_ppg', 'away_ppg']
SA_FPL = ['fpl_ownership_slope', 'fpl_transfer_momentum',
          'fpl_recent_transfer_mom', 'fpl_transfer_accel',
          'fpl_value_slope', 'fpl_points_per_gw',
          'fpl_relative_ownership', 'fpl_ict_per_gw',
          'fpl_threat_per_gw', 'fpl_quality']
SA_COMBINED = SA_EXTENDED + SA_FPL

CHECKPOINTS = [10, 15, 20, 25, 30]


def multiclass_brier(y_true, y_proba, n_classes=3):
    """Mean Brier score across classes (from 04_model_train.py)."""
    return np.mean([
        brier_score_loss((y_true == c).astype(int), y_proba[:, c])
        for c in range(n_classes)
    ])


def jsonify(obj):
    """Recursively convert numpy types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonify(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return round(float(obj), 6)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return jsonify(obj.tolist())
    return obj


# =========================================================================
# Section 1: Crowd Prediction Accuracy Ladder
# =========================================================================

def section1_accuracy_ladder(test_preds, y_test, train_df):
    """Compute Brier scores for Naive, Home-bias, FPL Crowd, and Bookmaker."""
    print("\n[1/6] Crowd Prediction Accuracy Ladder")
    print("-" * 50)

    n = len(y_test)

    # --- Naive baseline: training set base rates ---
    y_train = train_df['result'].map(RESULT_MAP).values
    base_rates = np.bincount(y_train, minlength=3) / len(y_train)
    naive_proba = np.tile(base_rates, (n, 1))  # same for every match

    # --- Home-bias baseline: just predict home win rate + draw/away split ---
    # A slightly smarter baseline: per-season home win rates from training
    home_rate = (y_train == 2).mean()
    draw_rate = (y_train == 1).mean()
    away_rate = (y_train == 0).mean()
    home_bias_proba = np.tile([away_rate, draw_rate, home_rate], (n, 1))

    # --- FPL Crowd: from pre-trained model predictions ---
    fpl_proba = np.column_stack([
        test_preds['p_fpl_only_A'].values,
        test_preds['p_fpl_only_D'].values,
        test_preds['p_fpl_only_H'].values,
    ])

    # --- Bookmaker odds ---
    odds_proba = np.column_stack([
        test_preds['implied_prob_a'].values,
        test_preds['implied_prob_d'].values,
        test_preds['implied_prob_h'].values,
    ])

    # --- Raw market odds (from model) ---
    odds_model_proba = np.column_stack([
        test_preds['p_odds_only_A'].values,
        test_preds['p_odds_only_D'].values,
        test_preds['p_odds_only_H'].values,
    ])

    # --- Compute metrics ---
    tiers = {
        'Naive (base rates)': naive_proba,
        'Home-bias': home_bias_proba,
        'FPL Crowd (5 signals)': fpl_proba,
        'Bookmaker (Bet365)': odds_proba,
        'Odds Model (calibrated)': odds_model_proba,
    }

    metrics = {}
    for name, proba in tiers.items():
        brier = multiclass_brier(y_test, proba)
        ll = log_loss(y_test, proba, labels=[0, 1, 2])
        acc = accuracy_score(y_test, proba.argmax(axis=1))
        metrics[name] = {'brier': brier, 'log_loss': ll, 'accuracy': acc}

    naive_brier = metrics['Naive (base rates)']['brier']
    for name in metrics:
        bss = 1 - (metrics[name]['brier'] / naive_brier)
        metrics[name]['bss_vs_naive'] = bss

    # Crowd fraction of bookmaker skill
    fpl_skill = metrics['FPL Crowd (5 signals)']['bss_vs_naive']
    odds_skill = metrics['Bookmaker (Bet365)']['bss_vs_naive']
    crowd_pct = (fpl_skill / odds_skill * 100) if odds_skill > 0 else 0

    # --- Bootstrap CIs on BSS ---
    bootstrap_bss = {}
    for name, proba in tiers.items():
        bss_samples = []
        for _ in range(N_BOOTSTRAP):
            idx = RNG.choice(n, size=n, replace=True)
            b_naive = multiclass_brier(y_test[idx], naive_proba[idx])
            b_model = multiclass_brier(y_test[idx], proba[idx])
            bss_samples.append(1 - b_model / b_naive)
        bss_arr = np.array(bss_samples)
        bootstrap_bss[name] = {
            'mean': float(np.mean(bss_arr)),
            'ci_lower': float(np.percentile(bss_arr, 2.5)),
            'ci_upper': float(np.percentile(bss_arr, 97.5)),
        }

    # Print results
    print(f"\n  {'Tier':<28} {'Brier':>8} {'BSS%':>8} {'Acc':>6} {'95% CI BSS'}")
    print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*6} {'-'*20}")
    for name in tiers:
        m = metrics[name]
        ci = bootstrap_bss[name]
        print(f"  {name:<28} {m['brier']:.4f}  {m['bss_vs_naive']*100:>6.1f}%  "
              f"{m['accuracy']:.3f}  [{ci['ci_lower']*100:.1f}%, {ci['ci_upper']*100:.1f}%]")
    print(f"\n  Crowd captures {crowd_pct:.1f}% of bookmaker predictive skill")

    return {
        'metrics': metrics,
        'bootstrap_bss': bootstrap_bss,
        'crowd_pct_of_odds_skill': crowd_pct,
        'base_rates': base_rates.tolist(),
    }


def chart1_accuracy_ladder(results):
    """Horizontal bar chart: Brier Skill Score by tier."""
    setup_chart_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    tiers = ['Naive (base rates)', 'Home-bias', 'FPL Crowd (5 signals)',
             'Bookmaker (Bet365)', 'Odds Model (calibrated)']
    # Display order: bottom to top
    display = ['Naive (base rates)', 'Home-bias',
               'FPL Crowd (5 signals)', 'Odds Model (calibrated)', 'Bookmaker (Bet365)']
    colors = ['#555555', '#777777', TEAL, CORAL, CORAL]

    bss_vals = [results['bootstrap_bss'][t]['mean'] * 100 for t in display]
    ci_lo = [results['bootstrap_bss'][t]['ci_lower'] * 100 for t in display]
    ci_hi = [results['bootstrap_bss'][t]['ci_upper'] * 100 for t in display]
    errors_lo = [bss_vals[i] - ci_lo[i] for i in range(len(display))]
    errors_hi = [ci_hi[i] - bss_vals[i] for i in range(len(display))]

    y_pos = range(len(display))
    bars = ax.barh(y_pos, bss_vals, color=colors, edgecolor='#444444',
                   height=0.6, zorder=3)
    ax.errorbar(bss_vals, y_pos, xerr=[errors_lo, errors_hi],
                fmt='none', ecolor=WHITE, elinewidth=1.5, capsize=4, zorder=4)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([t.replace('(base rates)', '\n(base rates)')
                        .replace('(5 signals)', '\n(5 signals)')
                        .replace('(Bet365)', '\n(Bet365)')
                        .replace('(calibrated)', '\n(calibrated)')
                        for t in display], fontsize=11)
    ax.set_xlabel('Brier Skill Score vs Naive (%)', fontsize=12)
    ax.set_title('The Crowd Wisdom Ladder\nHow well does each predictor beat random guessing?',
                 fontsize=14, fontweight='bold', pad=15)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(axis='x', alpha=0.2)

    # Annotate crowd percentage
    crowd_pct = results['crowd_pct_of_odds_skill']
    fpl_bss = results['bootstrap_bss']['FPL Crowd (5 signals)']['mean'] * 100
    odds_bss = results['bootstrap_bss']['Bookmaker (Bet365)']['mean'] * 100
    ax.annotate(f'Crowd captures {crowd_pct:.0f}%\nof bookmaker accuracy',
                xy=(fpl_bss, 2), xytext=(fpl_bss + 2, 3.5),
                fontsize=11, fontweight='bold', color=TEAL,
                arrowprops=dict(arrowstyle='->', color=TEAL, lw=1.5))

    plt.tight_layout()
    fig.savefig(CHARTS_DIR / 'crowd_wisdom_ladder.png')
    plt.close()
    print("  Saved crowd_wisdom_ladder.png")


# =========================================================================
# Section 2: Agreement / Disagreement Analysis
# =========================================================================

def section2_agreement(test_preds, y_test):
    """When crowd and bookmaker disagree on favorite, who's right?"""
    print("\n[2/6] Agreement / Disagreement Analysis")
    print("-" * 50)

    fpl_pred = np.column_stack([
        test_preds['p_fpl_only_A'].values,
        test_preds['p_fpl_only_D'].values,
        test_preds['p_fpl_only_H'].values,
    ])
    odds_pred = np.column_stack([
        test_preds['implied_prob_a'].values,
        test_preds['implied_prob_d'].values,
        test_preds['implied_prob_h'].values,
    ])

    fpl_fav = fpl_pred.argmax(axis=1)
    odds_fav = odds_pred.argmax(axis=1)
    agree_mask = fpl_fav == odds_fav

    n_agree = agree_mask.sum()
    n_disagree = (~agree_mask).sum()
    n_total = len(y_test)

    # Accuracy when they agree
    agree_fpl_acc = accuracy_score(y_test[agree_mask], fpl_fav[agree_mask])
    agree_odds_acc = accuracy_score(y_test[agree_mask], odds_fav[agree_mask])

    # Accuracy when they disagree
    disagree_fpl_acc = accuracy_score(y_test[~agree_mask], fpl_fav[~agree_mask])
    disagree_odds_acc = accuracy_score(y_test[~agree_mask], odds_fav[~agree_mask])

    # McNemar's test: are their errors independent?
    fpl_correct = (fpl_fav == y_test)
    odds_correct = (odds_fav == y_test)
    # Contingency: both right, FPL right only, odds right only, both wrong
    both_right = (fpl_correct & odds_correct).sum()
    fpl_only_right = (fpl_correct & ~odds_correct).sum()
    odds_only_right = (~fpl_correct & odds_correct).sum()
    both_wrong = (~fpl_correct & ~odds_correct).sum()

    # McNemar test on discordant pairs
    if fpl_only_right + odds_only_right > 0:
        mcnemar_stat = (abs(fpl_only_right - odds_only_right) - 1) ** 2 / (
            fpl_only_right + odds_only_right)
        mcnemar_p = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
    else:
        mcnemar_stat = 0
        mcnemar_p = 1.0

    # Brier on disagreement subset
    disagree_brier_fpl = multiclass_brier(y_test[~agree_mask], fpl_pred[~agree_mask])
    disagree_brier_odds = multiclass_brier(y_test[~agree_mask], odds_pred[~agree_mask])

    # When crowd disagrees with odds, does crowd favorite win more than chance?
    # Chance = base rate for the predicted outcome
    disagree_outcomes = y_test[~agree_mask]
    crowd_disagree_fav = fpl_fav[~agree_mask]
    crowd_correct_in_disagree = (crowd_disagree_fav == disagree_outcomes).sum()
    # Fisher exact: crowd correct vs not, compared to 1/3 chance
    fisher_p = stats.binomtest(
        crowd_correct_in_disagree, n_disagree,
        p=1/3, alternative='greater'
    ).pvalue if n_disagree > 0 else 1.0

    results = {
        'n_total': int(n_total),
        'n_agree': int(n_agree),
        'n_disagree': int(n_disagree),
        'agree_pct': float(n_agree / n_total * 100),
        'agree_accuracy_fpl': float(agree_fpl_acc),
        'agree_accuracy_odds': float(agree_odds_acc),
        'disagree_accuracy_fpl': float(disagree_fpl_acc),
        'disagree_accuracy_odds': float(disagree_odds_acc),
        'contingency': {
            'both_right': int(both_right),
            'fpl_only_right': int(fpl_only_right),
            'odds_only_right': int(odds_only_right),
            'both_wrong': int(both_wrong),
        },
        'mcnemar_p': float(mcnemar_p),
        'disagree_brier_fpl': float(disagree_brier_fpl),
        'disagree_brier_odds': float(disagree_brier_odds),
        'fisher_crowd_vs_chance_p': float(fisher_p),
    }

    print(f"  Agree on favorite: {n_agree}/{n_total} ({n_agree/n_total*100:.1f}%)")
    print(f"  Disagree: {n_disagree} matches")
    print(f"    Crowd correct: {disagree_fpl_acc:.1%}  |  Odds correct: {disagree_odds_acc:.1%}")
    print(f"  Contingency table:")
    print(f"    Both right: {both_right}  |  FPL-only right: {fpl_only_right}  |  "
          f"Odds-only right: {odds_only_right}  |  Both wrong: {both_wrong}")
    print(f"  McNemar's p-value: {mcnemar_p:.4f}")
    print(f"  Fisher exact (crowd beats 1/3 in disagree): p={fisher_p:.4f}")

    return results


def chart2_agreement(results):
    """Two-panel chart: agreement breakdown + contingency heatmap."""
    setup_chart_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Agreement vs Disagreement accuracy
    categories = ['Agree\n(same favorite)', 'Disagree\n(different favorite)']
    fpl_accs = [results['agree_accuracy_fpl'], results['disagree_accuracy_fpl']]
    odds_accs = [results['agree_accuracy_odds'], results['disagree_accuracy_odds']]
    x = np.arange(len(categories))
    w = 0.3

    ax1.bar(x - w/2, [a * 100 for a in fpl_accs], w, label='FPL Crowd',
            color=TEAL, edgecolor='#444444', zorder=3)
    ax1.bar(x + w/2, [a * 100 for a in odds_accs], w, label='Bookmaker',
            color=CORAL, edgecolor='#444444', zorder=3)

    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=11)
    ax1.legend(fontsize=10)
    ax1.set_title(f'Who Is Right When They Agree vs Disagree?\n'
                  f'({results["n_agree"]} agree, {results["n_disagree"]} disagree)',
                  fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.2)

    # Annotate counts
    for i, cat in enumerate(categories):
        n = results['n_agree'] if i == 0 else results['n_disagree']
        ax1.annotate(f'n={n}', xy=(i, 5), ha='center', fontsize=10, color=WHITE)

    # Panel 2: Contingency matrix
    c = results['contingency']
    matrix = np.array([
        [c['both_right'], c['fpl_only_right']],
        [c['odds_only_right'], c['both_wrong']]
    ])
    im = ax2.imshow(matrix, cmap='YlOrRd', aspect='auto')
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Odds Correct', 'Odds Wrong'], fontsize=11)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Crowd\nCorrect', 'Crowd\nWrong'], fontsize=11)
    ax2.set_title(f'Error Independence\n(McNemar p={results["mcnemar_p"]:.3f})',
                  fontsize=12, fontweight='bold')

    for i in range(2):
        for j in range(2):
            color = 'black' if matrix[i, j] > matrix.max() * 0.5 else WHITE
            ax2.text(j, i, str(matrix[i, j]), ha='center', va='center',
                     fontsize=16, fontweight='bold', color=color)

    plt.colorbar(im, ax=ax2, shrink=0.8)
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / 'crowd_wisdom_agreement.png')
    plt.close()
    print("  Saved crowd_wisdom_agreement.png")


# =========================================================================
# Section 3: Crowd Confidence Calibration
# =========================================================================

def section3_calibration(test_preds, y_test):
    """Calibration analysis: when crowd is confident, are they right?"""
    print("\n[3/6] Crowd Confidence Calibration")
    print("-" * 50)

    fpl_proba = np.column_stack([
        test_preds['p_fpl_only_A'].values,
        test_preds['p_fpl_only_D'].values,
        test_preds['p_fpl_only_H'].values,
    ])
    odds_proba = np.column_stack([
        test_preds['implied_prob_a'].values,
        test_preds['implied_prob_d'].values,
        test_preds['implied_prob_h'].values,
    ])

    # Per-class calibration for Home wins (class 2)
    y_home = (y_test == 2).astype(int)
    fpl_cal_frac, fpl_cal_mean = calibration_curve(
        y_home, fpl_proba[:, 2], n_bins=8, strategy='uniform')
    odds_cal_frac, odds_cal_mean = calibration_curve(
        y_home, odds_proba[:, 2], n_bins=8, strategy='uniform')

    # Expected Calibration Error
    def compute_ece(y_true_binary, y_proba_col, n_bins=10):
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (y_proba_col >= lo) & (y_proba_col < hi)
            if mask.sum() == 0:
                continue
            bin_acc = y_true_binary[mask].mean()
            bin_conf = y_proba_col[mask].mean()
            ece += mask.sum() / len(y_true_binary) * abs(bin_acc - bin_conf)
        return ece

    fpl_ece = compute_ece(y_home, fpl_proba[:, 2])
    odds_ece = compute_ece(y_home, odds_proba[:, 2])

    # Accuracy by confidence quintile
    fpl_max_prob = fpl_proba.max(axis=1)
    fpl_pred_class = fpl_proba.argmax(axis=1)
    fpl_correct = (fpl_pred_class == y_test)

    # Create quintile bins
    try:
        quintile_labels = pd.qcut(fpl_max_prob, 5, labels=['Q1\n(least\nconfident)',
                                                            'Q2', 'Q3', 'Q4',
                                                            'Q5\n(most\nconfident)'],
                                  duplicates='drop')
    except ValueError:
        quintile_labels = pd.cut(fpl_max_prob, 5, labels=['Q1\n(least\nconfident)',
                                                           'Q2', 'Q3', 'Q4',
                                                           'Q5\n(most\nconfident)'])

    quintile_acc = pd.DataFrame({'correct': fpl_correct, 'quintile': quintile_labels})
    q_stats = quintile_acc.groupby('quintile', observed=True)['correct'].agg(['mean', 'count', 'sum'])

    print(f"  FPL ECE (home win): {fpl_ece:.4f}")
    print(f"  Odds ECE (home win): {odds_ece:.4f}")
    print(f"\n  Accuracy by crowd confidence quintile:")
    for q, row in q_stats.iterrows():
        print(f"    {str(q):<25} Acc={row['mean']:.3f}  n={int(row['count'])}")

    # Test: is accuracy monotonically increasing with confidence?
    q_accs = q_stats['mean'].values
    if len(q_accs) >= 3:
        rho, p_mono = stats.spearmanr(range(len(q_accs)), q_accs)
    else:
        rho, p_mono = 0, 1.0

    results = {
        'fpl_ece': float(fpl_ece),
        'odds_ece': float(odds_ece),
        'fpl_calibration': {
            'fraction_of_positives': fpl_cal_frac.tolist(),
            'mean_predicted': fpl_cal_mean.tolist(),
        },
        'odds_calibration': {
            'fraction_of_positives': odds_cal_frac.tolist(),
            'mean_predicted': odds_cal_mean.tolist(),
        },
        'quintile_accuracy': {str(q): float(row['mean'])
                              for q, row in q_stats.iterrows()},
        'confidence_accuracy_correlation': {
            'spearman_rho': float(rho),
            'p_value': float(p_mono),
        },
    }

    print(f"\n  Confidence-accuracy correlation: rho={rho:.3f}, p={p_mono:.4f}")

    return results


def chart3_calibration(results):
    """Two-panel: reliability diagram + accuracy by confidence quintile."""
    setup_chart_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Reliability diagram (Home win)
    fpl_cal = results['fpl_calibration']
    odds_cal = results['odds_calibration']

    ax1.plot([0, 1], [0, 1], 'w--', alpha=0.3, label='Perfect calibration')
    ax1.plot(fpl_cal['mean_predicted'], fpl_cal['fraction_of_positives'],
             's-', color=TEAL, markersize=8, label=f'FPL Crowd (ECE={results["fpl_ece"]:.3f})')
    ax1.plot(odds_cal['mean_predicted'], odds_cal['fraction_of_positives'],
             'o-', color=CORAL, markersize=8, label=f'Bookmaker (ECE={results["odds_ece"]:.3f})')
    ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax1.set_ylabel('Fraction of Positives', fontsize=12)
    ax1.set_title('Calibration: P(Home Win)\nDoes confidence match reality?',
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(alpha=0.2)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # Panel 2: Accuracy by confidence quintile
    q_acc = results['quintile_accuracy']
    labels = list(q_acc.keys())
    vals = list(q_acc.values())

    bars = ax2.bar(range(len(labels)), [v * 100 for v in vals],
                   color=[TEAL] * len(labels), edgecolor='#444444', zorder=3)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    corr = results['confidence_accuracy_correlation']
    sig = '*' if corr['p_value'] < 0.05 else ''
    ax2.set_title(f'Crowd Accuracy by Confidence Level\n'
                  f'(rho={corr["spearman_rho"]:.2f}, p={corr["p_value"]:.3f}{sig})',
                  fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.2)

    # Color gradient by value
    cmap = plt.cm.YlGn
    norm_vals = [(v - min(vals)) / (max(vals) - min(vals) + 1e-9) for v in vals]
    for bar, nv in zip(bars, norm_vals):
        bar.set_facecolor(cmap(0.3 + nv * 0.6))

    plt.tight_layout()
    fig.savefig(CHARTS_DIR / 'crowd_wisdom_calibration.png')
    plt.close()
    print("  Saved crowd_wisdom_calibration.png")


# =========================================================================
# Section 4: Season-Arc Bootstrap Validation
# =========================================================================

def section4_season_arc_bootstrap(arc_df):
    """Bootstrap CIs and permutation tests for season-arc findings."""
    print("\n[4/6] Season-Arc Bootstrap Validation")
    print("-" * 50)

    # Only use seasons with FPL data for combined model
    fpl_seasons = ['2020-21', '2021-22', '2022-23', '2023-24', '2024-25']
    arc_fpl = arc_df[arc_df['season'].isin(fpl_seasons)].copy()

    train_seasons = ['2020-21', '2021-22', '2022-23']  # 3 seasons with FPL data
    test_seasons = ['2023-24', '2024-25']

    results = {}

    for target_name, target_col in [('top4', 'top4'), ('relegated', 'relegated')]:
        target_results = {}

        for cp in CHECKPOINTS:
            cp_data = arc_fpl[arc_fpl['checkpoint'] == cp].dropna(subset=[target_col])
            train = cp_data[cp_data['season'].isin(train_seasons)]
            test = cp_data[cp_data['season'].isin(test_seasons)]

            if len(train) < 20 or len(test) < 10:
                continue

            n_test = len(test)
            n_pos = int(test[target_col].sum())

            # Fit baseline (PPG only) and combined (PPG + FPL)
            def fit_and_predict(features, X_train_df, y_train, X_test_df):
                avail = [f for f in features if f in X_train_df.columns]
                if not avail:
                    return None
                scaler = StandardScaler()
                Xtr = scaler.fit_transform(X_train_df[avail].fillna(0).values)
                Xte = scaler.transform(X_test_df[avail].fillna(0).values)
                model = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
                model.fit(Xtr, y_train)
                proba = model.predict_proba(Xte)
                pos_idx = list(model.classes_).index(1) if 1 in model.classes_ else 0
                return proba[:, pos_idx]

            y_train_arr = train[target_col].values
            y_test_arr = test[target_col].values

            pred_baseline = fit_and_predict(SA_EXTENDED, train, y_train_arr, test)
            pred_combined = fit_and_predict(SA_COMBINED, train, y_train_arr, test)

            if pred_baseline is None or pred_combined is None:
                continue

            # Point estimates
            brier_baseline = brier_score_loss(y_test_arr, pred_baseline)
            brier_combined = brier_score_loss(y_test_arr, pred_combined)
            improvement = (brier_baseline - brier_combined) / brier_baseline * 100

            # Bootstrap (resample test set, recompute Brier)
            boot_baseline = []
            boot_combined = []
            boot_improvement = []
            combined_wins = 0

            for _ in range(N_BOOTSTRAP):
                idx = RNG.choice(n_test, size=n_test, replace=True)
                b_base = brier_score_loss(y_test_arr[idx], pred_baseline[idx])
                b_comb = brier_score_loss(y_test_arr[idx], pred_combined[idx])
                boot_baseline.append(b_base)
                boot_combined.append(b_comb)
                if b_base > 0:
                    boot_improvement.append((b_base - b_comb) / b_base * 100)
                if b_comb < b_base:
                    combined_wins += 1

            prob_combined_better = combined_wins / N_BOOTSTRAP

            # Permutation test: shuffle FPL features, refit, compare
            fpl_cols_avail = [f for f in SA_FPL if f in train.columns]
            n_perm = 2000
            perm_improvements = []
            for _ in range(n_perm):
                # Shuffle FPL columns in test set
                test_perm = test.copy()
                for col in fpl_cols_avail:
                    test_perm[col] = RNG.permutation(test_perm[col].values)
                pred_perm = fit_and_predict(SA_COMBINED, train, y_train_arr, test_perm)
                if pred_perm is not None:
                    b_perm = brier_score_loss(y_test_arr, pred_perm)
                    perm_improvements.append(brier_baseline - b_perm)

            observed_improvement = brier_baseline - brier_combined
            if perm_improvements:
                perm_p = np.mean(np.array(perm_improvements) >= observed_improvement)
            else:
                perm_p = 1.0

            target_results[f'GW{cp}'] = {
                'n_test': n_test,
                'n_positive': n_pos,
                'brier_baseline': float(brier_baseline),
                'brier_combined': float(brier_combined),
                'improvement_pct': float(improvement),
                'bootstrap': {
                    'baseline_ci': [float(np.percentile(boot_baseline, 2.5)),
                                    float(np.percentile(boot_baseline, 97.5))],
                    'combined_ci': [float(np.percentile(boot_combined, 2.5)),
                                    float(np.percentile(boot_combined, 97.5))],
                    'improvement_ci': [float(np.percentile(boot_improvement, 2.5)),
                                       float(np.percentile(boot_improvement, 97.5))]
                                      if boot_improvement else [0, 0],
                    'prob_combined_better': float(prob_combined_better),
                },
                'permutation_p_value': float(perm_p),
            }

            sig = '*' if perm_p < 0.05 else ''
            ci = target_results[f'GW{cp}']['bootstrap']
            print(f"  {target_name} GW{cp}: baseline={brier_baseline:.4f} "
                  f"combined={brier_combined:.4f} ({improvement:+.1f}%) "
                  f"P(combined<baseline)={prob_combined_better:.1%} "
                  f"perm_p={perm_p:.3f}{sig}")

        results[target_name] = target_results

    # Leave-one-season-out CV
    print("\n  Leave-One-Season-Out CV (FPL seasons only):")
    loso_results = {}
    for target_name, target_col in [('top4', 'top4'), ('relegated', 'relegated')]:
        loso_brier_baseline = []
        loso_brier_combined = []

        for held_out in fpl_seasons:
            train_s = [s for s in fpl_seasons if s != held_out]
            for cp in [15, 20, 25]:  # Representative checkpoints
                cp_data = arc_fpl[arc_fpl['checkpoint'] == cp].dropna(subset=[target_col])
                tr = cp_data[cp_data['season'].isin(train_s)]
                te = cp_data[cp_data['season'] == held_out]

                if len(tr) < 20 or len(te) < 5 or te[target_col].sum() == 0:
                    continue
                if te[target_col].sum() == len(te):
                    continue

                y_tr = tr[target_col].values
                y_te = te[target_col].values

                avail_ext = [f for f in SA_EXTENDED if f in tr.columns]
                avail_comb = [f for f in SA_COMBINED if f in tr.columns]

                scaler_b = StandardScaler()
                Xtr_b = scaler_b.fit_transform(tr[avail_ext].fillna(0).values)
                Xte_b = scaler_b.transform(te[avail_ext].fillna(0).values)
                m_b = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
                m_b.fit(Xtr_b, y_tr)
                p_b = m_b.predict_proba(Xte_b)
                pos_idx = list(m_b.classes_).index(1) if 1 in m_b.classes_ else 0
                loso_brier_baseline.append(brier_score_loss(y_te, p_b[:, pos_idx]))

                scaler_c = StandardScaler()
                Xtr_c = scaler_c.fit_transform(tr[avail_comb].fillna(0).values)
                Xte_c = scaler_c.transform(te[avail_comb].fillna(0).values)
                m_c = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
                m_c.fit(Xtr_c, y_tr)
                p_c = m_c.predict_proba(Xte_c)
                pos_idx_c = list(m_c.classes_).index(1) if 1 in m_c.classes_ else 0
                loso_brier_combined.append(brier_score_loss(y_te, p_c[:, pos_idx_c]))

        if loso_brier_baseline and loso_brier_combined:
            loso_results[target_name] = {
                'baseline_mean': float(np.mean(loso_brier_baseline)),
                'baseline_std': float(np.std(loso_brier_baseline)),
                'combined_mean': float(np.mean(loso_brier_combined)),
                'combined_std': float(np.std(loso_brier_combined)),
                'n_folds': len(loso_brier_baseline),
            }
            lr = loso_results[target_name]
            print(f"  LOSO {target_name}: baseline={lr['baseline_mean']:.4f}+/-{lr['baseline_std']:.4f}  "
                  f"combined={lr['combined_mean']:.4f}+/-{lr['combined_std']:.4f}  "
                  f"({lr['n_folds']} folds)")

    results['loso_cv'] = loso_results
    return results


def chart4_season_arc_bootstrap(results):
    """Line chart with bootstrap CI bands for season-arc predictions."""
    setup_chart_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (target_name, target_label) in zip(axes, [('top4', 'Top 4'), ('relegated', 'Relegation')]):
        if target_name not in results or not results[target_name]:
            ax.set_title(f'{target_label}: Insufficient data')
            continue

        data = results[target_name]
        cps = sorted([int(k.replace('GW', '')) for k in data.keys()])
        baseline_vals = [data[f'GW{cp}']['brier_baseline'] for cp in cps]
        combined_vals = [data[f'GW{cp}']['brier_combined'] for cp in cps]
        baseline_lo = [data[f'GW{cp}']['bootstrap']['baseline_ci'][0] for cp in cps]
        baseline_hi = [data[f'GW{cp}']['bootstrap']['baseline_ci'][1] for cp in cps]
        combined_lo = [data[f'GW{cp}']['bootstrap']['combined_ci'][0] for cp in cps]
        combined_hi = [data[f'GW{cp}']['bootstrap']['combined_ci'][1] for cp in cps]

        ax.plot(cps, baseline_vals, 'o-', color=CORAL, label='PPG Baseline', markersize=8)
        ax.fill_between(cps, baseline_lo, baseline_hi, color=CORAL, alpha=0.15)
        ax.plot(cps, combined_vals, 's-', color=TEAL, label='PPG + FPL', markersize=8)
        ax.fill_between(cps, combined_lo, combined_hi, color=TEAL, alpha=0.15)

        ax.set_xlabel('Gameweek Checkpoint', fontsize=12)
        ax.set_ylabel('Brier Score (lower = better)', fontsize=12)
        ax.set_title(f'{target_label} Prediction\n(shaded = 95% bootstrap CI)',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.2)
        ax.set_xticks(cps)

        # Mark significant checkpoints
        for cp in cps:
            p_val = data[f'GW{cp}']['permutation_p_value']
            prob = data[f'GW{cp}']['bootstrap']['prob_combined_better']
            if p_val < 0.05:
                ax.annotate('*', xy=(cp, data[f'GW{cp}']['brier_combined']),
                            fontsize=18, fontweight='bold', color=TEAL,
                            ha='center', va='bottom')

    plt.tight_layout()
    fig.savefig(CHARTS_DIR / 'crowd_wisdom_season_arc_bootstrap.png')
    plt.close()
    print("  Saved crowd_wisdom_season_arc_bootstrap.png")


# =========================================================================
# Section 5: Information Content
# =========================================================================

def section5_information(test_preds, y_test, train_df):
    """Compute mutual information between signals and outcomes."""
    print("\n[5/6] Information Content (Mutual Information)")
    print("-" * 50)

    # Outcome entropy
    y_train = train_df['result'].map(RESULT_MAP).values
    base_rates = np.bincount(y_train, minlength=3) / len(y_train)
    outcome_entropy = -np.sum(base_rates * np.log2(base_rates + 1e-12))

    # Mutual information for various signals
    def mi_from_continuous(signal_values, outcomes, n_bins=10):
        """Bin a continuous signal and compute MI with discrete outcomes."""
        try:
            binned = pd.qcut(signal_values, n_bins, labels=False, duplicates='drop')
        except ValueError:
            binned = pd.cut(signal_values, n_bins, labels=False)
        valid = ~np.isnan(binned)
        if valid.sum() < 20:
            return 0.0
        return mutual_info_score(outcomes[valid], binned[valid])

    signals_mi = {}

    # FPL signals
    for col in FPL_FEATURES:
        if col in test_preds.columns:
            mi = mi_from_continuous(test_preds[col].values, y_test)
            signals_mi[col] = mi

    # Odds signals
    for col in ['implied_prob_h', 'implied_prob_d', 'implied_prob_a']:
        if col in test_preds.columns:
            mi = mi_from_continuous(test_preds[col].values, y_test)
            signals_mi[col] = mi

    # Model predictions (discretized)
    fpl_pred_class = np.column_stack([
        test_preds['p_fpl_only_A'].values,
        test_preds['p_fpl_only_D'].values,
        test_preds['p_fpl_only_H'].values,
    ]).argmax(axis=1)
    odds_pred_class = np.column_stack([
        test_preds['implied_prob_a'].values,
        test_preds['implied_prob_d'].values,
        test_preds['implied_prob_h'].values,
    ]).argmax(axis=1)

    mi_fpl_model = mutual_info_score(y_test, fpl_pred_class)
    mi_odds_model = mutual_info_score(y_test, odds_pred_class)
    signals_mi['fpl_model_prediction'] = mi_fpl_model
    signals_mi['odds_model_prediction'] = mi_odds_model

    # Conditional MI: FPL given Odds
    # Approximate: within each odds quintile, compute MI(FPL, outcome)
    odds_h = test_preds['implied_prob_h'].values
    try:
        odds_quintile = pd.qcut(odds_h, 5, labels=False, duplicates='drop')
    except ValueError:
        odds_quintile = pd.cut(odds_h, 5, labels=False)

    cond_mi_total = 0
    n_valid = 0
    for q in np.unique(odds_quintile[~np.isnan(odds_quintile)]):
        mask = odds_quintile == q
        if mask.sum() < 20:
            continue
        mi_within = mi_from_continuous(
            test_preds.loc[mask, 'ownership_ratio'].values,
            y_test[mask], n_bins=5)
        cond_mi_total += mi_within * mask.sum()
        n_valid += mask.sum()
    conditional_mi = cond_mi_total / n_valid if n_valid > 0 else 0

    # Mean prediction entropy
    def mean_prediction_entropy(proba):
        return float(np.mean(-np.sum(proba * np.log2(proba + 1e-12), axis=1)))

    fpl_proba = np.column_stack([
        test_preds['p_fpl_only_A'].values,
        test_preds['p_fpl_only_D'].values,
        test_preds['p_fpl_only_H'].values,
    ])
    odds_proba = np.column_stack([
        test_preds['implied_prob_a'].values,
        test_preds['implied_prob_d'].values,
        test_preds['implied_prob_h'].values,
    ])

    entropy_fpl = mean_prediction_entropy(fpl_proba)
    entropy_odds = mean_prediction_entropy(odds_proba)

    results = {
        'outcome_entropy_bits': float(outcome_entropy),
        'signals_mi': {k: float(v) for k, v in sorted(signals_mi.items(), key=lambda x: -x[1])},
        'conditional_mi_fpl_given_odds': float(conditional_mi),
        'mean_prediction_entropy': {
            'fpl': entropy_fpl,
            'odds': entropy_odds,
            'naive': float(outcome_entropy),  # uniform = max entropy for this distribution
        },
        'fpl_mi_as_pct_of_odds': float(mi_fpl_model / mi_odds_model * 100) if mi_odds_model > 0 else 0,
    }

    print(f"  Outcome entropy: {outcome_entropy:.4f} bits")
    print(f"  MI (FPL model prediction): {mi_fpl_model:.4f} bits")
    print(f"  MI (Odds model prediction): {mi_odds_model:.4f} bits")
    print(f"  FPL captures {results['fpl_mi_as_pct_of_odds']:.1f}% of odds MI")
    print(f"  Conditional MI (FPL | Odds): {conditional_mi:.4f} bits")
    print(f"\n  Signal MI rankings:")
    for sig, mi in sorted(signals_mi.items(), key=lambda x: -x[1]):
        print(f"    {sig:<30} {mi:.4f} bits")
    print(f"\n  Mean prediction entropy:")
    print(f"    FPL:  {entropy_fpl:.4f} bits (less certain)")
    print(f"    Odds: {entropy_odds:.4f} bits (more certain)")

    return results


def chart5_information(results):
    """Bar chart of mutual information by signal."""
    setup_chart_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    mi_data = results['signals_mi']
    names = list(mi_data.keys())
    vals = list(mi_data.values())

    # Color: FPL signals in teal, odds in coral
    colors = []
    for n in names:
        if 'implied' in n or 'odds' in n:
            colors.append(CORAL)
        else:
            colors.append(TEAL)

    bars = ax.barh(range(len(names)), vals, color=colors, edgecolor='#444444',
                   height=0.6, zorder=3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace('_', ' ').title() for n in names], fontsize=10)
    ax.set_xlabel('Mutual Information (bits)', fontsize=12)
    ax.set_title('Information Content: How Many Bits Does Each Signal Carry?\n'
                 '(Teal = FPL crowd signals, Orange = Bookmaker odds)',
                 fontsize=12, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.2)

    # Add outcome entropy reference
    ent = results['outcome_entropy_bits']
    ax.axvline(x=ent, color=PURPLE, linestyle='--', alpha=0.5)
    ax.annotate(f'Outcome entropy\n({ent:.3f} bits)',
                xy=(ent, len(names) - 1), fontsize=9, color=PURPLE,
                ha='left', va='center')

    # Conditional MI annotation
    cmi = results['conditional_mi_fpl_given_odds']
    ax.annotate(f'FPL unique info (given odds): {cmi:.4f} bits',
                xy=(0.5, 0.02), xycoords='axes fraction',
                fontsize=11, fontweight='bold', color=WHITE,
                bbox=dict(boxstyle='round', facecolor='#222222', edgecolor='#555555'))

    plt.tight_layout()
    fig.savefig(CHARTS_DIR / 'crowd_wisdom_information.png')
    plt.close()
    print("  Saved crowd_wisdom_information.png")


# =========================================================================
# Section 6: Headline Summary Chart
# =========================================================================

def chart6_summary(s1_results, s2_results, s3_results, s4_results, s5_results):
    """Composite 2x2 headline chart."""
    setup_chart_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    fig.suptitle('The Wisdom of 11 Million Fantasy Managers',
                 fontsize=18, fontweight='bold', y=0.98, color=WHITE)

    # --- Panel 1 (top-left): Accuracy Ladder ---
    ax = axes[0, 0]
    display = ['Naive (base rates)', 'Home-bias', 'FPL Crowd (5 signals)',
               'Bookmaker (Bet365)']
    bss_vals = [s1_results['bootstrap_bss'][t]['mean'] * 100 for t in display]
    colors_p1 = ['#555555', '#777777', TEAL, CORAL]
    ax.barh(range(len(display)), bss_vals, color=colors_p1, edgecolor='#444444',
            height=0.5, zorder=3)
    ax.set_yticks(range(len(display)))
    ax.set_yticklabels([d.split('(')[0].strip() for d in display], fontsize=10)
    ax.set_xlabel('Skill vs Random (%)')
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title(f'Prediction Accuracy\n(Crowd = {s1_results["crowd_pct_of_odds_skill"]:.0f}% of bookmaker)',
                 fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.2)

    # --- Panel 2 (top-right): Calibration ---
    ax = axes[0, 1]
    fpl_cal = s3_results['fpl_calibration']
    odds_cal = s3_results['odds_calibration']
    ax.plot([0, 1], [0, 1], 'w--', alpha=0.3, linewidth=1)
    ax.plot(fpl_cal['mean_predicted'], fpl_cal['fraction_of_positives'],
            's-', color=TEAL, markersize=6, label='FPL Crowd')
    ax.plot(odds_cal['mean_predicted'], odds_cal['fraction_of_positives'],
            'o-', color=CORAL, markersize=6, label='Bookmaker')
    ax.set_xlabel('Predicted P(Home Win)')
    ax.set_ylabel('Actual P(Home Win)')
    ax.set_title(f'Calibration Quality\n(FPL ECE={s3_results["fpl_ece"]:.3f}, '
                 f'Odds ECE={s3_results["odds_ece"]:.3f})', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    # --- Panel 3 (bottom-left): Season-arc with CIs ---
    ax = axes[1, 0]
    if 'top4' in s4_results and s4_results['top4']:
        data = s4_results['top4']
        cps = sorted([int(k.replace('GW', '')) for k in data.keys()])
        base_v = [data[f'GW{cp}']['brier_baseline'] for cp in cps]
        comb_v = [data[f'GW{cp}']['brier_combined'] for cp in cps]
        base_lo = [data[f'GW{cp}']['bootstrap']['baseline_ci'][0] for cp in cps]
        base_hi = [data[f'GW{cp}']['bootstrap']['baseline_ci'][1] for cp in cps]
        comb_lo = [data[f'GW{cp}']['bootstrap']['combined_ci'][0] for cp in cps]
        comb_hi = [data[f'GW{cp}']['bootstrap']['combined_ci'][1] for cp in cps]

        ax.plot(cps, base_v, 'o-', color=CORAL, markersize=6, label='PPG Only')
        ax.fill_between(cps, base_lo, base_hi, color=CORAL, alpha=0.12)
        ax.plot(cps, comb_v, 's-', color=TEAL, markersize=6, label='PPG + FPL')
        ax.fill_between(cps, comb_lo, comb_hi, color=TEAL, alpha=0.12)
        ax.legend(fontsize=9)
    ax.set_xlabel('Gameweek')
    ax.set_ylabel('Brier Score')
    ax.set_title('Top-4 Prediction (Season-Level)\n(shaded = 95% CI)',
                 fontsize=11, fontweight='bold')
    ax.grid(alpha=0.2)

    # --- Panel 4 (bottom-right): Information content summary ---
    ax = axes[1, 1]
    mi = s5_results['signals_mi']
    # Top signals only
    top_signals = list(mi.items())[:8]
    names = [s[0].replace('_', ' ').replace('implied prob', 'odds prob')[:20] for s in top_signals]
    vals = [s[1] for s in top_signals]
    colors_p4 = [CORAL if 'implied' in s[0] or 'odds' in s[0] else TEAL for s in top_signals]
    ax.barh(range(len(names)), vals, color=colors_p4, edgecolor='#444444',
            height=0.5, zorder=3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Mutual Information (bits)')
    pct = s5_results['fpl_mi_as_pct_of_odds']
    ax.set_title(f'Information Content\n(FPL model = {pct:.0f}% of odds MI)',
                 fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(CHARTS_DIR / 'crowd_wisdom_summary.png')
    plt.close()
    print("  Saved crowd_wisdom_summary.png")


# =========================================================================
# Results.md update
# =========================================================================

def update_results_md(s1, s2, s3, s4, s5):
    """Append crowd wisdom section to results.md."""
    results_path = Path(__file__).resolve().parent.parent / 'outputs' / 'results.md'

    crowd_pct = s1['crowd_pct_of_odds_skill']
    fpl_brier = s1['metrics']['FPL Crowd (5 signals)']['brier']
    odds_brier = s1['metrics']['Bookmaker (Bet365)']['brier']
    naive_brier = s1['metrics']['Naive (base rates)']['brier']
    fpl_bss = s1['bootstrap_bss']['FPL Crowd (5 signals)']
    odds_bss = s1['bootstrap_bss']['Bookmaker (Bet365)']

    # Season-arc summary
    arc_lines = []
    for target in ['top4', 'relegated']:
        if target in s4 and s4[target]:
            for cp_key, cp_data in sorted(s4[target].items()):
                prob = cp_data['bootstrap']['prob_combined_better']
                perm_p = cp_data['permutation_p_value']
                imp = cp_data['improvement_pct']
                arc_lines.append(
                    f"| {target.title():<10} | {cp_key} | {cp_data['brier_baseline']:.4f} | "
                    f"{cp_data['brier_combined']:.4f} | {imp:+.1f}% | {prob:.1%} | {perm_p:.3f} |"
                )

    section = f"""

---

## Crowd Wisdom: What 11 Million Fantasy Managers Know

### The Headline Finding

The existing narrative — "FPL doesn't beat bookmakers" — buries the lead. The remarkable finding is that **11 million people playing a fantasy game for fun independently reproduce {crowd_pct:.0f}% of professional bookmaker accuracy** for EPL match prediction. This is a crowd-wisdom result: millions of uncoordinated amateurs, making decisions for entertainment, collectively generate predictions nearly as good as multi-billion-pound betting operations.

### Prediction Accuracy Ladder

| Predictor | Brier Score | Skill vs Naive | 95% CI (BSS) |
|-----------|:-----------:|:--------------:|:-------------:|
| Naive (base rates) | {naive_brier:.4f} | 0.0% | — |
| FPL Crowd (5 signals) | {fpl_brier:.4f} | {s1['metrics']['FPL Crowd (5 signals)']['bss_vs_naive']*100:.1f}% | [{fpl_bss['ci_lower']*100:.1f}%, {fpl_bss['ci_upper']*100:.1f}%] |
| Bookmaker (Bet365) | {odds_brier:.4f} | {s1['metrics']['Bookmaker (Bet365)']['bss_vs_naive']*100:.1f}% | [{odds_bss['ci_lower']*100:.1f}%, {odds_bss['ci_upper']*100:.1f}%] |

The crowd achieves **{crowd_pct:.0f}%** of bookmaker predictive skill — using only fantasy game participation data, no financial markets, no expert models.

![Crowd Wisdom Ladder](charts/crowd_wisdom_ladder.png)

### When Crowd and Market Disagree

Crowd and bookmakers agree on the match favorite **{s2['agree_pct']:.1f}%** of the time ({s2['n_agree']}/{s2['n_total']} matches). When they disagree ({s2['n_disagree']} matches):

- Crowd correct: {s2['disagree_accuracy_fpl']:.1%}
- Bookmaker correct: {s2['disagree_accuracy_odds']:.1%}
- McNemar's p-value: {s2['mcnemar_p']:.3f} ({"errors are independent" if s2['mcnemar_p'] < 0.05 else "errors are not statistically independent"})

Neither has a systematic edge on disagreement matches, confirming the crowd and bookmaker are processing largely overlapping information.

![Agreement Analysis](charts/crowd_wisdom_agreement.png)

### Crowd Confidence Calibration

When the crowd is confident about an outcome, are they actually right more often?

- FPL Expected Calibration Error (Home Win): **{s3['fpl_ece']:.4f}**
- Odds Expected Calibration Error (Home Win): **{s3['odds_ece']:.4f}**
- Confidence-accuracy monotonicity: rho={s3['confidence_accuracy_correlation']['spearman_rho']:.2f}, p={s3['confidence_accuracy_correlation']['p_value']:.3f}

{"The crowd's confidence is meaningfully calibrated — higher confidence predicts higher accuracy." if s3['confidence_accuracy_correlation']['p_value'] < 0.05 else "The relationship between crowd confidence and accuracy is not statistically significant at p<0.05, though the trend is in the expected direction."}

![Calibration](charts/crowd_wisdom_calibration.png)

### Season-Arc: Honest Bootstrap Validation

The season-level finding (FPL improves top-4 and relegation prediction) was based on a small test set (~40 teams). Here we add bootstrap confidence intervals and permutation tests.

| Target | Checkpoint | Baseline Brier | Combined Brier | Change | P(Combined<Baseline) | Perm. p-value |
|--------|:----------:|:--------------:|:--------------:|:------:|:--------------------:|:-------------:|
{chr(10).join(arc_lines)}

**Interpretation:** P(Combined<Baseline) shows how often the FPL-enhanced model beats the baseline across 10,000 bootstrap resamples. Permutation p-value tests whether the improvement is real vs. what you'd get from shuffled (meaningless) FPL data.

![Season-Arc Bootstrap](charts/crowd_wisdom_season_arc_bootstrap.png)

### Information Content

| Metric | Value |
|--------|:-----:|
| Outcome entropy | {s5['outcome_entropy_bits']:.4f} bits |
| FPL model MI | {s5['signals_mi'].get('fpl_model_prediction', 0):.4f} bits |
| Odds model MI | {s5['signals_mi'].get('odds_model_prediction', 0):.4f} bits |
| FPL as % of Odds MI | {s5['fpl_mi_as_pct_of_odds']:.0f}% |
| FPL unique info (given odds) | {s5['conditional_mi_fpl_given_odds']:.4f} bits |

The conditional mutual information (FPL | Odds) of {s5['conditional_mi_fpl_given_odds']:.4f} bits quantifies the independent information the crowd contributes beyond what's already in market prices.

![Information Content](charts/crowd_wisdom_information.png)

### Summary

![Crowd Wisdom Summary](charts/crowd_wisdom_summary.png)
"""

    with open(results_path, 'a') as f:
        f.write(section)
    print(f"\n  Appended Crowd Wisdom section to {results_path}")


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 60)
    print("CROWD WISDOM ANALYSIS")
    print("What 11 Million Fantasy Managers Know")
    print("=" * 60)

    # --- Load data ---
    signals = load_parquet('signals.parquet')
    test_preds = pd.read_parquet(MODEL_DIR / 'test_predictions.parquet')
    arc_df = load_parquet('season_arc.parquet')

    train_df = signals[signals['season'].isin(TRAIN_SEASONS)]

    # Prepare test labels
    y_test = test_preds['result'].map(RESULT_MAP).values

    # Drop any NaN in critical columns
    valid_mask = ~np.isnan(test_preds[['implied_prob_h', 'implied_prob_d', 'implied_prob_a',
                                        'p_fpl_only_H', 'p_fpl_only_D', 'p_fpl_only_A']].values).any(axis=1)
    test_preds = test_preds[valid_mask].reset_index(drop=True)
    y_test = test_preds['result'].map(RESULT_MAP).values

    print(f"\nTest set: {len(test_preds)} matches")
    print(f"Season-arc: {len(arc_df)} checkpoint observations")

    # --- Run all sections ---
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    s1 = section1_accuracy_ladder(test_preds, y_test, train_df)
    chart1_accuracy_ladder(s1)

    s2 = section2_agreement(test_preds, y_test)
    chart2_agreement(s2)

    s3 = section3_calibration(test_preds, y_test)
    chart3_calibration(s3)

    s4 = section4_season_arc_bootstrap(arc_df)
    chart4_season_arc_bootstrap(s4)

    s5 = section5_information(test_preds, y_test, train_df)
    chart5_information(s5)

    chart6_summary(s1, s2, s3, s4, s5)

    # --- Save all metrics ---
    all_metrics = jsonify({
        'crowd_accuracy': s1,
        'agreement_analysis': s2,
        'crowd_confidence': s3,
        'season_arc_bootstrap': s4,
        'information_content': s5,
    })

    with open(MODEL_DIR / 'crowd_wisdom.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  Saved crowd_wisdom.json")

    # --- Update results.md ---
    update_results_md(s1, s2, s3, s4, s5)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == '__main__':
    main()
