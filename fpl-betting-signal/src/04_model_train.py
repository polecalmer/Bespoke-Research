"""Train three logistic regression models and evaluate on validation set.

Models:
1. Odds-Only: implied_prob_h/d/a
2. FPL-Only: transfer_ratio, ownership_ratio, captain_proxy, dcs_ratio, velocity_delta
3. FPL+Odds: All features combined

Train: 2016-17 to 2021-22 | Val: 2022-23 | Test: 2023-24 + 2024-25

Outputs: model pickles, scaler, metrics JSON in outputs/model/
"""

import sys
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS,
    load_parquet, MODEL_DIR,
)

ODDS_FEATURES = ['implied_prob_h', 'implied_prob_d', 'implied_prob_a']
FPL_FEATURES = ['transfer_ratio', 'ownership_ratio', 'captain_proxy', 'dcs_ratio', 'velocity_delta']
ALL_FEATURES = ODDS_FEATURES + FPL_FEATURES

RESULT_MAP = {'A': 0, 'D': 1, 'H': 2}


def prepare_data(df: pd.DataFrame):
    """Split into train/val/test and prepare features/targets."""
    # Drop rows with NaN in any feature or odds
    feature_cols = ALL_FEATURES + ['result']
    df = df.dropna(subset=feature_cols)

    train = df[df['season'].isin(TRAIN_SEASONS)]
    val = df[df['season'].isin(VAL_SEASONS)]
    test = df[df['season'].isin(TEST_SEASONS)]

    print(f"Train: {len(train)} matches ({TRAIN_SEASONS[0]}–{TRAIN_SEASONS[-1]})")
    print(f"Val:   {len(val)} matches ({VAL_SEASONS[0]})")
    print(f"Test:  {len(test)} matches ({TEST_SEASONS[0]}–{TEST_SEASONS[-1]})")

    y_train = train['result'].map(RESULT_MAP).values
    y_val = val['result'].map(RESULT_MAP).values
    y_test = test['result'].map(RESULT_MAP).values

    return train, val, test, y_train, y_val, y_test


def multiclass_brier(y_true, y_proba, n_classes=3):
    """Compute mean Brier score across classes."""
    return np.mean([
        brier_score_loss((y_true == c).astype(int), y_proba[:, c])
        for c in range(n_classes)
    ])


def train_and_evaluate(X_train, y_train, X_val, y_val, features, name, C=1.0):
    """Train a calibrated logistic regression and evaluate."""
    base = LogisticRegression(
        solver='lbfgs', max_iter=1000, C=C
    )
    model = CalibratedClassifierCV(base, cv=5, method='isotonic')
    model.fit(X_train[features].values, y_train)

    y_pred_proba = model.predict_proba(X_val[features].values)
    y_pred = model.predict(X_val[features].values)

    brier = multiclass_brier(y_val, y_pred_proba)
    ll = log_loss(y_val, y_pred_proba)
    acc = accuracy_score(y_val, y_pred)

    return model, {'brier': brier, 'log_loss': ll, 'accuracy': acc, 'proba': y_pred_proba}


def main():
    print("=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)

    df = load_parquet('signals.parquet')
    train, val, test, y_train, y_val, y_test = prepare_data(df)

    # Standardize features
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(train[ALL_FEATURES]), columns=ALL_FEATURES, index=train.index)
    X_val = pd.DataFrame(scaler.transform(val[ALL_FEATURES]), columns=ALL_FEATURES, index=val.index)
    X_test = pd.DataFrame(scaler.transform(test[ALL_FEATURES]), columns=ALL_FEATURES, index=test.index)

    # --- Train 3 models ---
    print("\n--- Training Models ---")
    models = {}
    metrics = {}

    for name, features in [('odds_only', ODDS_FEATURES), ('fpl_only', FPL_FEATURES), ('fpl_odds', ALL_FEATURES)]:
        model, result = train_and_evaluate(X_train, y_train, X_val, y_val, features, name)
        models[name] = model
        metrics[name] = {k: v for k, v in result.items() if k != 'proba'}
        metrics[name]['val_proba'] = result['proba'].tolist()
        print(f"  {name:<12} Brier={result['brier']:.4f}  LogLoss={result['log_loss']:.4f}  Acc={result['accuracy']:.3f}")

    # Brier Skill Score
    bss = 1 - (metrics['fpl_odds']['brier'] / metrics['odds_only']['brier'])
    print(f"\nBrier Skill Score (FPL+Odds vs Odds-Only): {bss:.4f}")
    if bss > 0.02:
        print("  → Meaningful signal detected. Proceeding with full backtest.")
    elif bss > 0:
        print("  → Weak signal detected. May have value in specific subsets.")
    else:
        print("  → FPL signals don't improve over odds alone on validation set.")

    # --- Hyperparameter sensitivity ---
    print("\n--- Hyperparameter Sensitivity (FPL+Odds, varying C) ---")
    c_results = {}
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        _, result = train_and_evaluate(X_train, y_train, X_val, y_val, ALL_FEATURES, 'fpl_odds', C=C)
        c_results[C] = result['brier']
        print(f"  C={C:<8} Brier={result['brier']:.4f}")

    best_C = min(c_results, key=c_results.get)
    print(f"  Best C: {best_C}")

    # Retrain with best C
    if best_C != 1.0:
        print(f"\nRetraining FPL+Odds with C={best_C}...")
        model, result = train_and_evaluate(X_train, y_train, X_val, y_val, ALL_FEATURES, 'fpl_odds', C=best_C)
        models['fpl_odds'] = model
        metrics['fpl_odds'] = {k: v for k, v in result.items() if k != 'proba'}
        metrics['fpl_odds']['val_proba'] = result['proba'].tolist()
        bss = 1 - (metrics['fpl_odds']['brier'] / metrics['odds_only']['brier'])
        print(f"  Updated Brier={result['brier']:.4f}, BSS={bss:.4f}")

    # --- Feature importance ---
    print("\n--- Feature Importance (FPL+Odds model coefficients) ---")
    # Extract base estimator coefficients from the calibrated model
    base_model = LogisticRegression(
        solver='lbfgs', max_iter=1000, C=best_C
    )
    base_model.fit(X_train[ALL_FEATURES].values, y_train)
    coefs = pd.DataFrame(
        base_model.coef_,
        columns=ALL_FEATURES,
        index=['Away', 'Draw', 'Home']
    )
    print(coefs.round(3).to_string())
    print("\nMean absolute coefficient per feature:")
    importance = coefs.abs().mean().sort_values(ascending=False)
    for feat, imp in importance.items():
        marker = "  *** FPL" if feat in FPL_FEATURES else "      Odds"
        print(f"  {feat:<20} {imp:.4f}  {marker}")

    # --- Evaluate on test set ---
    print("\n--- Test Set Performance ---")
    for name, features in [('odds_only', ODDS_FEATURES), ('fpl_only', FPL_FEATURES), ('fpl_odds', ALL_FEATURES)]:
        y_pred_proba = models[name].predict_proba(X_test[features].values)
        y_pred = models[name].predict(X_test[features].values)
        brier = multiclass_brier(y_test, y_pred_proba)
        ll = log_loss(y_test, y_pred_proba)
        acc = accuracy_score(y_test, y_pred)
        print(f"  {name:<12} Brier={brier:.4f}  LogLoss={ll:.4f}  Acc={acc:.3f}")
        metrics[name]['test_brier'] = brier
        metrics[name]['test_log_loss'] = ll
        metrics[name]['test_accuracy'] = acc

    test_bss = 1 - (metrics['fpl_odds']['test_brier'] / metrics['odds_only']['test_brier'])
    print(f"\nTest BSS: {test_bss:.4f}")
    metrics['brier_skill_score_val'] = float(1 - (metrics['fpl_odds']['brier'] / metrics['odds_only']['brier']))
    metrics['brier_skill_score_test'] = float(test_bss)
    metrics['best_C'] = float(best_C)

    # --- Save artifacts ---
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        with open(MODEL_DIR / f'model_{name}.pkl', 'wb') as f:
            pickle.dump(model, f)

    with open(MODEL_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Save metrics (convert numpy types for JSON serialization)
    metrics_clean = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            metrics_clean[k] = {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                                for kk, vv in v.items() if kk != 'val_proba'}
        else:
            metrics_clean[k] = float(v) if isinstance(v, (np.floating, float)) else v

    with open(MODEL_DIR / 'metrics.json', 'w') as f:
        json.dump(metrics_clean, f, indent=2)

    # Save test predictions for backtesting
    test_preds = pd.DataFrame()
    for col in test.columns:
        test_preds[col] = test[col].values

    for name, features in [('odds_only', ODDS_FEATURES), ('fpl_only', FPL_FEATURES), ('fpl_odds', ALL_FEATURES)]:
        proba = models[name].predict_proba(X_test[features].values)
        test_preds[f'p_{name}_A'] = proba[:, 0]
        test_preds[f'p_{name}_D'] = proba[:, 1]
        test_preds[f'p_{name}_H'] = proba[:, 2]

    test_preds.to_parquet(MODEL_DIR / 'test_predictions.parquet', index=False)
    print(f"\nSaved model artifacts to {MODEL_DIR}")


if __name__ == '__main__':
    main()
