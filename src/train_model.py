"""
MODEL TRAINING
==============
For each ticker:
  1. Walk-forward cross-validation (TimeSeriesSplit) for honest evaluation
  2. Optuna hyperparameter tuning for XGBoost and LightGBM (best performers)
  3. Train final ensemble on ALL data with best params
  4. Save {'model': model, 'feature_names': feature_cols} for exact feature alignment in prod

Requires: pip install optuna
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from features import DROP_FROM_MODEL

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("WARNING: optuna not installed — using default hyperparameters.")
    print("Install with: pip install optuna\n")

# ==================== CONFIG ====================

from bot_config import TICKERS as tickers

N_OPTUNA_TRIALS = 40     # trials per model (XGB + LGBM)
N_CV_SPLITS = 5          # walk-forward folds for both Optuna and final evaluation
MIN_ACCURACY = 0.50      # minimum to include in ensemble
EVAL_SPLIT = 0.80        # train fraction for holdout evaluation reporting


# ==================== WALK-FORWARD EVALUATION ====================

def walk_forward_auc(model_fn, X, y, n_splits=N_CV_SPLITS):
    """TimeSeriesSplit AUC — used as Optuna objective and for final eval reporting."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        m = model_fn()
        m.fit(X_tr, y_tr)
        try:
            scores.append(roc_auc_score(y_val, m.predict_proba(X_val)[:, 1]))
        except Exception:
            scores.append(0.5)
    return float(np.mean(scores))


# ==================== OPTUNA OBJECTIVES ====================

def _xgb_objective(trial, X, y):
    params = dict(
        n_estimators=trial.suggest_int('n_estimators', 200, 700),
        max_depth=trial.suggest_int('max_depth', 3, 9),
        learning_rate=trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
        subsample=trial.suggest_float('subsample', 0.5, 1.0),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.4, 1.0),
        colsample_bylevel=trial.suggest_float('colsample_bylevel', 0.4, 1.0),
        reg_alpha=trial.suggest_float('reg_alpha', 1e-3, 20.0, log=True),
        reg_lambda=trial.suggest_float('reg_lambda', 1e-3, 20.0, log=True),
        min_child_weight=trial.suggest_int('min_child_weight', 1, 40),
        gamma=trial.suggest_float('gamma', 0.0, 8.0),
        scale_pos_weight=trial.suggest_float('scale_pos_weight', 0.5, 3.0),
    )
    return walk_forward_auc(
        lambda: XGBClassifier(**params, random_state=42,
                              eval_metric='logloss', verbosity=0),
        X, y,
    )


def _lgbm_objective(trial, X, y):
    params = dict(
        n_estimators=trial.suggest_int('n_estimators', 200, 700),
        max_depth=trial.suggest_int('max_depth', 3, 9),
        learning_rate=trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
        num_leaves=trial.suggest_int('num_leaves', 15, 127),
        subsample=trial.suggest_float('subsample', 0.5, 1.0),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.4, 1.0),
        reg_alpha=trial.suggest_float('reg_alpha', 1e-3, 20.0, log=True),
        reg_lambda=trial.suggest_float('reg_lambda', 1e-3, 20.0, log=True),
        min_child_samples=trial.suggest_int('min_child_samples', 5, 60),
        scale_pos_weight=trial.suggest_float('scale_pos_weight', 0.5, 3.0),
    )
    return walk_forward_auc(
        lambda: LGBMClassifier(**params, random_state=42, verbosity=-1),
        X, y,
    )


def tune(objective_fn, X, y, n_trials, name):
    """Run Optuna study and return best params dict."""
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda t: objective_fn(t, X, y),
                   n_trials=n_trials, show_progress_bar=False)
    best = study.best_trial
    print(f"  {name} best AUC: {best.value:.4f} (trial {best.number})")
    return best.params


# ==================== MAIN TRAINING LOOP ====================

for ticker in tickers:
    print(f"\n{'='*65}")
    print(f"  {ticker}")
    print(f"{'='*65}")

    df = pd.read_csv(f'data/{ticker}_features.csv')
    feature_cols = [c for c in df.columns if c not in DROP_FROM_MODEL]
    X = df[feature_cols]
    y = df['target']

    split = int(len(df) * EVAL_SPLIT)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    print(f"Rows: {len(df)}  |  Features: {len(feature_cols)}")
    print(f"Train: {len(X_train)}  |  Holdout: {len(X_test)}")
    print(f"Target balance (full): UP={y.mean():.1%}  DOWN={(1-y.mean()):.1%}\n")

    # ===== HYPERPARAMETER TUNING =====
    if HAS_OPTUNA:
        print(f"Tuning XGBoost ({N_OPTUNA_TRIALS} trials)...")
        xgb_params = tune(_xgb_objective, X_train, y_train, N_OPTUNA_TRIALS, 'XGBoost')

        print(f"Tuning LightGBM ({N_OPTUNA_TRIALS} trials)...")
        lgbm_params = tune(_lgbm_objective, X_train, y_train, N_OPTUNA_TRIALS, 'LightGBM')
    else:
        xgb_params = dict(n_estimators=300, max_depth=6, learning_rate=0.03,
                          subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0,
                          reg_lambda=1.0)
        lgbm_params = dict(n_estimators=300, max_depth=6, learning_rate=0.03,
                           num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                           reg_alpha=1.0, reg_lambda=1.0)

    # ===== TRAIN BASE MODELS ON TRAINING SET (for holdout evaluation) =====
    print("\nTraining base models on train split...")

    rf = RandomForestClassifier(
        n_estimators=400, max_depth=12, min_samples_leaf=15,
        max_features='sqrt', class_weight='balanced', n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)

    gb = GradientBoostingClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.04,
        min_samples_leaf=15, subsample=0.8, random_state=42)
    gb.fit(X_train, y_train)

    xgb = XGBClassifier(**xgb_params, random_state=42,
                         eval_metric='logloss', verbosity=0)
    xgb.fit(X_train, y_train)

    lgbm = LGBMClassifier(**lgbm_params, random_state=42, verbosity=-1)
    lgbm.fit(X_train, y_train)

    # ===== EVALUATE ON HOLDOUT =====
    models = {
        'Random Forest':      rf,
        'Gradient Boosting':  gb,
        'XGBoost':            xgb,
        'LightGBM':           lgbm,
    }

    print("\nHoldout results:")
    print(f"  {'Model':<25}  {'Acc':>6}  {'AUC':>6}")
    print("  " + "-" * 40)
    results = {}
    for name, m in models.items():
        acc = accuracy_score(y_test, m.predict(X_test))
        try:
            auc = roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])
        except Exception:
            auc = 0.5
        results[name] = {'model': m, 'acc': acc, 'auc': auc}
        print(f"  {name:<25}  {acc:.4f}  {auc:.4f}")

    # ===== SOFT VOTING ENSEMBLE =====
    ensemble_members = [
        (n.lower().replace(' ', '_'), info['model'])
        for n, info in results.items()
        if info['acc'] >= MIN_ACCURACY
    ]

    ensemble = VotingClassifier(estimators=ensemble_members, voting='soft', n_jobs=-1)
    ensemble.fit(X_train, y_train)
    ens_acc = accuracy_score(y_test, ensemble.predict(X_test))
    try:
        ens_auc = roc_auc_score(y_test, ensemble.predict_proba(X_test)[:, 1])
    except Exception:
        ens_auc = 0.5
    print(f"  {'Soft Voting Ensemble':<25}  {ens_acc:.4f}  {ens_auc:.4f}")

    # ===== STACKING ENSEMBLE =====
    stack_members = [
        (n.lower().replace(' ', '_'), info['model'])
        for n, info in results.items()
        if info['acc'] >= MIN_ACCURACY
    ]
    meta = LogisticRegression(C=0.1, max_iter=500, random_state=42)
    stacker = StackingClassifier(
        estimators=stack_members,
        final_estimator=meta,
        cv=5,
        stack_method='predict_proba',
        n_jobs=-1,
        passthrough=False,
    )
    stacker.fit(X_train, y_train)
    stack_acc = accuracy_score(y_test, stacker.predict(X_test))
    try:
        stack_auc = roc_auc_score(y_test, stacker.predict_proba(X_test)[:, 1])
    except Exception:
        stack_auc = 0.5
    print(f"  {'Stacking Ensemble':<25}  {stack_acc:.4f}  {stack_auc:.4f}")

    # ===== PICK BEST MODEL BY AUC =====
    candidates = {**{n: (i['auc'], i['model']) for n, i in results.items()},
                  'Soft Voting': (ens_auc, ensemble),
                  'Stacking':    (stack_auc, stacker)}
    best_name = max(candidates, key=lambda k: candidates[k][0])
    best_auc, best_holdout_model = candidates[best_name]
    print(f"\n>>> SELECTED for production: {best_name} (AUC={best_auc:.4f})")

    # ===== CLASSIFICATION REPORT =====
    print(classification_report(y_test, best_holdout_model.predict(X_test),
                                 target_names=['DOWN', 'UP']))

    # ===== FEATURE IMPORTANCE =====
    # Get importance from the best tree model (XGB or LGBM preferred)
    for pref in ['XGBoost', 'LightGBM', 'Random Forest', 'Gradient Boosting']:
        if pref in results:
            importances = pd.Series(
                results[pref]['model'].feature_importances_, index=feature_cols)
            print("Top 20 features:")
            print(importances.sort_values(ascending=False).head(20).to_string())
            break

    # ===== RETRAIN ON ALL DATA =====
    # Determine which model type won and retrain on full dataset
    print(f"\nRetraining {best_name} on ALL {len(X)} rows...")

    if best_name == 'Stacking':
        # Rebuild stacker with base models retrained on all data
        rf_f = RandomForestClassifier(
            n_estimators=400, max_depth=12, min_samples_leaf=15,
            max_features='sqrt', class_weight='balanced', n_jobs=-1, random_state=42)
        gb_f = GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.04,
            min_samples_leaf=15, subsample=0.8, random_state=42)
        xgb_f = XGBClassifier(**xgb_params, random_state=42,
                               eval_metric='logloss', verbosity=0)
        lgbm_f = LGBMClassifier(**lgbm_params, random_state=42, verbosity=-1)
        meta_f = LogisticRegression(C=0.1, max_iter=500, random_state=42)
        final_model = StackingClassifier(
            estimators=[
                ('random_forest', rf_f), ('gradient_boosting', gb_f),
                ('xgboost', xgb_f), ('lightgbm', lgbm_f),
            ],
            final_estimator=meta_f,
            cv=5,
            stack_method='predict_proba',
            n_jobs=-1,
        )

    elif best_name == 'Soft Voting':
        rf_f = RandomForestClassifier(
            n_estimators=400, max_depth=12, min_samples_leaf=15,
            max_features='sqrt', class_weight='balanced', n_jobs=-1, random_state=42)
        gb_f = GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.04,
            min_samples_leaf=15, subsample=0.8, random_state=42)
        xgb_f = XGBClassifier(**xgb_params, random_state=42,
                               eval_metric='logloss', verbosity=0)
        lgbm_f = LGBMClassifier(**lgbm_params, random_state=42, verbosity=-1)
        final_model = VotingClassifier(
            estimators=[('rf', rf_f), ('gb', gb_f), ('xgb', xgb_f), ('lgbm', lgbm_f)],
            voting='soft', n_jobs=-1,
        )

    elif best_name == 'XGBoost':
        final_model = XGBClassifier(**xgb_params, random_state=42,
                                     eval_metric='logloss', verbosity=0)
    elif best_name == 'LightGBM':
        final_model = LGBMClassifier(**lgbm_params, random_state=42, verbosity=-1)
    elif best_name == 'Random Forest':
        final_model = RandomForestClassifier(
            n_estimators=400, max_depth=12, min_samples_leaf=15,
            max_features='sqrt', class_weight='balanced', n_jobs=-1, random_state=42)
    else:
        final_model = GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.04,
            min_samples_leaf=15, subsample=0.8, random_state=42)

    final_model.fit(X, y)

    # Save model + feature names for exact alignment in prod
    payload = {'model': final_model, 'feature_names': feature_cols}
    joblib.dump(payload, f'models/{ticker}.pkl')
    print(f"Saved models/{ticker}.pkl  ({best_name}, {len(feature_cols)} features)\n")
