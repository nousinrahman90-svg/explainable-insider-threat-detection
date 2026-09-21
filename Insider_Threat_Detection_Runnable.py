"""Runnable explainable insider-threat detection demonstration.

Required packages:
    python -m pip install numpy pandas scikit-learn

Optional packages (the script falls back safely when unavailable):
    python -m pip install xgboost shap

Important: this program generates synthetic demonstration data. It does not
load or reproduce the CMU CERT r4.2 dataset.
"""

from __future__ import annotations

import argparse
import warnings

try:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise SystemExit(
        "Missing a required package. Run:\n"
        "python -m pip install numpy pandas scikit-learn\n"
        f"Original error: {exc}"
    ) from exc

warnings.filterwarnings("ignore", category=UserWarning)


FEATURES = [
    "login_failed_ratio",
    "after_hours_ratio",
    "usb_bytes_norm",
    "dest_host_entropy",
    "http_business_ratio",
    "session_duration_var",
]


def generate_synthetic_telemetry(
    n_samples: int = 25_000,
    threat_ratio: float = 0.01,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate reproducible synthetic benign and threat-like sessions."""
    if n_samples < 200:
        raise ValueError("n_samples must be at least 200.")
    if not 0.001 <= threat_ratio < 0.5:
        raise ValueError("threat_ratio must be between 0.001 and 0.5.")

    rng = np.random.default_rng(random_state)
    n_threats = max(2, int(round(n_samples * threat_ratio)))
    n_benign = n_samples - n_threats

    benign = pd.DataFrame(
        {
            "login_failed_ratio": rng.beta(0.5, 15, n_benign),
            "after_hours_ratio": rng.beta(0.8, 10, n_benign),
            "usb_bytes_norm": rng.exponential(0.08, n_benign),
            "dest_host_entropy": rng.normal(1.15, 0.20, n_benign),
            "http_business_ratio": rng.beta(9, 2, n_benign),
            "session_duration_var": rng.gamma(2.0, 1.0, n_benign),
            "is_threat": np.zeros(n_benign, dtype=int),
        }
    )
    threat = pd.DataFrame(
        {
            "login_failed_ratio": rng.beta(4.5, 2.5, n_threats),
            "after_hours_ratio": rng.beta(7.5, 2.0, n_threats),
            "usb_bytes_norm": rng.exponential(3.2, n_threats) + 1.8,
            "dest_host_entropy": rng.normal(3.85, 0.45, n_threats),
            "http_business_ratio": rng.beta(1.8, 6.0, n_threats),
            "session_duration_var": rng.gamma(5.2, 2.1, n_threats),
            "is_threat": np.ones(n_threats, dtype=int),
        }
    )
    return (
        pd.concat([benign, threat], ignore_index=True)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )


def evaluate_model(y_true, y_pred, y_score, model_name: str) -> dict[str, float | str]:
    """Calculate imbalance-aware binary classification metrics safely."""
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = 100.0 * fp / max(fp + tn, 1)
    pr_auc = average_precision_score(y_true, y_score)
    print(
        f"| {model_name:<36} | {precision:>6.3f} | {recall:>6.3f} | "
        f"{f1:>6.3f} | {pr_auc:>7.3f} | {fpr:>7.2f}% |"
    )
    return {
        "Model": model_name,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "PR-AUC": pr_auc,
        "FPR (%)": fpr,
    }


def fit_supervised_model(X_train, y_train, random_state: int):
    """Use XGBoost when functional; otherwise use a dependable RF fallback."""
    n_positive = int(y_train.sum())
    n_negative = int(len(y_train) - n_positive)
    positive_weight = n_negative / max(n_positive, 1)
    try:
        import xgboost as xgb

        model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=positive_weight,
            eval_metric="aucpr",
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        return model, "XGBoost (cost-sensitive)"
    except Exception as exc:  # optional dependency/version failures
        print(f"[!] XGBoost unavailable ({type(exc).__name__}); using Random Forest.")
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        return model, "Random Forest (cost-sensitive fallback)"


def get_feature_importance(model, X_test, y_test) -> tuple[np.ndarray, str]:
    """Return SHAP importance when compatible, otherwise permutation importance."""
    try:
        import shap

        explanation = shap.TreeExplainer(model)(X_test)
        values = np.asarray(explanation.values)
        # Some classifiers return (samples, features, classes).
        if values.ndim == 3:
            values = values[:, :, 1]
        if values.ndim != 2 or values.shape[1] != X_test.shape[1]:
            raise ValueError(f"Unexpected SHAP output shape: {values.shape}")
        return np.mean(np.abs(values), axis=0), "mean absolute Tree-SHAP"
    except Exception as exc:
        print(f"[!] SHAP unavailable/incompatible ({type(exc).__name__}); using permutation importance.")
        result = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="average_precision",
            n_repeats=5,
            random_state=42,
            n_jobs=-1,
        )
        return result.importances_mean, "permutation importance (PR-AUC decrease)"


def run_pipeline(n_samples: int, threat_ratio: float, random_state: int) -> pd.DataFrame:
    print("=" * 96)
    print("EXPLAINABLE MACHINE-LEARNING DEMO FOR INSIDER-THREAT DETECTION")
    print("Synthetic data demonstration - not a CMU CERT benchmark evaluation")
    print("=" * 96)

    df = generate_synthetic_telemetry(n_samples, threat_ratio, random_state)
    X, y = df[FEATURES], df["is_threat"]

    # A validation set prevents threshold selection from leaking test labels.
    X_development, X_test, y_development, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=random_state
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development,
        y_development,
        test_size=0.25,
        stratify=y_development,
        random_state=random_state,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    X_test_scaled = scaler.transform(X_test)

    print(f"Sessions: {len(df):,} | Threats: {int(y.sum()):,} ({100*y.mean():.2f}%)")
    print(f"Train/validation/test: {len(X_train):,}/{len(X_validation):,}/{len(X_test):,}\n")
    print("-" * 96)
    print(f"| {'Model':<36} | {'Prec.':>6} | {'Recall':>6} | {'F1':>6} | {'PR-AUC':>7} | {'FPR':>8} |")
    print("-" * 96)

    results = []
    iso = IsolationForest(
        n_estimators=150,
        contamination=threat_ratio,
        random_state=random_state,
        n_jobs=-1,
    )
    iso.fit(X_train_scaled[y_train.to_numpy() == 0])
    iso_score = -iso.decision_function(X_test_scaled)
    iso_pred = (iso.predict(X_test_scaled) == -1).astype(int)
    results.append(evaluate_model(y_test, iso_pred, iso_score, "Isolation Forest (unsupervised)"))

    train_benign = X_train_scaled[y_train.to_numpy() == 0]
    validation_benign = X_validation_scaled[y_validation.to_numpy() == 0]
    autoencoder = MLPRegressor(
        hidden_layer_sizes=(16, 8, 16),
        activation="relu",
        solver="adam",
        max_iter=150,
        early_stopping=True,
        validation_fraction=0.10,
        random_state=random_state,
    )
    autoencoder.fit(train_benign, train_benign)
    validation_reconstruction = autoencoder.predict(validation_benign)
    validation_error = np.mean((validation_benign - validation_reconstruction) ** 2, axis=1)
    threshold = float(np.quantile(validation_error, 0.99))
    test_reconstruction = autoencoder.predict(X_test_scaled)
    test_error = np.mean((X_test_scaled - test_reconstruction) ** 2, axis=1)
    ae_pred = (test_error >= threshold).astype(int)
    results.append(evaluate_model(y_test, ae_pred, test_error, "MLP autoencoder (unsupervised)"))

    model, model_name = fit_supervised_model(X_train, y_train, random_state)
    probability = model.predict_proba(X_test)[:, 1]
    prediction = (probability >= 0.50).astype(int)
    results.append(evaluate_model(y_test, prediction, probability, model_name))
    print("-" * 96)

    importance, method = get_feature_importance(model, X_test, y_test)
    importance_table = (
        pd.DataFrame({"Feature": FEATURES, "Importance": importance})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    print(f"\nExplanation method: {method}")
    print(importance_table.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nPipeline completed successfully.")
    return pd.DataFrame(results)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=25_000)
    parser.add_argument("--threat-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.samples, args.threat_ratio, args.seed)
