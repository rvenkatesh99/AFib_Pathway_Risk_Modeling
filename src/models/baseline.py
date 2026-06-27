"""
Baseline models:
  1. Logistic regression on PRS + covariates (clinical benchmark)
  2. L1-penalized logistic regression on flattened pathway matrix + covariates
  3. Random forest on pathway features + covariates
"""

import numpy as np
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib


def build_l2_logistic(C: float = 1.0):
    """Logistic regression on standardized PRS + covariates."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2", C=C, solver="lbfgs",
            max_iter=1000, class_weight="balanced",
        )),
    ])


def build_l1_logistic():
    """L1-penalized logistic regression with 5-fold CV for C selection."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegressionCV(
            Cs=np.logspace(-4, 2, 20),
            cv=5,
            l1_ratios=(1,),        # pure L1 (replaces deprecated penalty="l1")
            solver="saga",
            max_iter=2000,
            class_weight="balanced",
            scoring="roc_auc",
            n_jobs=-1,
        )),
    ])


def build_random_forest(
    n_estimators: int = 500,
    max_depth: int = 6,
    min_samples_leaf: int = 50,
):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )


def flatten_pathway_matrix(pathway_matrix: np.ndarray) -> np.ndarray:
    """(N, K, T) -> (N, K*T)"""
    if pathway_matrix.ndim == 3:
        N = pathway_matrix.shape[0]
        return pathway_matrix.reshape(N, -1)
    return pathway_matrix


def get_rf_pathway_importances(rf_model, pathway_names, n_features_per_pathway: int = 1):
    """
    Aggregate per-feature importances to per-pathway importances by summing
    importance across features belonging to the same pathway.
    Returns a dict {pathway_name: importance}.
    """
    importances = rf_model.feature_importances_
    K = len(pathway_names)
    T = n_features_per_pathway
    pathway_importances = importances.reshape(K, T).sum(axis=1)
    return dict(zip(pathway_names, pathway_importances))


def get_l1_pathway_coefs(l1_pipeline, pathway_names, n_features_per_pathway: int = 1):
    """
    Aggregate L1 logistic regression coefficients to pathway level (L2 norm per pathway).
    Returns a dict {pathway_name: coef_magnitude}.
    """
    coefs = l1_pipeline.named_steps["clf"].coef_[0]
    K = len(pathway_names)
    T = n_features_per_pathway
    n_pathway_features = K * T
    pathway_coefs = coefs[:n_pathway_features].reshape(K, T)
    magnitudes = np.linalg.norm(pathway_coefs, axis=1)
    return dict(zip(pathway_names, magnitudes))


def save_model(model, path: str):
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)
