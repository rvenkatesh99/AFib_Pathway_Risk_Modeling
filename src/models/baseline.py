"""
Baseline models:
  1. L1-penalized logistic regression on flattened pathway matrix + covariates
  2. Elasticnet logistic regression on flattened pathway matrix + covariates
  3. Random forest on pathway features + covariates
  4. Covariates-only logistic regression (clinical benchmark)
"""

import numpy as np
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib


def build_l1_logistic():
    """L1-penalized logistic regression; CV selects C."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegressionCV(
            Cs=np.logspace(-4, 2, 10),
            cv=5,
            l1_ratios=(1,),
            solver="saga",
            max_iter=1000,
            tol=1e-3,
            class_weight="balanced",
            scoring="roc_auc",
            n_jobs=-1,
        )),
    ])


def build_elasticnet():
    """Elasticnet logistic regression; CV jointly selects C and l1_ratio."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegressionCV(
            Cs=np.logspace(-4, 2, 10),
            cv=5,
            l1_ratios=[0.1, 0.5, 0.9, 1.0],
            solver="saga",
            max_iter=1000,
            tol=1e-3,
            class_weight="balanced",
            scoring="roc_auc",
            n_jobs=-1,
        )),
    ])


def build_covariates_logistic(C: float = 1.0):
    """L2 logistic regression on covariates only — clinical benchmark."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=C, solver="lbfgs",
            max_iter=1000, class_weight="balanced",
        )),
    ])


def build_unregularized_logistic():
    """Unregularized logistic regression — standard for PRS + small covariate sets."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty=None, solver="lbfgs",
            max_iter=1000, class_weight="balanced",
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


def get_rf_pathway_importances(rf_model, pathway_names, n_features_per_pathway: int = 1,
                               covariate_names=None):
    """Aggregate per-feature importances to per-pathway level by summing.
    Returns (pathway_dict, covariate_dict)."""
    importances = rf_model.feature_importances_
    K = len(pathway_names)
    T = n_features_per_pathway
    pathway_importances = importances[:K * T].reshape(K, T).sum(axis=1)
    pathway_dict = dict(zip(pathway_names, pathway_importances))
    if covariate_names:
        cov_importances = importances[K * T: K * T + len(covariate_names)]
        cov_dict = dict(zip(covariate_names, cov_importances))
    else:
        cov_dict = {}
    return pathway_dict, cov_dict


def get_l1_pathway_coefs(l1_pipeline, pathway_names, n_features_per_pathway: int = 1,
                         covariate_names=None):
    """Aggregate L1/elasticnet coefficients to pathway level (L2 norm per pathway).
    Returns (pathway_dict, covariate_dict)."""
    coefs = l1_pipeline.named_steps["clf"].coef_[0]
    K = len(pathway_names)
    T = n_features_per_pathway
    n_pathway_features = K * T
    pathway_coefs = coefs[:n_pathway_features].reshape(K, T)
    magnitudes = np.linalg.norm(pathway_coefs, axis=1)
    pathway_dict = dict(zip(pathway_names, magnitudes))
    if covariate_names:
        cov_coefs = np.abs(coefs[n_pathway_features: n_pathway_features + len(covariate_names)])
        cov_dict = dict(zip(covariate_names, cov_coefs))
    else:
        cov_dict = {}
    return pathway_dict, cov_dict


def save_model(model, path: str):
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)
