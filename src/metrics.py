"""
Evaluation metrics:
  - AUROC, AUPRC, F1 at optimal threshold, Brier score
  - Bootstrap CIs (1000 iterations)
  - DeLong's test for correlated AUCs
  - Subgroup (ancestry) reporting
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, brier_score_loss,
    precision_recall_curve, roc_curve,
)
from scipy import stats


def optimal_threshold_f1(y_true, y_prob):
    """F1 score at the threshold maximizing F1 on the provided data."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = np.where(
        (precisions + recalls) == 0, 0,
        2 * precisions * recalls / (precisions + recalls + 1e-9),
    )
    best_idx = np.argmax(f1s[:-1])  # last threshold is undefined
    best_thresh = thresholds[best_idx]
    y_pred = (y_prob >= best_thresh).astype(int)
    return f1_score(y_true, y_pred), best_thresh


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    f1, thresh = optimal_threshold_f1(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1_at_opt_threshold": f1,
        "optimal_threshold": thresh,
        "brier_score": brier,
    }


def bootstrap_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_iterations: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Bootstrap confidence intervals for all metrics.
    Returns dict of {metric: {"mean": ..., "ci_lower": ..., "ci_upper": ...}}.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    records = {k: [] for k in ["auroc", "auprc", "f1_at_opt_threshold", "brier_score"]}

    for _ in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        m = compute_metrics(yt, yp)
        for k in records:
            records[k].append(m[k])

    alpha = (1 - ci_level) / 2
    result = {}
    for k, vals in records.items():
        vals = np.array(vals)
        result[k] = {
            "mean": float(np.mean(vals)),
            "ci_lower": float(np.quantile(vals, alpha)),
            "ci_upper": float(np.quantile(vals, 1 - alpha)),
        }
    return result


# ---- DeLong's test for correlated AUCs ----
# Based on: DeLong et al. (1988) Biometrics and the fastDeLong algorithm.

def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2


def _fastDeLong(y_true, prob_pred_1, prob_pred_2):
    """Returns (auc1, auc2, var_auc1, var_auc2, covar)."""
    order = np.argsort(-y_true)
    label_1 = y_true[order]
    p1 = prob_pred_1[order]
    p2 = prob_pred_2[order]

    n_pos = int(label_1.sum())
    n_neg = len(label_1) - n_pos

    pos_mask = label_1 == 1
    neg_mask = ~pos_mask

    def _auc_and_structural_components(probs):
        m_r = _compute_midrank(probs)
        auc = (m_r[pos_mask].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        tx = np.zeros(n_pos)
        ty = np.zeros(n_neg)
        for pi, pv in enumerate(probs[pos_mask]):
            ty += (pv > probs[neg_mask]).astype(float) + 0.5 * (pv == probs[neg_mask]).astype(float)
        tx = ty.mean() * np.ones(n_pos)  # simplified; full version below
        # Structural components (placement values)
        pos_probs = probs[pos_mask]
        neg_probs = probs[neg_mask]
        V10 = np.array([
            ((p > neg_probs).sum() + 0.5 * (p == neg_probs).sum()) / n_neg
            for p in pos_probs
        ])
        V01 = np.array([
            ((p < pos_probs).sum() + 0.5 * (p == pos_probs).sum()) / n_pos
            for p in neg_probs
        ])
        return auc, V10, V01

    auc1, V10_1, V01_1 = _auc_and_structural_components(p1)
    auc2, V10_2, V01_2 = _auc_and_structural_components(p2)

    var1 = (np.var(V10_1, ddof=1) / n_pos + np.var(V01_1, ddof=1) / n_neg)
    var2 = (np.var(V10_2, ddof=1) / n_pos + np.var(V01_2, ddof=1) / n_neg)
    covar = (np.cov(V10_1, V10_2, ddof=1)[0, 1] / n_pos
             + np.cov(V01_1, V01_2, ddof=1)[0, 1] / n_neg)

    return auc1, auc2, var1, var2, covar


def delong_test(y_true: np.ndarray, prob1: np.ndarray, prob2: np.ndarray) -> dict:
    """
    DeLong's test for comparing two correlated AUCs.
    Returns {"auc1", "auc2", "z_stat", "p_value"}.
    """
    auc1, auc2, var1, var2, covar = _fastDeLong(y_true, prob1, prob2)
    diff_var = var1 + var2 - 2 * covar
    z = (auc1 - auc2) / np.sqrt(max(diff_var, 1e-12))
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"auc1": auc1, "auc2": auc2, "z_stat": z, "p_value": p}


def evaluate_subgroups(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ancestry_labels: np.ndarray,
) -> dict:
    """
    Evaluate metrics within each ancestry subgroup.
    ancestry_labels: array of strings (e.g., "EUR", "AFR")
    Returns {ancestry: metrics_dict}.
    """
    results = {}
    for anc in np.unique(ancestry_labels):
        mask = ancestry_labels == anc
        if mask.sum() < 20 or len(np.unique(y_true[mask])) < 2:
            continue
        results[anc] = compute_metrics(y_true[mask], y_prob[mask])
    return results


def pairwise_delong(model_probs: dict, y_true: np.ndarray) -> dict:
    """
    All pairwise DeLong tests.
    model_probs: {model_name: y_prob array}
    Returns {(name1, name2): delong_result_dict}.
    """
    names = list(model_probs.keys())
    results = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            results[(n1, n2)] = delong_test(y_true, model_probs[n1], model_probs[n2])
    return results
