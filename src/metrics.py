import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, brier_score_loss,
    precision_recall_curve, roc_curve,
)
from sklearn.linear_model import LogisticRegression
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

def threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    """Sensitivity, specificity, and balanced accuracy at a fixed threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "threshold": float(threshold),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "balanced_accuracy": float((sensitivity + specificity) / 2),
    }

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    threshold: float = None) -> dict:
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    f1, thresh = optimal_threshold_f1(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    t = threshold if threshold is not None else float(y_true.mean())
    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1_at_opt_threshold": f1,
        "optimal_threshold": thresh,
        "brier_score": brier,
        **threshold_metrics(y_true, y_prob, t),
    }

def bootstrap_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_iterations: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    prevalence = float(y_true.mean())
    records = {k: [] for k in [
        "auroc", "auprc", "f1_at_opt_threshold", "brier_score",
        "sensitivity", "specificity", "balanced_accuracy",
    ]}

    for _ in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        m = compute_metrics(yt, yp, threshold=prevalence)
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
    results = {}
    for anc in np.unique(ancestry_labels):
        mask = ancestry_labels == anc
        if mask.sum() < 20 or len(np.unique(y_true[mask])) < 2:
            continue
        results[anc] = compute_metrics(y_true[mask], y_prob[mask])
    return results

def hosmer_lemeshow(y_true: np.ndarray, y_prob: np.ndarray, n_groups: int = 10) -> dict:
    order  = np.argsort(y_prob)
    y_sort = y_true[order]
    p_sort = y_prob[order]
    groups = np.array_split(np.arange(len(y_true)), n_groups)

    hl_stat = 0.0
    for g in groups:
        obs = y_sort[g].sum()
        exp = p_sort[g].sum()
        n_g = len(g)
        exp_neg = n_g - exp
        if exp > 0:
            hl_stat += (obs - exp) ** 2 / exp
        if exp_neg > 0:
            hl_stat += (n_g - obs - exp_neg) ** 2 / exp_neg

    df = n_groups - 2
    p  = 1 - stats.chi2.cdf(hl_stat, df)
    return {"hl_stat": float(hl_stat), "hl_df": df, "hl_p": float(p)}

def calibration_slope(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    eps    = 1e-7
    p      = np.clip(y_prob, eps, 1 - eps)
    logits = np.log(p / (1 - p))
    lr     = LogisticRegression(fit_intercept=True, C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(logits.reshape(-1, 1), y_true)
    return {
        "cal_slope":     float(lr.coef_[0][0]),
        "cal_intercept": float(lr.intercept_[0]),
    }

def compute_calibration(y_true: np.ndarray, y_prob: np.ndarray, n_groups: int = 10) -> dict:
    """Combined calibration metrics: HL test + slope/intercept."""
    return {**hosmer_lemeshow(y_true, y_prob, n_groups),
            **calibration_slope(y_true, y_prob)}

def pairwise_delong(model_probs: dict, y_true: np.ndarray) -> dict:
    names = list(model_probs.keys())
    results = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            results[(n1, n2)] = delong_test(y_true, model_probs[n1], model_probs[n2])
    return results
