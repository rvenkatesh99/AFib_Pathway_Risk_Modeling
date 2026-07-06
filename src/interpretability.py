"""
Interpretability analyses:
  - Global attention weight extraction (global attention model)
  - RF permutation importance -> pathway ranking
  - L1 logistic regression coefficient magnitudes -> pathway ranking
  - Spearman rank correlation between pathway rankings
  - Transformer attention head specialization analysis
  - GradCAM node attribution (GNN)
"""

import numpy as np
from scipy.stats import spearmanr


def get_attention_pathway_ranking(model, pathway_names: list) -> dict:
    """
    Extract alpha weights from GlobalPathwayAttentionModel.
    Returns {pathway_name: weight} sorted descending.
    """
    importance = model.get_pathway_importance_dict(pathway_names)
    return dict(sorted(importance.items(), key=lambda x: -x[1]))


def get_rf_pathway_ranking(rf_model, pathway_names: list, n_features_per_pathway: int = 1,
                           covariate_names=None):
    """Returns (pathway_ranking, covariate_ranking) — both sorted descending."""
    from src.models.baseline import get_rf_pathway_importances
    pw_dict, cov_dict = get_rf_pathway_importances(rf_model, pathway_names, n_features_per_pathway, covariate_names)
    return (dict(sorted(pw_dict.items(), key=lambda x: -x[1])),
            dict(sorted(cov_dict.items(), key=lambda x: -x[1])))


def get_l1_pathway_ranking(l1_pipeline, pathway_names: list, n_features_per_pathway: int = 1,
                           covariate_names=None):
    """Returns (pathway_ranking, covariate_ranking) — both sorted descending."""
    from src.models.baseline import get_l1_pathway_coefs
    pw_dict, cov_dict = get_l1_pathway_coefs(l1_pipeline, pathway_names, n_features_per_pathway, covariate_names)
    return (dict(sorted(pw_dict.items(), key=lambda x: -x[1])),
            dict(sorted(cov_dict.items(), key=lambda x: -x[1])))


def compute_gradient_covariate_importance(
    model,
    test_loader,
    cov_cols: list,
    device: str = "cpu",
) -> dict:
    """
    Gradient × input covariate attribution for any differentiable pathway model.

    For each covariate j, importance = mean over test set of |∂logit/∂cov_j × cov_j|.
    This captures both the model's sensitivity to the covariate and its typical magnitude,
    giving a non-negative importance score comparable across covariates.

    Returns {covariate_name: importance_score} sorted descending.
    """
    import torch

    model = model.to(device)
    model.eval()

    accum = np.zeros(len(cov_cols), dtype=np.float64)
    n_samples = 0

    for batch in test_loader:
        pw  = batch["pathway_features"].to(device)
        cov = batch["covariates"].to(device).float()
        cov.requires_grad_(True)

        logits = model(pw, cov)
        # Scalar output: sum logits so backward gives per-sample gradients summed
        logits.sum().backward()

        if cov.grad is not None:
            # gradient × input attribution, averaged over batch
            attr = (cov.grad * cov).detach().abs().cpu().numpy()  # (batch, C)
            accum += attr.sum(axis=0)
            n_samples += attr.shape[0]

        cov.requires_grad_(False)

    if n_samples == 0:
        return {}

    importance = accum / n_samples
    return dict(sorted(zip(cov_cols, importance.tolist()), key=lambda x: -x[1]))


def spearman_rank_concordance(rankings: dict) -> dict:
    """
    Compute pairwise Spearman rank correlation between pathway rankings.
    rankings: {model_name: {pathway_name: importance_score}}
    Returns {(model1, model2): {"rho": ..., "p_value": ...}}.
    """
    model_names = list(rankings.keys())
    all_pathways = list(next(iter(rankings.values())).keys())

    results = {}
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            n1, n2 = model_names[i], model_names[j]
            scores1 = [rankings[n1].get(p, 0.0) for p in all_pathways]
            scores2 = [rankings[n2].get(p, 0.0) for p in all_pathways]
            rho, p = spearmanr(scores1, scores2)
            results[(n1, n2)] = {"rho": float(rho), "p_value": float(p)}
    return results


def analyze_transformer_attention_heads(
    mean_attn_weights: "torch.Tensor",
    pathway_names: list,
) -> dict:
    """
    Analyze head specialization in the pathway transformer.

    mean_attn_weights: (n_layers, n_heads, K+1, K+1) tensor (mean over test set)
                       Index 0 in the sequence dimension is the CLS token.

    Returns per-layer, per-head summary of top attended pathways
    and an entropy measure of head specialization.
    """
    import torch
    n_layers, n_heads, S, _ = mean_attn_weights.shape
    K = len(pathway_names)

    results = {}
    for layer in range(n_layers):
        layer_results = {}
        for head in range(n_heads):
            # CLS token attention over pathways (exclude CLS-to-CLS)
            cls_attn = mean_attn_weights[layer, head, 0, 1:].numpy()  # (K,)
            cls_attn = cls_attn / (cls_attn.sum() + 1e-9)

            # Entropy: low entropy = specialized head
            entropy = float(-np.sum(cls_attn * np.log(cls_attn + 1e-9)))

            top_idx = np.argsort(-cls_attn)[:10]
            top_pathways = [(pathway_names[i], float(cls_attn[i])) for i in top_idx]

            layer_results[f"head_{head}"] = {
                "entropy": entropy,
                "top_pathways": top_pathways,
            }
        results[f"layer_{layer}"] = layer_results
    return results


def compute_gradcam_pathway_ranking(
    gnn_model,
    test_loader,
    pathway_names: list,
    device: str = "cpu",
) -> dict:
    """
    Compute mean GradCAM node attribution across test set for the GNN.
    Returns {pathway_name: mean_gradcam_score} sorted descending.
    """
    gnn_model.eval()
    all_scores = []
    for batch in test_loader:
        pw = batch["pathway_features"].to(device)
        cov = batch["covariates"].to(device)
        scores = gnn_model.get_node_gradcam(pw, cov)  # (batch, K)
        all_scores.append(scores.cpu().numpy())
    mean_scores = np.concatenate(all_scores, axis=0).mean(axis=0)  # (K,)
    importance = dict(zip(pathway_names, mean_scores.tolist()))
    return dict(sorted(importance.items(), key=lambda x: -x[1]))
