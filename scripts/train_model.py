"""
Train a single model. Run one process per model, then aggregate_results.py.

Usage:
  python scripts/train_model.py \
      --model {covariates_logistic,l1_logistic,elasticnet,random_forest,global_attention,transformer,gnn} \
      --results_dir results/ \
      [--tune] \
      [--prs_col prs] \
      [--string_edges data/string_edges.csv --pathway_gene_sets data/pathway_gene_sets.json]

Reads:  results_dir/splits.npz and results_dir/data_config.json (from prepare_splits.py)
Writes: results_dir/<model>/
          probs_test.npy, labels_test.npy  — predictions on the held-out test set
          metrics.json                     — AUROC, AUPRC, F1, Brier + bootstrap 95% CIs
          ranking.json                     — pathway importance scores
          best_hparams.json                — chosen hyperparameters
          hparam_search.json               — full grid results (only when --tune)
          <model>.pkl or <model>.pt        — saved model weights
          mean_attention.npy               — transformer only
"""

import argparse
import itertools
import json
import os
import sys
import time
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trainer import (load_data, make_loaders, train, tune_hyperparameters,
                         fit_platt_scaler, predict_calibrated)
from src.models.baseline import (
    build_l1_logistic, build_elasticnet,
    build_random_forest, flatten_pathway_matrix, save_model,
)
from src.models.global_attention import GlobalPathwayAttentionModel
from src.models.pathway_transformer import PathwayTransformer
from src.models.pathway_gnn import (
    PathwayGNN, build_string_edge_index, build_jaccard_edge_index,
    build_score_correlation_edge_index, build_fully_connected_edge_index,
)
from src.metrics import compute_metrics, bootstrap_metrics
from src.interpretability import (
    get_attention_pathway_ranking, get_l1_pathway_ranking, get_rf_pathway_ranking,
    compute_gradcam_pathway_ranking,
)

# ── Hyperparameter grids (Cartesian product of each list) ────────────────────
# Training params (lr, weight_decay) are separated from model constructor params
# automatically in tune_hyperparameters().

HPARAM_GRIDS = {
    "random_forest": {
        "n_estimators": [500],
        "max_depth":    [8, 12, 15, None],
        "min_samples_leaf": [20, 50],
    },
    "global_attention": {
        "embed_dim":    [64, 128, 256],
        "dropout":      [0.1, 0.2, 0.3],
        "lr":           [1e-3, 5e-4, 1e-4],
        "weight_decay": [1e-4, 1e-3],
    },
    "transformer": {
        "embed_dim":    [64, 128],
        "n_heads":      [2, 4],
        "dropout":      [0.1, 0.2],
        "lr":           [1e-3, 1e-4],
        "weight_decay": [1e-4],
    },
    "gnn": {
        "embed_dim":    [64, 128, 256],
        "dropout":      [0.1, 0.2],
        "lr":           [1e-3, 5e-4, 1e-4],
        "weight_decay": [1e-4, 1e-3],
    },
}

# Default hyperparameters used when --tune is not set.
DEFAULTS = {
    "l1_logistic":         {},
    "elasticnet":          {},
    "random_forest":       {"n_estimators": 500, "max_depth": 12, "min_samples_leaf": 20},
    "global_attention":    {"embed_dim": 64, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4},
    "transformer":         {"embed_dim": 64, "n_heads": 4, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4},
    "gnn":                 {"embed_dim": 64, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4},
}

# Final training settings (not tuned).
TRAIN_CFG = dict(n_epochs=200, patience=15, batch_size=256)
TUNE_CFG  = dict(n_epochs=60,  patience=8)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(DEFAULTS))
    p.add_argument("--results_dir", default="results/",
                   help="Directory where this run's outputs are written (model subdir created inside).")
    p.add_argument("--splits_dir", default=None,
                   help="Directory containing splits.npz and data_config.json from prepare_splits.py. "
                        "Defaults to --results_dir when not set (single-experiment mode).")
    p.add_argument("--tune", action="store_true")
    p.add_argument("--prs_col", default=None,
                   help="Column name for PRS in the covariates file. "
                        "If provided it is included as a covariate by default. "
                        "Omit to exclude PRS from the model.")
    p.add_argument("--pathway_cols", nargs="+", default=None,
                   help="Pathway matrix columns to use as structured pathway input. "
                        "Accepts explicit column names or a path to a text file (one name per line). "
                        "Default: all pathway columns. Pass an empty string or omit to run "
                        "covariates-only (no pathway branch).")
    p.add_argument("--covariate_cols", nargs="+", default=None,
                   help="Covariate columns to pass through the flat covariate encoder. "
                        "Default: all columns in the covariates file except the label. "
                        "PRS is included here (not in the pathway matrix) when --prs_col is set.")
    p.add_argument("--covariates_file", default=None,
                   help="Override the covariates CSV path from data_config.json. "
                        "Must have the same row order as the original covariates file.")
    p.add_argument("--graph_method", default="jaccard",
                   choices=["jaccard", "score_correlation", "string", "fully_connected"],
                   help="GNN only. How to construct pathway graph edges.")
    p.add_argument("--string_edges", default=None,
                   help="GNN + graph_method=string: CSV with columns gene1,gene2,confidence (HGNC symbols).")
    p.add_argument("--pathway_gene_sets", default=None,
                   help="GNN + graph_method in {jaccard,string}: JSON mapping prefixed pathway name → list of gene symbols.")
    p.add_argument("--jaccard_min_overlap", type=int, default=3,
                   help="GNN + graph_method=jaccard: minimum shared genes for an edge.")
    p.add_argument("--jaccard_min_score", type=float, default=0.1,
                   help="GNN + graph_method=jaccard: minimum Jaccard similarity for an edge.")
    p.add_argument("--corr_threshold", type=float, default=0.3,
                   help="GNN + graph_method=score_correlation: absolute Pearson threshold for an edge.")
    p.add_argument("--bootstrap_iters", type=int, default=1000)
    return p.parse_args()


def _get_all_pw_names(splits_dir):
    with open(os.path.join(splits_dir, "data_config.json")) as f:
        return json.load(f)["pathway_names"]


def _resolve_pathway_cols(args, all_pw_names):
    """
    Resolve --pathway_cols to a list of column names.
    Returns an empty list for covariates-only runs (K=0).
    Accepts explicit column names or a single path to a text file.
    """
    if args.pathway_cols is None:
        return all_pw_names                          # default: all pathways

    if args.pathway_cols == [""]:
        return []                                    # explicit covariates-only

    if len(args.pathway_cols) == 1 and (args.pathway_cols[0].endswith(".txt") or os.sep in args.pathway_cols[0]):
        p = args.pathway_cols[0]
        if not os.path.isfile(p):
            raise FileNotFoundError(f"--pathway_cols file not found: {p!r}")
        with open(p) as f:
            requested = [l.strip() for l in f if l.strip()]
    else:
        requested = args.pathway_cols

    unknown = set(requested) - set(all_pw_names)
    if unknown:
        raise ValueError(f"Unknown pathway columns: {sorted(unknown)}")
    return requested


# ── Data loading ──────────────────────────────────────────────────────────────

def load_splits(splits_dir, pathway_cols=None, covariate_cols=None, covariates_file=None):
    config_path = os.path.join(splits_dir, "data_config.json")
    splits_path = os.path.join(splits_dir, "splits.npz")
    if not os.path.exists(config_path) or not os.path.exists(splits_path):
        raise FileNotFoundError(f"Run prepare_splits.py first (missing files in {splits_dir})")

    with open(config_path) as f:
        config = json.load(f)

    pw, cov, labels, pw_names, cov_cols_all, _ = load_data(
        config["pathway_matrix"], covariates_file or config["covariates"],
        config["label_col"], covariate_cols or config["covariate_cols"],
    )

    if pathway_cols is not None:
        if len(pathway_cols) == 0:
            pw       = np.empty((pw.shape[0], 0), dtype=np.float32)
            pw_names = []
        else:
            idx      = [pw_names.index(c) for c in pathway_cols]
            pw       = pw[:, idx] if pw.ndim == 2 else pw[:, idx, :]
            pw_names = pathway_cols

    splits = np.load(splits_path)
    return pw, cov, labels, pw_names, cov_cols_all, splits["idx_train"], splits["idx_val"], splits["idx_test"]


def _make_loaders(pw, cov, labels, idx_tr, idx_va, idx_te):
    return make_loaders(
        pw[idx_tr], cov[idx_tr], labels[idx_tr],
        pw[idx_va], cov[idx_va], labels[idx_va],
        pw[idx_te], cov[idx_te], labels[idx_te],
        batch_size=TRAIN_CFG["batch_size"],
    )


# ── Output saving ─────────────────────────────────────────────────────────────

class _NumpyEncoder(json.JSONEncoder):
    """Convert numpy scalars/arrays to plain Python types for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_outputs(out_dir, probs, labels_test, ranking, best_hparams,
                 search_results=None, bootstrap_iters=1000, covariate_ranking=None):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "probs_test.npy"), probs)
    np.save(os.path.join(out_dir, "labels_test.npy"), labels_test)

    metrics = compute_metrics(labels_test, probs)
    ci = bootstrap_metrics(labels_test, probs, n_iterations=bootstrap_iters)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"metrics": metrics, "bootstrap_95ci": ci}, f, indent=2, cls=_NumpyEncoder)

    combined_ranking = {**ranking, **(covariate_ranking or {})}
    with open(os.path.join(out_dir, "ranking.json"), "w") as f:
        json.dump(combined_ranking, f, indent=2, cls=_NumpyEncoder)
    with open(os.path.join(out_dir, "best_hparams.json"), "w") as f:
        json.dump(best_hparams, f, indent=2, cls=_NumpyEncoder)
    if search_results is not None:
        with open(os.path.join(out_dir, "hparam_search.json"), "w") as f:
            json.dump(search_results, f, indent=2, cls=_NumpyEncoder)

    auroc = metrics["auroc"]
    ci_auroc = ci["auroc"]
    print(f"  AUROC={auroc:.4f} [{ci_auroc['ci_lower']:.4f}, {ci_auroc['ci_upper']:.4f}]  "
          f"AUPRC={metrics['auprc']:.4f}  Brier={metrics['brier_score']:.4f}")


# ── Sklearn grid search helper ────────────────────────────────────────────────

def sklearn_grid_search(build_fn, grid, train_X, train_y, val_X, val_y):
    from sklearn.metrics import roc_auc_score
    candidates = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    results, best_auroc, best_params = [], -1.0, {}
    for i, params in enumerate(candidates):
        m = build_fn(**params)
        m.fit(train_X, train_y)
        auroc = roc_auc_score(val_y, m.predict_proba(val_X)[:, 1])
        results.append({"params": params, "val_auroc": auroc})
        print(f"  [{i+1}/{len(candidates)}] {params} -> val_auroc={auroc:.4f}")
        if auroc > best_auroc:
            best_auroc, best_params = auroc, params
    results.sort(key=lambda x: -x["val_auroc"])
    print(f"  Best: {best_params} (val_auroc={best_auroc:.4f})")
    return best_params, results


# ── Sklearn runners ───────────────────────────────────────────────────────────


def run_elasticnet(args, pw, cov, labels, pw_names, cov_cols, idx_tr, idx_va, idx_te, out_dir, **_):
    """Elasticnet logistic regression on flattened pathway matrix + covariates."""
    T = pw.shape[2] if pw.ndim == 3 else 1
    flat = {s: flatten_pathway_matrix(pw[i]) for s, i in [("tr", idx_tr), ("va", idx_va), ("te", idx_te)]}
    X = {s: np.concatenate([flat[s], cov[i]], axis=1)
         for s, i in [("tr", idx_tr), ("va", idx_va), ("te", idx_te)]}

    final_X = np.concatenate([X["tr"], X["va"]])
    final_y = np.concatenate([labels[idx_tr], labels[idx_va]])
    model = build_elasticnet()
    model.fit(final_X, final_y)
    best_C = float(np.atleast_1d(model.named_steps["clf"].C_)[0])
    best_l1_ratio = float(np.atleast_1d(model.named_steps["clf"].l1_ratio_)[0])
    print(f"  Best C (internal CV): {best_C:.5g}, l1_ratio: {best_l1_ratio:.3g}")
    save_model(model, os.path.join(out_dir, "elasticnet.pkl"))
    probs = model.predict_proba(X["te"])[:, 1]
    ranking, cov_ranking = get_l1_pathway_ranking(model, pw_names, T, cov_cols) if pw_names else ({}, {})
    save_outputs(out_dir, probs, labels[idx_te], ranking,
                 {"C": best_C, "l1_ratio": best_l1_ratio}, None, args.bootstrap_iters,
                 covariate_ranking=cov_ranking)


def run_l1_logistic(args, pw, cov, labels, pw_names, cov_cols, idx_tr, idx_va, idx_te, out_dir, **_):
    T = pw.shape[2] if pw.ndim == 3 else 1
    flat = {s: flatten_pathway_matrix(pw[i]) for s, i in [("tr", idx_tr), ("va", idx_va), ("te", idx_te)]}
    X = {s: np.concatenate([flat[s], cov[i]], axis=1)
         for s, i in [("tr", idx_tr), ("va", idx_va), ("te", idx_te)]}

    # Refit on train+val; internal 5-fold CV selects C within that set.
    final_X = np.concatenate([X["tr"], X["va"]])
    final_y = np.concatenate([labels[idx_tr], labels[idx_va]])
    model = build_l1_logistic()
    model.fit(final_X, final_y)
    best_C = float(np.atleast_1d(model.named_steps["clf"].C_)[0])
    print(f"  Best C (internal CV): {best_C:.5g}")
    save_model(model, os.path.join(out_dir, "l1_logistic.pkl"))
    probs = model.predict_proba(X["te"])[:, 1]
    ranking, cov_ranking = get_l1_pathway_ranking(model, pw_names, T, cov_cols)
    save_outputs(out_dir, probs, labels[idx_te], ranking, {"C": best_C}, None, args.bootstrap_iters,
                 covariate_ranking=cov_ranking)


def run_random_forest(args, pw, cov, labels, pw_names, cov_cols, idx_tr, idx_va, idx_te, out_dir, **_):
    T = pw.shape[2] if pw.ndim == 3 else 1
    pw, cov = _scale_splits(pw, cov, idx_tr, idx_va, idx_te)
    flat = {s: flatten_pathway_matrix(pw[i]) for s, i in [("tr", idx_tr), ("va", idx_va), ("te", idx_te)]}
    X = {s: np.concatenate([flat[s], cov[i]], axis=1)
         for s, i in [("tr", idx_tr), ("va", idx_va), ("te", idx_te)]}

    if args.tune:
        best_hparams, search_results = sklearn_grid_search(
            build_random_forest, HPARAM_GRIDS["random_forest"],
            X["tr"], labels[idx_tr], X["va"], labels[idx_va],
        )
    else:
        best_hparams, search_results = DEFAULTS["random_forest"], None

    final_X = np.concatenate([X["tr"], X["va"]])
    final_y = np.concatenate([labels[idx_tr], labels[idx_va]])
    model = build_random_forest(**best_hparams)
    model.fit(final_X, final_y)
    save_model(model, os.path.join(out_dir, "random_forest.pkl"))
    probs = model.predict_proba(X["te"])[:, 1]
    ranking, cov_ranking = get_rf_pathway_ranking(model, pw_names, T, cov_cols)
    save_outputs(out_dir, probs, labels[idx_te], ranking, best_hparams, search_results, args.bootstrap_iters,
                 covariate_ranking=cov_ranking)


# ── Neural runners ────────────────────────────────────────────────────────────

@torch.no_grad()

def _scale_splits(pw, cov, idx_tr, idx_va, idx_te):
    """Fit StandardScaler on train, apply to val/test for both pathway and covariate arrays."""
    # Covariates: (N, C)
    cov_scaler = StandardScaler().fit(cov[idx_tr])
    cov_scaled = cov.copy().astype(np.float32)
    for idx in (idx_tr, idx_va, idx_te):
        cov_scaled[idx] = cov_scaler.transform(cov[idx])

    # Pathway matrix: (N, K, T) or (N, K) — reshape to 2D, scale, reshape back
    pw_scaled = pw.copy().astype(np.float32)
    if pw.shape[1] > 0:
        orig_shape = pw.shape[1:]
        pw_2d = pw.reshape(len(pw), -1)
        pw_scaler = StandardScaler().fit(pw_2d[idx_tr])
        for idx in (idx_tr, idx_va, idx_te):
            pw_scaled[idx] = pw_scaler.transform(pw_2d[idx]).reshape(-1, *orig_shape)

    return pw_scaled, cov_scaled


def run_neural(args, model_cls, model_kwargs, train_kwargs,
               pw, cov, labels, pw_names, idx_tr, idx_va, idx_te,
               out_dir, weight_file, post_fn=None):
    """Shared training loop for all three neural models."""
    pw, cov = _scale_splits(pw, cov, idx_tr, idx_va, idx_te)
    train_loader, val_loader, test_loader = _make_loaders(pw, cov, labels, idx_tr, idx_va, idx_te)

    if args.tune:
        grid = HPARAM_GRIDS[args.model]
        candidates = [{**model_kwargs, **dict(zip(grid, v))}
                      for v in itertools.product(*grid.values())]
        print(f"Tuning {args.model} over {len(candidates)} candidates...")
        best_model_kw, best_train_kw, best_auroc, search_results = tune_hyperparameters(
            model_cls=model_cls,
            model_kwargs_grid=candidates,
            train_loader=train_loader,
            val_loader=val_loader,
            train_labels=labels[idx_tr],
            device="cpu",
            **TUNE_CFG,
        )
        best_hparams = {**{k: v for k, v in best_model_kw.items() if k not in model_kwargs},
                        **best_train_kw}
        search_results_out = search_results
    else:
        defaults = DEFAULTS[args.model]
        best_model_kw = {**model_kwargs, **{k: v for k, v in defaults.items()
                                             if k not in {"lr", "weight_decay"}}}
        best_train_kw = {k: defaults[k] for k in ("lr", "weight_decay") if k in defaults}
        best_hparams = defaults
        search_results_out = None

    model = model_cls(**best_model_kw)
    model, _ = train(
        model, train_loader, val_loader, labels[idx_tr],
        n_epochs=TRAIN_CFG["n_epochs"], patience=TRAIN_CFG["patience"],
        device="cpu", **best_train_kw,
    )
    torch.save(model.state_dict(), os.path.join(out_dir, weight_file))

    # Platt scaling: calibrate probabilities to true prevalence using val set
    # (WeightedRandomSampler trains on balanced batches → raw sigmoid overestimates P(Y=1))
    platt_scaler = fit_platt_scaler(model, val_loader, device="cpu")
    probs, _ = predict_calibrated(model, test_loader, device="cpu", platt_scaler=platt_scaler)

    ranking, extra = post_fn(model, test_loader, pw_names, out_dir) if post_fn else ({}, None)
    save_outputs(out_dir, probs, labels[idx_te], ranking, best_hparams,
                 search_results_out, args.bootstrap_iters)


def _dims(pw, cov):
    K = pw.shape[1]
    T = pw.shape[2] if pw.ndim == 3 else (1 if K > 0 else 0)
    C = cov.shape[1]
    return K, T, C


def run_global_attention(args, pw, cov, labels, pw_names, cov_cols, idx_tr, idx_va, idx_te, out_dir, **_):
    K, T, C = _dims(pw, cov)
    fixed = dict(n_pathways=K, pathway_input_dim=T, covariate_dim=C)

    def post(model, test_loader, pw_names, out_dir):
        ranking = get_attention_pathway_ranking(model, pw_names)
        weights = model.get_attention_weights()
        if weights is not None:
            np.save(os.path.join(out_dir, "attention_weights.npy"), weights.detach().cpu().numpy())
        return ranking, None

    run_neural(args, GlobalPathwayAttentionModel, fixed, {},
               pw, cov, labels, pw_names, idx_tr, idx_va, idx_te,
               out_dir, "global_attention.pt", post_fn=post)


def run_transformer(args, pw, cov, labels, pw_names, cov_cols, idx_tr, idx_va, idx_te, out_dir, **_):
    K, T, C = _dims(pw, cov)
    fixed = dict(n_pathways=K, pathway_input_dim=T, covariate_dim=C, n_layers=2)

    def post(model, test_loader, pw_names, out_dir):
        mean_attn = model.compute_mean_attention(test_loader, device="cpu")
        if mean_attn is None:
            return {}, None
        np.save(os.path.join(out_dir, "mean_attention.npy"), mean_attn.numpy())
        ranking = {name: float(mean_attn[-1, :, 0, i + 1].mean().item())
                   for i, name in enumerate(pw_names)}
        return ranking, None

    run_neural(args, PathwayTransformer, fixed, {},
               pw, cov, labels, pw_names, idx_tr, idx_va, idx_te,
               out_dir, "transformer.pt", post_fn=post)


def run_gnn(args, pw, cov, labels, pw_names, cov_cols, idx_tr, idx_va, idx_te, out_dir, **_):
    import pandas as pd
    K, T, C = _dims(pw, cov)
    method = args.graph_method

    print(f"  Graph method: {method}")

    if method == "jaccard":
        if not args.pathway_gene_sets:
            raise ValueError("--graph_method jaccard requires --pathway_gene_sets")
        with open(args.pathway_gene_sets) as f:
            gene_sets = {k: set(v) for k, v in json.load(f).items()}
        edge_index = build_jaccard_edge_index(
            gene_sets, pw_names,
            min_overlap=args.jaccard_min_overlap,
            min_jaccard=args.jaccard_min_score,
        )

    elif method == "score_correlation":
        edge_index = build_score_correlation_edge_index(
            pw[idx_tr], pw_names, threshold=args.corr_threshold,
        )

    elif method == "string":
        if not args.string_edges or not args.pathway_gene_sets:
            raise ValueError("--graph_method string requires --string_edges and --pathway_gene_sets")
        with open(args.pathway_gene_sets) as f:
            gene_sets = {k: set(v) for k, v in json.load(f).items()}
        ppi_df = pd.read_csv(args.string_edges)
        edge_index = build_string_edge_index(
            gene_sets, list(zip(ppi_df.gene1, ppi_df.gene2, ppi_df.confidence)),
            pathway_names=pw_names,
        )

    elif method == "fully_connected":
        edge_index = build_fully_connected_edge_index(K)

    else:
        raise ValueError(f"Unknown graph method: {method}")

    n_edges = edge_index.shape[1] // 2
    print(f"  Graph: {K} nodes, {n_edges} undirected edges "
          f"({n_edges / max(K*(K-1)//2, 1)*100:.1f}% of possible)")

    # Save graph structure and pathway names for downstream plotting
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "edge_index.npy"), edge_index.numpy())
    with open(os.path.join(out_dir, "graph_info.json"), "w") as f:
        json.dump({"graph_method": method, "n_nodes": K, "n_edges": n_edges,
                   "pathway_names": pw_names}, f, indent=2)

    fixed = dict(n_pathways=K, pathway_input_dim=T, covariate_dim=C, edge_index=edge_index)

    def post(model, test_loader, pw_names, out_dir):
        ranking = compute_gradcam_pathway_ranking(model, test_loader, pw_names, device="cpu")
        return ranking, None

    run_neural(args, PathwayGNN, fixed, {},
               pw, cov, labels, pw_names, idx_tr, idx_va, idx_te,
               out_dir, "gnn.pt", post_fn=post)


# ── Dispatch ──────────────────────────────────────────────────────────────────

RUNNERS = {
"l1_logistic":         run_l1_logistic,
    "elasticnet":          run_elasticnet,
    "random_forest":       run_random_forest,
    "global_attention":    run_global_attention,
    "transformer":         run_transformer,
    "gnn":                 run_gnn,
}


def _fmt(seconds):
    """Format elapsed seconds as m:ss or s."""
    if seconds >= 60:
        return f"{int(seconds)//60}m {int(seconds)%60:02d}s"
    return f"{seconds:.1f}s"


def _step(label, model_name):
    """Print a timestamped progress line and return the start time."""
    msg = f"[{model_name}] {label}..."
    print(msg, flush=True)
    return time.time()


def _done(label, model_name, t0):
    elapsed = time.time() - t0
    print(f"[{model_name}] {label} done ({_fmt(elapsed)})", flush=True)


def main():
    args = parse_args()
    splits_dir = args.splits_dir or args.results_dir
    torch.manual_seed(42)
    np.random.seed(42)

    wall_start = time.time()

    t = _step("loading data", args.model)
    pw, cov, labels, pw_names, cov_cols, idx_tr, idx_va, idx_te = load_splits(
        splits_dir,
        pathway_cols=_resolve_pathway_cols(args, _get_all_pw_names(splits_dir)),
        covariate_cols=args.covariate_cols,
        covariates_file=args.covariates_file,
    )
    _done("loading data", args.model, t)

    K, T, C = _dims(pw, cov) if pw.ndim >= 2 else (0, 0, cov.shape[1])
    feature_desc = f"{K} pathways (T={T}), {C} covariates"
    print(f"[{args.model}] tune={args.tune} | features: {feature_desc} | "
          f"train={len(idx_tr)}, val={len(idx_va)}, test={len(idx_te)}", flush=True)

    out_dir = os.path.join(args.results_dir, args.model)
    os.makedirs(out_dir, exist_ok=True)

    t = _step("training", args.model)
    RUNNERS[args.model](args, pw, cov, labels, pw_names, cov_cols, idx_tr, idx_va, idx_te, out_dir)
    _done("training", args.model, t)

    total = time.time() - wall_start
    print(f"[{args.model}] finished -> {out_dir}/  (total: {_fmt(total)})", flush=True)


if __name__ == "__main__":
    main()
