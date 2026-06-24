"""
Train a single model. Run one process per model, then aggregate_results.py.

Usage:
  python scripts/train_model.py \
      --model {prs_logistic,l1_logistic,random_forest,global_attention,transformer,gnn} \
      --results_dir results/ \
      [--tune] \
      [--prs_col prs] \
      [--string_edges data/string_edges.csv --pathway_gene_sets data/pathway_gene_sets.json] \
      [--reactome_edges data/reactome_hierarchy.csv]

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
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trainer import load_data, make_loaders, train, tune_hyperparameters
from src.models.baseline import (
    build_prs_logistic, build_l1_logistic, build_random_forest,
    flatten_pathway_matrix, save_model,
)
from src.models.global_attention import GlobalPathwayAttentionModel
from src.models.pathway_transformer import PathwayTransformer
from src.models.pathway_gnn import PathwayGNN, build_string_edge_index, build_reactome_edge_index
from src.metrics import compute_metrics, bootstrap_metrics
from src.interpretability import (
    get_attention_pathway_ranking, get_l1_pathway_ranking, get_rf_pathway_ranking,
    compute_gradcam_pathway_ranking,
)

# ── Hyperparameter grids (Cartesian product of each list) ────────────────────
# Training params (lr, weight_decay) are separated from model constructor params
# automatically in tune_hyperparameters().

HPARAM_GRIDS = {
    "prs_logistic": {
        "C": [0.01, 0.1, 1.0, 10.0],
    },
    "random_forest": {
        "n_estimators": [200, 500],
        "max_depth":    [4, 6, 8],
        "min_samples_leaf": [20, 50, 100],
    },
    # l1_logistic uses internal LogisticRegressionCV — no outer grid.
    "global_attention": {
        "embed_dim":    [32, 64, 128],
        "dropout":      [0.1, 0.2],
        "lr":           [1e-3, 5e-4],
        "weight_decay": [1e-4, 1e-3],
    },
    "transformer": {
        "embed_dim":    [32, 64, 128],
        "n_heads":      [2, 4],
        "dropout":      [0.1, 0.2],
        "lr":           [1e-3, 5e-4],
        "weight_decay": [1e-4, 1e-3],
    },
    "gnn": {
        "embed_dim":    [32, 64, 128],
        "dropout":      [0.1, 0.2],
        "lr":           [1e-3, 5e-4],
        "weight_decay": [1e-4, 1e-3],
    },
}

# Default hyperparameters used when --tune is not set.
DEFAULTS = {
    "prs_logistic":    {"C": 1.0},
    "l1_logistic":     {},
    "random_forest":   {"n_estimators": 500, "max_depth": 6, "min_samples_leaf": 50},
    "global_attention":{"embed_dim": 64, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4},
    "transformer":     {"embed_dim": 64, "n_heads": 4,   "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4},
    "gnn":             {"embed_dim": 64, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4},
}

# Final training settings (not tuned).
TRAIN_CFG = dict(n_epochs=200, patience=15, batch_size=256)
TUNE_CFG  = dict(n_epochs=60,  patience=8)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(DEFAULTS))
    p.add_argument("--results_dir", default="results/")
    p.add_argument("--tune", action="store_true")
    p.add_argument("--prs_col", default=None)
    p.add_argument("--string_edges", default=None)
    p.add_argument("--reactome_edges", default=None)
    p.add_argument("--pathway_gene_sets", default=None)
    p.add_argument("--bootstrap_iters", type=int, default=1000)
    return p.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_splits(results_dir):
    config_path = os.path.join(results_dir, "data_config.json")
    splits_path = os.path.join(results_dir, "splits.npz")
    if not os.path.exists(config_path) or not os.path.exists(splits_path):
        raise FileNotFoundError(f"Run prepare_splits.py first (missing files in {results_dir})")

    with open(config_path) as f:
        config = json.load(f)

    pw, cov, labels, pw_names, cov_cols, _ = load_data(
        config["pathway_matrix"], config["covariates"],
        config["label_col"], config["covariate_cols"],
    )
    splits = np.load(splits_path)
    return pw, cov, labels, pw_names, cov_cols, splits["idx_train"], splits["idx_val"], splits["idx_test"], config


def _make_loaders(pw, cov, labels, idx_tr, idx_va, idx_te):
    return make_loaders(
        pw[idx_tr], cov[idx_tr], labels[idx_tr],
        pw[idx_va], cov[idx_va], labels[idx_va],
        pw[idx_te], cov[idx_te], labels[idx_te],
        batch_size=TRAIN_CFG["batch_size"],
    )


# ── Output saving ─────────────────────────────────────────────────────────────

def save_outputs(out_dir, probs, labels_test, ranking, best_hparams,
                 search_results=None, bootstrap_iters=1000):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "probs_test.npy"), probs)
    np.save(os.path.join(out_dir, "labels_test.npy"), labels_test)

    metrics = compute_metrics(labels_test, probs)
    ci = bootstrap_metrics(labels_test, probs, n_iterations=bootstrap_iters)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"metrics": metrics, "bootstrap_95ci": ci}, f, indent=2)

    with open(os.path.join(out_dir, "ranking.json"), "w") as f:
        json.dump(ranking, f, indent=2)
    with open(os.path.join(out_dir, "best_hparams.json"), "w") as f:
        json.dump(best_hparams, f, indent=2)
    if search_results is not None:
        with open(os.path.join(out_dir, "hparam_search.json"), "w") as f:
            json.dump(search_results, f, indent=2)

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

def run_prs_logistic(args, pw, cov, labels, pw_names, cov_cols, idx_tr, idx_va, idx_te, out_dir, **_):
    if args.prs_col and args.prs_col in cov_cols:
        prs_i = cov_cols.index(args.prs_col)
        slices = {s: np.column_stack([cov[i, prs_i], cov[i]])
                  for s, i in [("tr", idx_tr), ("va", idx_va), ("te", idx_te)]}
    else:
        slices = {"tr": cov[idx_tr], "va": cov[idx_va], "te": cov[idx_te]}

    if args.tune:
        best_hparams, search_results = sklearn_grid_search(
            build_prs_logistic, HPARAM_GRIDS["prs_logistic"],
            slices["tr"], labels[idx_tr], slices["va"], labels[idx_va],
        )
    else:
        best_hparams, search_results = DEFAULTS["prs_logistic"], None

    final_X = np.concatenate([slices["tr"], slices["va"]])
    final_y = np.concatenate([labels[idx_tr], labels[idx_va]])
    model = build_prs_logistic(**best_hparams)
    model.fit(final_X, final_y)
    save_model(model, os.path.join(out_dir, "prs_logistic.pkl"))
    probs = model.predict_proba(slices["te"])[:, 1]
    save_outputs(out_dir, probs, labels[idx_te], {}, best_hparams, search_results, args.bootstrap_iters)


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
    best_C = float(model.named_steps["clf"].C_[0])
    print(f"  Best C (internal CV): {best_C:.5g}")
    save_model(model, os.path.join(out_dir, "l1_logistic.pkl"))
    probs = model.predict_proba(X["te"])[:, 1]
    ranking = get_l1_pathway_ranking(model, pw_names, T)
    save_outputs(out_dir, probs, labels[idx_te], ranking, {"C": best_C}, None, args.bootstrap_iters)


def run_random_forest(args, pw, cov, labels, pw_names, cov_cols, idx_tr, idx_va, idx_te, out_dir, **_):
    T = pw.shape[2] if pw.ndim == 3 else 1
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
    ranking = get_rf_pathway_ranking(model, pw_names, T)
    save_outputs(out_dir, probs, labels[idx_te], ranking, best_hparams, search_results, args.bootstrap_iters)


# ── Neural runners ────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, loader):
    model.eval()
    probs = []
    for batch in loader:
        logits = model(batch["pathway_features"], batch["covariates"])
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def run_neural(args, model_cls, model_kwargs, train_kwargs,
               pw, cov, labels, pw_names, idx_tr, idx_va, idx_te,
               out_dir, weight_file, post_fn=None):
    """Shared training loop for all three neural models."""
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

    probs = predict(model, test_loader)
    ranking, extra = post_fn(model, test_loader, pw_names, out_dir) if post_fn else ({}, None)
    save_outputs(out_dir, probs, labels[idx_te], ranking, best_hparams,
                 search_results_out, args.bootstrap_iters)


def run_global_attention(args, pw, cov, labels, pw_names, cov_cols, idx_tr, idx_va, idx_te, out_dir, config, **_):
    K, T, C = config["n_pathways"], config["pathway_input_dim"], config["covariate_dim"]
    fixed = dict(n_pathways=K, pathway_input_dim=T, covariate_dim=C)

    def post(model, test_loader, pw_names, out_dir):
        return get_attention_pathway_ranking(model, pw_names), None

    run_neural(args, GlobalPathwayAttentionModel, fixed, {},
               pw, cov, labels, pw_names, idx_tr, idx_va, idx_te,
               out_dir, "global_attention.pt", post_fn=post)


def run_transformer(args, pw, cov, labels, pw_names, cov_cols, idx_tr, idx_va, idx_te, out_dir, config, **_):
    K, T, C = config["n_pathways"], config["pathway_input_dim"], config["covariate_dim"]
    fixed = dict(n_pathways=K, pathway_input_dim=T, covariate_dim=C, n_layers=2)

    def post(model, test_loader, pw_names, out_dir):
        mean_attn = model.compute_mean_attention(test_loader, device="cpu")
        np.save(os.path.join(out_dir, "mean_attention.npy"), mean_attn.numpy())
        ranking = {name: float(mean_attn[-1, :, 0, i + 1].mean().item())
                   for i, name in enumerate(pw_names)}
        return ranking, None

    run_neural(args, PathwayTransformer, fixed, {},
               pw, cov, labels, pw_names, idx_tr, idx_va, idx_te,
               out_dir, "transformer.pt", post_fn=post)


def run_gnn(args, pw, cov, labels, pw_names, cov_cols, idx_tr, idx_va, idx_te, out_dir, config, **_):
    import pandas as pd
    K, T, C = config["n_pathways"], config["pathway_input_dim"], config["covariate_dim"]

    if args.string_edges and args.pathway_gene_sets:
        with open(args.pathway_gene_sets) as f:
            gene_sets = {k: set(v) for k, v in json.load(f).items()}
        ppi_df = pd.read_csv(args.string_edges)
        edge_index = build_string_edge_index(
            gene_sets, list(zip(ppi_df.gene1, ppi_df.gene2, ppi_df.confidence)),
            pathway_names=pw_names,
        )
    elif args.reactome_edges:
        react_df = pd.read_csv(args.reactome_edges)
        edge_index = build_reactome_edge_index(
            list(zip(react_df.parent, react_df.child)), pw_names)
    else:
        raise ValueError("GNN requires --string_edges + --pathway_gene_sets, or --reactome_edges")

    print(f"  Graph: {K} nodes, {edge_index.shape[1]//2} undirected edges")
    fixed = dict(n_pathways=K, pathway_input_dim=T, covariate_dim=C, edge_index=edge_index)

    def post(model, test_loader, pw_names, out_dir):
        ranking = compute_gradcam_pathway_ranking(model, test_loader, pw_names, device="cpu")
        return ranking, None

    run_neural(args, PathwayGNN, fixed, {},
               pw, cov, labels, pw_names, idx_tr, idx_va, idx_te,
               out_dir, "gnn.pt", post_fn=post)


# ── Dispatch ──────────────────────────────────────────────────────────────────

RUNNERS = {
    "prs_logistic":    run_prs_logistic,
    "l1_logistic":     run_l1_logistic,
    "random_forest":   run_random_forest,
    "global_attention":run_global_attention,
    "transformer":     run_transformer,
    "gnn":             run_gnn,
}


def main():
    args = parse_args()
    torch.manual_seed(42)
    np.random.seed(42)

    pw, cov, labels, pw_names, cov_cols, idx_tr, idx_va, idx_te, config = load_splits(args.results_dir)
    out_dir = os.path.join(args.results_dir, args.model)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[{args.model}] tune={args.tune} | "
          f"train={len(idx_tr)}, val={len(idx_va)}, test={len(idx_te)}")

    RUNNERS[args.model](
        args, pw, cov, labels, pw_names, cov_cols,
        idx_tr, idx_va, idx_te, out_dir, config=config,
    )
    print(f"[{args.model}] done -> {out_dir}/")


if __name__ == "__main__":
    main()
