import argparse
import itertools
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trainer import (load_data, make_loaders, train, tune_hyperparameters,
                         fit_platt_scaler, predict_calibrated)
from src.models.baseline import (
    build_l1_logistic, build_elasticnet, build_logistic,
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
    compute_gradcam_pathway_ranking, compute_node_embeddings, compute_gradient_covariate_importance,
)

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

DEFAULTS = {
    "l1_logistic":         {},
    "elasticnet":          {},
    "logistic":               {},
    "random_forest":       {"n_estimators": 500, "max_depth": 12, "min_samples_leaf": 20},
    "global_attention":    {"embed_dim": 128, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4},
    "transformer":         {"embed_dim": 128, "n_heads": 4, "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-4},
    "gnn":                 {"embed_dim": 128, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4},
}

TRAIN_CFG = dict(n_epochs=200, patience=15, batch_size=256)
TUNE_CFG  = dict(n_epochs=60,  patience=8)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(DEFAULTS))
    p.add_argument("--results_dir", default="results/")
    p.add_argument("--splits_dir", default=None)
    p.add_argument("--tune", action="store_true")
    p.add_argument("--prs_col", default=None)
    p.add_argument("--pathway_cols", nargs="+", default=None)
    p.add_argument("--covariate_cols", nargs="+", default=None)
    p.add_argument("--covariates_file", default=None)
    p.add_argument("--graph_method", default="jaccard",
                   choices=["jaccard", "score_correlation", "string", "fully_connected"])
    p.add_argument("--string_edges", default=None)
    p.add_argument("--string_min_shared_genes", type=int, default=5)
    p.add_argument("--string_min_confidence", type=int, default=700)
    p.add_argument("--pathway_gene_sets", default=None)
    p.add_argument("--jaccard_min_overlap", type=int, default=3)
    p.add_argument("--jaccard_min_score", type=float, default=0.1)
    p.add_argument("--corr_threshold", type=float, default=0.3)
    p.add_argument("--corr_top_k", type=int, default=None)
    p.add_argument("--top_k_pathways", type=int, default=None)
    p.add_argument("--feature_selection_method", default="variance", choices=["variance", "auroc"])
    p.add_argument("--bootstrap_iters", type=int, default=1000)
    p.add_argument("--device", default=None)
    return p.parse_args()

def _get_all_pw_names(splits_dir):
    with open(os.path.join(splits_dir, "data_config.json")) as f:
        return json.load(f)["pathway_names"]

def _resolve_pathway_cols(args, all_pw_names):
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

def _select_top_k_pathways(pw, pw_names, labels, idx_tr, k, method="variance"):
    pw_tr = pw[idx_tr]

    if method == "variance":
        raw = pw_tr.var(axis=0)
        if raw.ndim == 2:
            raw = raw.mean(axis=1)

        prefixes = np.array([n.split("__")[0] for n in pw_names])
        unique_prefixes = np.unique(prefixes)

        if len(unique_prefixes) > 1:
            scores = np.zeros(len(pw_names))
            for prefix in unique_prefixes:
                mask = prefixes == prefix
                group = raw[mask]
                ranks = group.argsort().argsort().astype(float)
                scores[mask] = ranks / max(mask.sum() - 1, 1)
            label = f"rank-normalized variance ({', '.join(unique_prefixes)})"
        else:
            scores = raw
            label = "variance"

    elif method == "auroc":
        from sklearn.metrics import roc_auc_score
        y_tr = labels[idx_tr]
        pw_flat = pw_tr if pw_tr.ndim == 2 else pw_tr[:, :, 0]
        scores = np.array([roc_auc_score(y_tr, pw_flat[:, i]) for i in range(pw_flat.shape[1])])
        scores = np.maximum(scores, 1 - scores)
        label = "univariate AUROC"

    else:
        raise ValueError(f"Unknown feature selection method: {method!r}. Use 'variance' or 'auroc'.")

    top_idx = np.sort(np.argsort(scores)[::-1][:k])
    pw_sel  = pw[:, top_idx] if pw.ndim == 2 else pw[:, top_idx, :]
    names_sel = [pw_names[i] for i in top_idx]

    prefixes_sel = np.array([n.split("__")[0] for n in names_sel])
    type_counts = {p: (prefixes_sel == p).sum() for p in np.unique(prefixes_sel)}
    type_str = ", ".join(f"{p}={c}" for p, c in type_counts.items())
    print(f"  [feature selection] kept {k}/{len(pw_names)} pathways by {label} ({type_str})")
    return pw_sel, names_sel

def _make_loaders(pw, cov, labels, idx_tr, idx_va, idx_te):
    return make_loaders(
        pw[idx_tr], cov[idx_tr], labels[idx_tr],
        pw[idx_va], cov[idx_va], labels[idx_va],
        pw[idx_te], cov[idx_te], labels[idx_te],
        batch_size=TRAIN_CFG["batch_size"],
    )

class _NumpyEncoder(json.JSONEncoder):
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

def _platt_scale_sklearn(build_fn, X_tr, y_tr, X_va, y_va):
    from sklearn.linear_model import LogisticRegression as _LR
    cal_model = build_fn()
    cal_model.fit(X_tr, y_tr)
    val_probs = cal_model.predict_proba(X_va)[:, 1]
    eps = 1e-7
    val_logits = np.log(np.clip(val_probs, eps, 1 - eps) / (1 - np.clip(val_probs, eps, 1 - eps)))
    scaler = _LR(C=1e6, solver="lbfgs", max_iter=1000)
    scaler.fit(val_logits.reshape(-1, 1), y_va)
    return scaler

def _apply_platt(scaler, raw_probs):
    eps = 1e-7
    logits = np.log(np.clip(raw_probs, eps, 1 - eps) / (1 - np.clip(raw_probs, eps, 1 - eps)))
    return scaler.predict_proba(logits.reshape(-1, 1))[:, 1]

def _run_sklearn_logistic(args, build_fn, model_name,
                          pw, cov, labels, pw_names, cov_cols,
                          idx_tr, idx_va, idx_te, out_dir, hparams_fn=None):
    T = pw.shape[2] if pw.ndim == 3 else 1
    splits = [("tr", idx_tr), ("va", idx_va), ("te", idx_te)]
    X = {s: np.concatenate([flatten_pathway_matrix(pw[i]), cov[i]], axis=1) for s, i in splits}

    platt = _platt_scale_sklearn(build_fn, X["tr"], labels[idx_tr], X["va"], labels[idx_va])

    final_X = np.concatenate([X["tr"], X["va"]])
    final_y = np.concatenate([labels[idx_tr], labels[idx_va]])
    model = build_fn()
    model.fit(final_X, final_y)
    save_model(model, os.path.join(out_dir, f"{model_name}.pkl"))
    probs = _apply_platt(platt, model.predict_proba(X["te"])[:, 1])
    ranking, cov_ranking = get_l1_pathway_ranking(model, pw_names, T, cov_cols)
    hparams = hparams_fn(model) if hparams_fn else {}
    save_outputs(out_dir, probs, labels[idx_te], ranking, hparams, None, args.bootstrap_iters,
                 covariate_ranking=cov_ranking)

def run_elasticnet(args, pw, cov, labels, pw_names, cov_cols, idx_tr, idx_va, idx_te, out_dir, **_):
    def _hparams(m):
        best_C = float(np.atleast_1d(m.named_steps["clf"].C_)[0])
        best_l1_ratio = float(np.atleast_1d(m.named_steps["clf"].l1_ratio_)[0])
        print(f"  Best C (internal CV): {best_C:.5g}, l1_ratio: {best_l1_ratio:.3g}")
        return {"C": best_C, "l1_ratio": best_l1_ratio}
    _run_sklearn_logistic(args, build_elasticnet, "elasticnet",
                          pw, cov, labels, pw_names, cov_cols,
                          idx_tr, idx_va, idx_te, out_dir, hparams_fn=_hparams)

def run_l1_logistic(args, pw, cov, labels, pw_names, cov_cols, idx_tr, idx_va, idx_te, out_dir, **_):
    def _hparams(m):
        best_C = float(np.atleast_1d(m.named_steps["clf"].C_)[0])
        print(f"  Best C (internal CV): {best_C:.5g}")
        return {"C": best_C}
    _run_sklearn_logistic(args, build_l1_logistic, "l1_logistic",
                          pw, cov, labels, pw_names, cov_cols,
                          idx_tr, idx_va, idx_te, out_dir, hparams_fn=_hparams)

def run_logistic(args, pw, cov, labels, pw_names, cov_cols, idx_tr, idx_va, idx_te, out_dir, **_):
    _run_sklearn_logistic(args, build_logistic, "logistic",
                          pw, cov, labels, pw_names, cov_cols,
                          idx_tr, idx_va, idx_te, out_dir)

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

    platt = _platt_scale_sklearn(lambda: build_random_forest(**best_hparams),
                                 X["tr"], labels[idx_tr], X["va"], labels[idx_va])

    final_X = np.concatenate([X["tr"], X["va"]])
    final_y = np.concatenate([labels[idx_tr], labels[idx_va]])
    model = build_random_forest(**best_hparams)
    model.fit(final_X, final_y)
    save_model(model, os.path.join(out_dir, "random_forest.pkl"))
    probs = _apply_platt(platt, model.predict_proba(X["te"])[:, 1])
    ranking, cov_ranking = get_rf_pathway_ranking(model, pw_names, T, cov_cols)
    save_outputs(out_dir, probs, labels[idx_te], ranking, best_hparams, search_results, args.bootstrap_iters,
                 covariate_ranking=cov_ranking)

@torch.no_grad()

def _scale_splits(pw, cov, idx_tr, idx_va, idx_te):
    cov_scaler = StandardScaler().fit(cov[idx_tr])
    cov_scaled = cov.copy().astype(np.float32)
    for idx in (idx_tr, idx_va, idx_te):
        cov_scaled[idx] = cov_scaler.transform(cov[idx])

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
               out_dir, weight_file, post_fn=None, batch_size=None):
    pw, cov = _scale_splits(pw, cov, idx_tr, idx_va, idx_te)
    bs = batch_size or TRAIN_CFG["batch_size"]
    train_loader, val_loader, test_loader = make_loaders(
        pw[idx_tr], cov[idx_tr], labels[idx_tr],
        pw[idx_va], cov[idx_va], labels[idx_va],
        pw[idx_te], cov[idx_te], labels[idx_te],
        batch_size=bs,
    )

    device = args.device
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
            device=device,
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
        device=device, **best_train_kw,
    )
    torch.save(model.state_dict(), os.path.join(out_dir, weight_file))

    platt_scaler = fit_platt_scaler(model, val_loader, device=device)
    probs, _ = predict_calibrated(model, test_loader, device=device, platt_scaler=platt_scaler)

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
        cov_ranking = compute_gradient_covariate_importance(model, test_loader, cov_cols,
                                                            device=args.device)
        return ranking, cov_ranking

    run_neural(args, GlobalPathwayAttentionModel, fixed, {},
               pw, cov, labels, pw_names, idx_tr, idx_va, idx_te,
               out_dir, "global_attention.pt", post_fn=post)

def run_transformer(args, pw, cov, labels, pw_names, cov_cols, idx_tr, idx_va, idx_te, out_dir, **_):
    K, T, C = _dims(pw, cov)
    fixed = dict(n_pathways=K, pathway_input_dim=T, covariate_dim=C, n_layers=2)

    def post(model, test_loader, pw_names, out_dir):
        mean_attn = model.compute_mean_attention(test_loader, device=args.device)
        if mean_attn is None:
            return {}, None
        np.save(os.path.join(out_dir, "mean_attention.npy"), mean_attn.numpy())
        ranking = {name: float(mean_attn[-1, :, 0, i + 1].mean().item())
                   for i, name in enumerate(pw_names)}
        cov_ranking = compute_gradient_covariate_importance(model, test_loader, cov_cols,
                                                            device=args.device)
        return ranking, cov_ranking

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
            pw[idx_tr], pw_names,
            threshold=args.corr_threshold,
            top_k=args.corr_top_k,
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
            min_shared_genes=args.string_min_shared_genes,
            min_confidence=args.string_min_confidence,
        )

    elif method == "fully_connected":
        edge_index = None  # handled internally by fully_connected=True flag

    else:
        raise ValueError(f"Unknown graph method: {method}")

    if method == "fully_connected":
        n_edges = K * (K - 1) // 2
        print(f"  Graph: {K} nodes, fully connected ({n_edges} theoretical edges, no materialisation)")
    else:
        n_edges = edge_index.shape[1] // 2
        print(f"  Graph: {K} nodes, {n_edges} undirected edges "
              f"({n_edges / max(K*(K-1)//2, 1)*100:.1f}% of possible)")

    # Save graph structure and pathway names for downstream plotting
    os.makedirs(out_dir, exist_ok=True)
    if edge_index is not None:
        np.save(os.path.join(out_dir, "edge_index.npy"), edge_index.numpy())
    with open(os.path.join(out_dir, "graph_info.json"), "w") as f:
        json.dump({"graph_method": method, "n_nodes": K, "n_edges": n_edges,
                   "pathway_names": pw_names}, f, indent=2)

    fixed = dict(n_pathways=K, pathway_input_dim=T, covariate_dim=C,
                 edge_index=edge_index, fully_connected=(method == "fully_connected"))

    def post(model, test_loader, pw_names, out_dir):
        ranking, per_sample_gradcam = compute_gradcam_pathway_ranking(
            model, test_loader, pw_names, device=args.device
        )
        np.save(os.path.join(out_dir, "gradcam_scores.npy"), per_sample_gradcam)
        node_embeddings = compute_node_embeddings(model, test_loader, device=args.device)
        np.save(os.path.join(out_dir, "node_embeddings.npy"), node_embeddings)
        return ranking, None

    # batch_size=32: GNN materialises the full edge set per sample in the batch;
    # at K=1654 the default batch of 256 exhausts RAM
    run_neural(args, PathwayGNN, fixed, {},
               pw, cov, labels, pw_names, idx_tr, idx_va, idx_te,
               out_dir, "gnn.pt", post_fn=post, batch_size=32)

RUNNERS = {
    "l1_logistic":         run_l1_logistic,
    "elasticnet":          run_elasticnet,
    "logistic":               run_logistic,
    "random_forest":       run_random_forest,
    "global_attention":    run_global_attention,
    "transformer":         run_transformer,
    "gnn":                 run_gnn,
}

def _fmt(seconds):
    if seconds >= 60:
        return f"{int(seconds)//60}m {int(seconds)%60:02d}s"
    return f"{seconds:.1f}s"

def _step(label, model_name):
    msg = f"[{model_name}] {label}..."
    print(msg, flush=True)
    return time.time()

def _done(label, model_name, t0):
    elapsed = time.time() - t0
    print(f"[{model_name}] {label} done ({_fmt(elapsed)})", flush=True)

def _autodevice(requested):
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def main():
    args = parse_args()
    args.device = _autodevice(args.device)
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

    if args.prs_col is not None:
        splits_dir = args.splits_dir or args.results_dir
        with open(os.path.join(splits_dir, "data_config.json")) as f:
            _cfg = json.load(f)
        _cov_path = args.covariates_file or _cfg["covariates"]
        _cov_df = pd.read_csv(_cov_path) if _cov_path.endswith(".csv") else pd.read_parquet(_cov_path)
        if args.prs_col not in _cov_df.columns:
            raise ValueError(f"--prs_col {args.prs_col!r} not found in {_cov_path}. "
                             f"Available columns: {list(_cov_df.columns)}")
        prs_vals = _cov_df[args.prs_col].values.astype(np.float32).reshape(-1, 1)
        cov = np.concatenate([cov, prs_vals], axis=1)
        cov_cols = list(cov_cols) + [args.prs_col]
        print(f"  [prs] appended {args.prs_col} as covariate (C={cov.shape[1]})", flush=True)

    if args.top_k_pathways is not None and pw.shape[1] > args.top_k_pathways:
        pw, pw_names = _select_top_k_pathways(
            pw, pw_names, labels, idx_tr, args.top_k_pathways, args.feature_selection_method)

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
