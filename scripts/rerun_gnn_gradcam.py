"""
Rerun GNN GradCAM attribution only — no retraining.

Loads saved gnn.pt weights from a completed run, reconstructs the graph and
test DataLoader, then regenerates:
  - ranking.json          (pathway GradCAM scores)
  - gradcam_scores.npy    (per-sample scores, shape [N_test, K])
  - node_embeddings.npy   (per-sample node embeddings)

ranking.json is overwritten in-place, so the existing metrics.json / probs_test.npy
/ best_hparams.json are untouched — AUROC numbers do not change.

Usage (example — jaccard, gwas_prs):
  python scripts/rerun_gnn_gradcam.py \
    --results_dir AF_PATHWAY_SCORES/PATHWAY_MODELING/gwas_prs_covs_jaccard/gnn/ \
    --splits_dir  AF_PATHWAY_SCORES/PATHWAY_MODELING/splits/ \
    --pathway_cols data/pathway_cols_gwas.txt \
    --covariates_file AF_PATHWAY_SCORES/PATHWAY_MODELING/Feature_Files/covariates_PRS.csv \
    --prs_col PRS_std \
    --graph_method jaccard \
    --pathway_gene_sets AF_PATHWAY_SCORES/GNN_PATHWAY/pathway_gene_sets.json

  python scripts/rerun_gnn_gradcam.py \
    --results_dir AF_PATHWAY_SCORES/PATHWAY_MODELING/gwas_grex_HAA_HLV_prs_covs_jaccard/gnn/ \
    --splits_dir  AF_PATHWAY_SCORES/PATHWAY_MODELING/splits/ \
    --pathway_cols data/pathway_cols_gwas_grex_HAA_HLV.txt \
    --covariates_file AF_PATHWAY_SCORES/PATHWAY_MODELING/Feature_Files/covariates_PRS.csv \
    --prs_col PRS_std \
    --graph_method jaccard \
    --pathway_gene_sets AF_PATHWAY_SCORES/GNN_PATHWAY/pathway_gene_sets.json

  python scripts/rerun_gnn_gradcam.py \
    --results_dir AF_PATHWAY_SCORES/PATHWAY_MODELING/gwas_grex_HAA_HLV_AA_prs_covs_jaccard/gnn/ \
    --splits_dir  AF_PATHWAY_SCORES/PATHWAY_MODELING/splits/ \
    --pathway_cols data/pathway_cols_gwas_grex_HAA_HLV_AA.txt \
    --covariates_file AF_PATHWAY_SCORES/PATHWAY_MODELING/Feature_Files/covariates_PRS.csv \
    --prs_col PRS_std \
    --graph_method jaccard \
    --pathway_gene_sets AF_PATHWAY_SCORES/GNN_PATHWAY/pathway_gene_sets.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trainer import load_data, make_loaders
from src.models.pathway_gnn import (
    PathwayGNN,
    build_jaccard_edge_index,
    build_score_correlation_edge_index,
    build_fully_connected_edge_index,
)
from src.interpretability import compute_gradcam_pathway_ranking, compute_node_embeddings


# ── helpers ───────────────────────────────────────────────────────────────────

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _scale_splits(pw, cov, idx_tr, idx_va, idx_te):
    pw_scaled  = pw.copy().astype(float)
    cov_scaled = cov.copy().astype(float)

    if cov.shape[1] > 0:
        cov_scaler = StandardScaler().fit(cov[idx_tr])
        for idx in (idx_tr, idx_va, idx_te):
            cov_scaled[idx] = cov_scaler.transform(cov[idx])

    if pw.shape[1] > 0:
        orig_shape = pw.shape[1:]
        pw_2d = pw.reshape(len(pw), -1)
        pw_scaler = StandardScaler().fit(pw_2d[idx_tr])
        for idx in (idx_tr, idx_va, idx_te):
            pw_scaled[idx] = pw_scaler.transform(pw_2d[idx]).reshape(-1, *orig_shape)

    return pw_scaled, cov_scaled


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Rerun GNN GradCAM without retraining.")
    parser.add_argument("--results_dir",       required=True,
                        help="Path to the gnn/ output directory (contains gnn.pt, best_hparams.json)")
    parser.add_argument("--splits_dir",        required=True,
                        help="Directory with splits.npz / data_config.json")
    parser.add_argument("--pathway_cols",      required=True,
                        help="Text file listing pathway column names (one per line)")
    parser.add_argument("--covariates_file",   required=True,
                        help="CSV with covariate columns")
    parser.add_argument("--prs_col",           default=None,
                        help="Name of PRS column in covariates_file (optional)")
    parser.add_argument("--graph_method",      required=True,
                        choices=["jaccard", "score_correlation", "fully_connected"],
                        help="Graph construction method used in the original run")
    parser.add_argument("--pathway_gene_sets", default=None,
                        help="JSON file mapping pathway → gene list (required for jaccard)")
    parser.add_argument("--jaccard_min_overlap", type=int,   default=1)
    parser.add_argument("--jaccard_min_score",   type=float, default=0.0)
    parser.add_argument("--corr_threshold",      type=float, default=0.0)
    parser.add_argument("--corr_top_k",          type=int,   default=50)
    parser.add_argument("--top_k_pathways",      type=int,   default=500)
    parser.add_argument("--device",              default="cpu")
    args = parser.parse_args()

    weight_path = os.path.join(args.results_dir, "gnn.pt")
    hparam_path = os.path.join(args.results_dir, "best_hparams.json")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"No saved weights at {weight_path} — has the GNN been trained?")

    print(f"Results dir  : {args.results_dir}")
    print(f"Graph method : {args.graph_method}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading data...")
    config_path = os.path.join(args.splits_dir, "data_config.json")
    splits_path = os.path.join(args.splits_dir, "splits.npz")
    with open(config_path) as f:
        config = json.load(f)

    # Read pathway cols from file
    with open(args.pathway_cols) as f:
        pathway_cols = [line.strip() for line in f if line.strip()]

    cov_path = args.covariates_file or config["covariates"]
    pw, cov, labels, pw_names_all, cov_cols, _ = load_data(
        config["pathway_matrix"],
        cov_path,
        config["label_col"],
        config["covariate_cols"],
    )

    # Append PRS column exactly as train_model.py does (--prs_col)
    if args.prs_col is not None:
        import pandas as pd
        _cov_df = pd.read_csv(cov_path) if cov_path.endswith(".csv") else pd.read_parquet(cov_path)
        if args.prs_col not in _cov_df.columns:
            raise ValueError(f"--prs_col {args.prs_col!r} not found in {cov_path}")
        prs_vals = _cov_df[args.prs_col].values.astype(np.float32).reshape(-1, 1)
        cov      = np.concatenate([cov, prs_vals], axis=1)
        cov_cols = list(cov_cols) + [args.prs_col]
        print(f"  Appended PRS column '{args.prs_col}' (C now {cov.shape[1]})")

    # Subset to requested pathway columns
    idx_pw   = [pw_names_all.index(c) for c in pathway_cols]
    pw       = pw[:, idx_pw] if pw.ndim == 2 else pw[:, idx_pw, :]
    pw_names = pathway_cols

    splits   = np.load(splits_path)
    idx_tr   = splits["idx_train"]
    idx_va   = splits["idx_val"]
    idx_te   = splits["idx_test"]

    # Determine K from the checkpoint's saved edge_index — this is the ground
    # truth for how many pathways the model was trained on, regardless of what
    # pathway_cols or graph_info say.
    print("\nPeeking at checkpoint to determine training K...")
    _ckpt_peek = torch.load(weight_path, map_location="cpu")
    if "edge_index" in _ckpt_peek:
        _k_from_ckpt = int(_ckpt_peek["edge_index"].max()) + 1
        print(f"  K={_k_from_ckpt} (from checkpoint edge_index)")
    else:
        _k_from_ckpt = None
        print("  edge_index not in checkpoint — will use graph_info or --top_k_pathways")

    # Apply the same top-k pathway selection used during training.
    # Priority: (1) graph_info.pathway_names if length matches checkpoint K,
    #           (2) variance selection using checkpoint K,
    #           (3) --top_k_pathways flag as last resort.
    graph_info_path = os.path.join(args.results_dir, "graph_info.json")
    _k_target = _k_from_ckpt
    _used_saved_names = False
    if os.path.exists(graph_info_path):
        with open(graph_info_path) as f:
            _gi = json.load(f)
        _saved_names = _gi.get("pathway_names", [])
        if _k_target and len(_saved_names) == _k_target:
            _name_to_col = {n: i for i, n in enumerate(pw_names)}
            _sel_idx = [_name_to_col[n] for n in _saved_names if n in _name_to_col]
            if len(_sel_idx) == _k_target:
                pw       = pw[:, _sel_idx] if pw.ndim == 2 else pw[:, _sel_idx, :]
                pw_names = list(_saved_names)
                _used_saved_names = True
                print(f"  Restored {len(pw_names)} training-time pathway names from graph_info.json")

    if not _used_saved_names and pw.shape[1] > (_k_target or args.top_k_pathways or pw.shape[1]):
        _k = _k_target or args.top_k_pathways
        # Inline rank-normalized variance selection (matches train_model.py logic)
        _pw_tr = pw[idx_tr]
        _raw = _pw_tr.var(axis=0)
        if _raw.ndim == 2:
            _raw = _raw.mean(axis=1)
        _prefixes = np.array([n.split("__")[0] for n in pw_names])
        _unique_px = np.unique(_prefixes)
        if len(_unique_px) > 1:
            _scores = np.zeros(len(pw_names))
            for _px in _unique_px:
                _mask = _prefixes == _px
                _grp  = _raw[_mask]
                _rnks = _grp.argsort().argsort().astype(float)
                _scores[_mask] = _rnks / max(_mask.sum() - 1, 1)
        else:
            _scores = _raw
        _top_idx  = np.argsort(-_scores)[:_k]
        pw        = pw[:, _top_idx] if pw.ndim == 2 else pw[:, _top_idx, :]
        pw_names  = [pw_names[i] for i in _top_idx]
        print(f"  Applied top-{_k} variance selection to match training")

    K, T = pw.shape[1], (pw.shape[2] if pw.ndim == 3 else 1)
    C    = cov.shape[1]
    print(f"  K={K} pathways, T={T} tissues, C={C} covariates")
    print(f"  Test set: {len(idx_te)} samples")

    pw_s, cov_s = _scale_splits(pw, cov, idx_tr, idx_va, idx_te)

    _, _, test_loader = make_loaders(
        pw_s[idx_tr], cov_s[idx_tr], labels[idx_tr],
        pw_s[idx_va], cov_s[idx_va], labels[idx_va],
        pw_s[idx_te], cov_s[idx_te], labels[idx_te],
        batch_size=32,
    )

    # ── Load checkpoint and extract saved edge_index ──────────────────────────
    print("\nLoading saved model weights...")
    checkpoint = torch.load(weight_path, map_location=args.device)

    # Use the edge_index that was saved in the checkpoint rather than
    # reconstructing it — avoids mismatches from Jaccard threshold differences.
    if "edge_index" in checkpoint:
        edge_index = checkpoint["edge_index"].to(args.device)
        n_edges = edge_index.shape[1] // 2
        print(f"  edge_index loaded from checkpoint: {K} nodes, {n_edges} undirected edges")
    else:
        print("\nBuilding graph (edge_index not in checkpoint)...")
        if args.graph_method == "jaccard":
            if not args.pathway_gene_sets:
                raise ValueError("--pathway_gene_sets required for jaccard")
            with open(args.pathway_gene_sets) as f:
                gene_sets = {k: set(v) for k, v in json.load(f).items()}
            edge_index = build_jaccard_edge_index(
                gene_sets, pw_names,
                min_overlap=args.jaccard_min_overlap,
                min_jaccard=args.jaccard_min_score,
            )
        elif args.graph_method == "score_correlation":
            edge_index = build_score_correlation_edge_index(
                pw_s[idx_tr], pw_names,
                threshold=args.corr_threshold,
                top_k=args.corr_top_k,
            )
        else:
            edge_index = None
        n_edges = edge_index.shape[1] // 2 if edge_index is not None else K * (K - 1) // 2
        print(f"  {K} nodes, {n_edges} undirected edges")

    with open(hparam_path) as f:
        hparams = json.load(f)

    # Extract only model constructor kwargs (not lr / weight_decay / epochs etc.)
    MODEL_HPARAM_KEYS = {"embed_dim", "dropout", "n_layers", "n_heads",
                         "hidden_dim", "aggregator"}
    model_kwargs = {k: v for k, v in hparams.items() if k in MODEL_HPARAM_KEYS}

    model = PathwayGNN(
        n_pathways=K,
        pathway_input_dim=T,
        covariate_dim=C,
        edge_index=edge_index,
        fully_connected=(args.graph_method == "fully_connected"),
        **model_kwargs,
    )
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"  Loaded: {weight_path}")

    # ── Rerun GradCAM ─────────────────────────────────────────────────────────
    print("\nRunning GradCAM attribution...")
    ranking, per_sample_gradcam = compute_gradcam_pathway_ranking(
        model, test_loader, pw_names, device=args.device
    )
    np.save(os.path.join(args.results_dir, "gradcam_scores.npy"), per_sample_gradcam)
    print(f"  gradcam_scores.npy: shape {per_sample_gradcam.shape}")

    print("\nRecomputing node embeddings...")
    node_embeddings = compute_node_embeddings(model, test_loader, device=args.device)
    np.save(os.path.join(args.results_dir, "node_embeddings.npy"), node_embeddings)
    print(f"  node_embeddings.npy: shape {node_embeddings.shape}")

    # ── Overwrite ranking.json (merge with existing covariate ranking if present) ──
    ranking_path = os.path.join(args.results_dir, "ranking.json")
    if os.path.exists(ranking_path):
        with open(ranking_path) as f:
            existing = json.load(f)
        # Preserve any covariate entries (non-pathway keys) from the original run
        existing.update(ranking)
        ranking = existing

    with open(ranking_path, "w") as f:
        json.dump(ranking, f, indent=2, cls=_NumpyEncoder)

    top5 = sorted(ranking.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nTop-5 pathways by GradCAM:")
    for name, score in top5:
        print(f"  {score:.6f}  {name}")

    print(f"\nDone. Outputs written to: {args.results_dir}")


if __name__ == "__main__":
    main()
