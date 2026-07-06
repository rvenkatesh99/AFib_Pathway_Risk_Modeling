"""
Post-hoc covariate importance patching for global_attention and transformer runs.

Retraining is NOT required. This script:
  1. Loads the saved model checkpoint (.pt) and best_hparams.json
  2. Reconstructs the original test data loader (same scaling as training)
  3. Computes gradient × input covariate attribution on the test set
  4. Appends the covariate scores to the existing ranking.json

Run this once per (feature_set, model) directory that's missing covariate scores.

Usage:
  # Single run:
  python scripts/patch_covariate_importance.py \
      --run_dir  /path/PATHWAY_MODELING/gwas_prs/global_attention/ \
      --splits_dir /path/PATHWAY_MODELING/gwas_prs/

  # All global_attention + transformer runs under a parent directory:
  python scripts/patch_covariate_importance.py \
      --parent_dir /path/PATHWAY_MODELING/

Output:
  Overwrites ranking.json in each run_dir, appending covariate scores.
  Original pathway scores are preserved exactly.
  A backup (ranking.json.bak) is written before modification.
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.interpretability import compute_gradient_covariate_importance
from src.trainer import make_loaders
from scripts.train_model import load_splits, _scale_splits

PATCHABLE_MODELS = {
    "global_attention": ("global_attention.pt", "GlobalPathwayAttentionModel"),
    "transformer":      ("transformer.pt",       "PathwayTransformer"),
}

# Fixed architecture kwargs that aren't in best_hparams
FIXED_EXTRA = {
    "transformer": {"n_layers": 2},
}


def parse_args():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--run_dir",
                       help="Single model run directory (contains ranking.json + .pt file).")
    group.add_argument("--parent_dir",
                       help="Parent directory; patches all global_attention and transformer "
                            "subdirs found recursively.")
    p.add_argument("--splits_dir", default=None,
                   help="Directory containing data_config.json + splits.npz. "
                        "When using --parent_dir, defaults to {parent_dir}/splits/. "
                        "When using --run_dir, defaults to the parent of run_dir.")
    p.add_argument("--pathway_cols_file", default=None,
                   help="Optional .txt file listing pathway column names (one per line). "
                        "Pass if the run used a subset of pathways.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print what would be patched without modifying any files.")
    return p.parse_args()


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        return super().default(obj)


def discover_runs(parent_dir, splits_dir):
    """Walk parent_dir for global_attention / transformer run dirs."""
    found = []
    for root, dirs, files in os.walk(parent_dir):
        model_name = os.path.basename(root)
        if model_name not in PATCHABLE_MODELS:
            continue
        weight_file = PATCHABLE_MODELS[model_name][0]
        if (os.path.isfile(os.path.join(root, "ranking.json")) and
                os.path.isfile(os.path.join(root, weight_file)) and
                os.path.isfile(os.path.join(root, "best_hparams.json"))):
            found.append((root, splits_dir, model_name))
    return found


def covariate_scores_present(ranking_path, cov_cols):
    """Return True if at least one covariate name is already in ranking.json."""
    with open(ranking_path) as f:
        ranking = json.load(f)
    return any(c in ranking for c in cov_cols)


def load_model(model_name, run_dir, hparams, n_pathways, pathway_input_dim, covariate_dim):
    if model_name == "global_attention":
        from src.models.global_attention import GlobalPathwayAttentionModel
        extra = FIXED_EXTRA.get(model_name, {})
        model = GlobalPathwayAttentionModel(
            n_pathways=n_pathways,
            pathway_input_dim=pathway_input_dim,
            covariate_dim=covariate_dim,
            **{k: v for k, v in {**hparams, **extra}.items()
               if k not in ("lr", "weight_decay", "n_epochs", "patience")},
        )
    elif model_name == "transformer":
        from src.models.global_attention import PathwayTransformer
        extra = FIXED_EXTRA.get(model_name, {})
        model = PathwayTransformer(
            n_pathways=n_pathways,
            pathway_input_dim=pathway_input_dim,
            covariate_dim=covariate_dim,
            **{k: v for k, v in {**hparams, **extra}.items()
               if k not in ("lr", "weight_decay", "n_epochs", "patience")},
        )
    else:
        raise ValueError(f"Unknown patchable model: {model_name}")

    weight_path = os.path.join(run_dir, PATCHABLE_MODELS[model_name][0])
    state = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def patch_run(run_dir, splits_dir, model_name, pathway_cols_file=None, dry_run=False):
    ranking_path = os.path.join(run_dir, "ranking.json")
    hparams_path = os.path.join(run_dir, "best_hparams.json")

    with open(hparams_path) as f:
        hparams = json.load(f)

    # Load pathway column list if provided
    pathway_cols = None
    if pathway_cols_file and os.path.isfile(pathway_cols_file):
        with open(pathway_cols_file) as f:
            pathway_cols = [l.strip() for l in f if l.strip()]

    # Reconstruct data exactly as during training
    pw, cov, labels, pw_names, cov_cols, idx_tr, idx_va, idx_te = load_splits(
        splits_dir, pathway_cols=pathway_cols
    )
    pw_s, cov_s = _scale_splits(pw, cov, idx_tr, idx_va, idx_te)

    # Check if already patched
    if covariate_scores_present(ranking_path, cov_cols):
        print(f"  SKIP (already has covariate scores): {run_dir}")
        return

    K = pw_s.shape[1] if pw_s.ndim >= 2 else 0
    T = pw_s.shape[2] if pw_s.ndim == 3 else (1 if K > 0 else 0)
    C = cov_s.shape[1]

    print(f"  Patching {model_name}: K={K}, T={T}, C={C}, "
          f"{len(cov_cols)} covariates, {len(idx_te)} test samples")

    if dry_run:
        print(f"  [dry-run] Would patch: {ranking_path}")
        return

    # Build test loader (same batch size as training doesn't matter for attribution)
    _, _, test_loader = make_loaders(
        pw_s[idx_tr], cov_s[idx_tr], labels[idx_tr],
        pw_s[idx_va], cov_s[idx_va], labels[idx_va],
        pw_s[idx_te], cov_s[idx_te], labels[idx_te],
        batch_size=256,
    )

    # Load model from checkpoint
    try:
        model = load_model(model_name, run_dir, hparams, K, T, C)
    except Exception as e:
        print(f"  ERROR loading model from {run_dir}: {e}")
        return

    # Compute gradient × input covariate attribution
    cov_ranking = compute_gradient_covariate_importance(model, test_loader, cov_cols)
    if not cov_ranking:
        print(f"  WARNING: empty covariate ranking for {run_dir}")
        return

    # Backup + patch ranking.json
    shutil.copy2(ranking_path, ranking_path + ".bak")
    with open(ranking_path) as f:
        existing = json.load(f)
    existing.update(cov_ranking)
    with open(ranking_path, "w") as f:
        json.dump(existing, f, indent=2, cls=_NumpyEncoder)

    top3 = list(cov_ranking.items())[:3]
    print(f"  Patched. Top covariates: "
          + ", ".join(f"{k}={v:.4f}" for k, v in top3))


def main():
    args = parse_args()

    if args.run_dir:
        model_name = os.path.basename(args.run_dir.rstrip("/"))
        if model_name not in PATCHABLE_MODELS:
            raise ValueError(f"run_dir must be a global_attention or transformer directory, "
                             f"got: {model_name}")
        splits_dir = args.splits_dir or os.path.dirname(args.run_dir.rstrip("/"))
        runs = [(args.run_dir, splits_dir, model_name)]
    else:
        splits_dir = args.splits_dir or os.path.join(args.parent_dir, "splits")
        if not os.path.isfile(os.path.join(splits_dir, "splits.npz")):
            raise FileNotFoundError(
                f"splits.npz not found in {splits_dir}. "
                f"Pass --splits_dir explicitly if it lives elsewhere."
            )
        runs = discover_runs(args.parent_dir, splits_dir)
        if not runs:
            raise RuntimeError(f"No patchable runs found under {args.parent_dir}")

    print(f"Found {len(runs)} run(s) to patch")
    for run_dir, splits_dir, model_name in runs:
        print(f"\n[{model_name}] {run_dir}")
        try:
            patch_run(run_dir, splits_dir, model_name,
                      pathway_cols_file=args.pathway_cols_file,
                      dry_run=args.dry_run)
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
