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
    p.add_argument("--prs_col", default=None,
                   help="Name of PRS column appended as an extra covariate during training.")
    p.add_argument("--prs_file", default=None,
                   help="CSV/parquet file containing --prs_col. Required when PRS_std "
                        "is in a separate file from the main covariates.csv.")
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


def checkpoint_dims(run_dir, model_name):
    """Read K and C directly from the saved checkpoint weights to avoid mismatches."""
    weight_path = os.path.join(run_dir, PATCHABLE_MODELS[model_name][0])
    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    C = state["covariate_encoder.0.weight"].shape[1]
    if model_name == "global_attention":
        # attention_logits absent when K=0 (covariates-only run)
        K = state["attention_logits"].shape[0] if "attention_logits" in state else 0
    elif model_name == "transformer":
        # pathway_id_embed absent when K=0
        K = state["pathway_id_embed.weight"].shape[0] if "pathway_id_embed.weight" in state else 0
    return int(K), int(C)


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
        from src.models.pathway_transformer import PathwayTransformer
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
    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def patch_run(run_dir, splits_dir, model_name, prs_col=None, prs_file=None, dry_run=False):
    ranking_path = os.path.join(run_dir, "ranking.json")
    hparams_path = os.path.join(run_dir, "best_hparams.json")

    with open(hparams_path) as f:
        hparams = json.load(f)

    try:
        K_ckpt, C_ckpt = checkpoint_dims(run_dir, model_name)
    except Exception as e:
        print(f"  ERROR reading checkpoint dims: {e}")
        return

    # Pathway names: use ranking.json keys — these are exactly the K pathways trained on
    with open(ranking_path) as f:
        existing_ranking = json.load(f)
    pathway_cols = [k for k in existing_ranking
                    if "__" in k or not any(c.isspace() for c in k)]
    # Separate true pathway keys (have __ prefix) from any covariate keys already present
    pathway_cols = [k for k in existing_ranking.keys() if "__" in k]
    if not pathway_cols:
        pathway_cols = list(existing_ranking.keys())

    cov_keys_present = [k for k in existing_ranking if "__" not in k]
    if cov_keys_present:
        print(f"  SKIP (already has {len(cov_keys_present)} covariate scores): {run_dir}")
        return

    # Load data with the pathway subset the model was trained on
    pw, cov, labels, pw_names, cov_cols, idx_tr, idx_va, idx_te = load_splits(
        splits_dir, pathway_cols=pathway_cols
    )

    # Append PRS as extra covariate if checkpoint C > base covariate count
    if C_ckpt == cov.shape[1] + 1:
        if prs_col is None:
            print(f"  ERROR: checkpoint C={C_ckpt} but data C={cov.shape[1]}. "
                  f"Pass --prs_col and --prs_file to append PRS.")
            return
        import pandas as pd
        # Use --prs_file if given; otherwise fall back to the default covariates file
        if prs_file:
            src_path = prs_file
        else:
            config_path = os.path.join(splits_dir, "data_config.json")
            with open(config_path) as f:
                cfg = json.load(f)
            src_path = cfg["covariates"]
        prs_df = (pd.read_csv(src_path) if src_path.endswith(".csv")
                  else pd.read_parquet(src_path))
        if prs_col not in prs_df.columns:
            print(f"  ERROR: --prs_col '{prs_col}' not found in {src_path}. "
                  f"Available: {list(prs_df.columns)}")
            return
        prs_vals = prs_df[prs_col].values.astype(np.float32).reshape(-1, 1)
        cov = np.hstack([cov, prs_vals])
        cov_cols = list(cov_cols) + [prs_col]
        print(f"  Appended '{prs_col}' from {src_path} (C now {cov.shape[1]})")
    elif C_ckpt != cov.shape[1]:
        print(f"  ERROR: checkpoint C={C_ckpt} but data C={cov.shape[1]} "
              f"(difference > 1 — cannot auto-resolve). Skipping.")
        return

    pw_s, cov_s = _scale_splits(pw, cov, idx_tr, idx_va, idx_te)

    K = pw_s.shape[1] if pw_s.ndim >= 2 else 0
    T = pw_s.shape[2] if pw_s.ndim == 3 else (1 if K > 0 else 0)
    C = cov_s.shape[1]

    print(f"  Patching {model_name}: K={K}, T={T}, C={C}, "
          f"{len(cov_cols)} covariates, {len(idx_te)} test samples")

    if dry_run:
        print(f"  [dry-run] Would patch: {ranking_path}")
        return

    _, _, test_loader = make_loaders(
        pw_s[idx_tr], cov_s[idx_tr], labels[idx_tr],
        pw_s[idx_va], cov_s[idx_va], labels[idx_va],
        pw_s[idx_te], cov_s[idx_te], labels[idx_te],
        batch_size=256,
    )

    try:
        model = load_model(model_name, run_dir, hparams, K, T, C)
    except Exception as e:
        print(f"  ERROR loading model from {run_dir}: {e}")
        return

    cov_ranking = compute_gradient_covariate_importance(model, test_loader, cov_cols)
    if not cov_ranking:
        print(f"  WARNING: empty covariate ranking for {run_dir}")
        return

    # Backup + patch
    shutil.copy2(ranking_path, ranking_path + ".bak")
    existing_ranking.update(cov_ranking)
    with open(ranking_path, "w") as f:
        json.dump(existing_ranking, f, indent=2, cls=_NumpyEncoder)

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
                      prs_col=args.prs_col,
                      prs_file=args.prs_file,
                      dry_run=args.dry_run)
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
