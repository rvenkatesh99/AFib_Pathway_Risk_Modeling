"""
train/val/test split indices 
[--val_frac 0.1] [--test_frac 0.2] [--seed 42]

splits.npz  — arrays: idx_train, idx_val, idx_test
"""

import argparse
import json
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.trainer import load_data as load_data_from_files


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pathway_matrix", required=True)
    p.add_argument("--covariates", required=True)
    p.add_argument("--label_col", default="afib")
    p.add_argument("--covariate_cols", nargs="+", default=None)
    p.add_argument("--output_dir", default="results/")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)

    pathway_matrix, covariate_matrix, labels, pathway_names, covariate_cols, sample_ids = \
        load_data_from_files(args.pathway_matrix, args.covariates, args.label_col, args.covariate_cols)

    N = len(labels)
    idx = np.arange(N)
    idx_trainval, idx_test = train_test_split(
        idx, test_size=args.test_frac, stratify=labels, random_state=args.seed)
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=args.val_frac / (1 - args.test_frac),
        stratify=labels[idx_trainval],
        random_state=args.seed,
    )

    splits_path = os.path.join(args.output_dir, "splits.npz")
    np.savez(splits_path, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)

    T = pathway_matrix.shape[2] if pathway_matrix.ndim == 3 else 1
    config = {
        "pathway_matrix": args.pathway_matrix,
        "covariates": args.covariates,
        "label_col": args.label_col,
        "covariate_cols": covariate_cols,
        "pathway_names": pathway_names,
        "n_individuals": N,
        "n_pathways": pathway_matrix.shape[1],
        "pathway_input_dim": T,
        "covariate_dim": covariate_matrix.shape[1],
        "n_train": len(idx_train),
        "n_val": len(idx_val),
        "n_test": len(idx_test),
        "case_fraction": float(labels.mean()),
        "seed": args.seed,
    }
    with open(os.path.join(args.output_dir, "data_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Split: {len(idx_train)} train / {len(idx_val)} val / {len(idx_test)} test")
    print(f"Saved splits to {splits_path}")
    print(f"Saved config to {args.output_dir}/data_config.json")


if __name__ == "__main__":
    main()
