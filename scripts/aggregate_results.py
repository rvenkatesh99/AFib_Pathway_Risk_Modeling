"""
Collect per-model outputs and run cross-model analyses.
Run after all train_model.py jobs finish.

Usage:
  python scripts/aggregate_results.py \
      --results_dir results/ \
      [--ancestry_col ancestry --covariates data/covariates.csv]

Reads:  results_dir/<model>/metrics.json, probs_test.npy, labels_test.npy, ranking.json
Writes: results_dir/results.json        (input for plot_interpretability.py)
        results_dir/pathway_rankings.csv
        results_dir/mean_attention.npy  (copied from transformer/ for the plotting script)
"""

import json
import os
import shutil
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.metrics import pairwise_delong, evaluate_subgroups
from src.evaluation.interpretability import spearman_rank_concordance

MODELS = ["prs_logistic", "l1_logistic", "random_forest", "global_attention", "transformer", "gnn"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/")
    p.add_argument("--ancestry_col", default=None)
    p.add_argument("--covariates", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    all_probs, all_labels, all_metrics, all_rankings = {}, {}, {}, {}
    missing = []

    for model in MODELS:
        d = os.path.join(args.results_dir, model)
        prob_path  = os.path.join(d, "probs_test.npy")
        label_path = os.path.join(d, "labels_test.npy")
        metric_path= os.path.join(d, "metrics.json")
        rank_path  = os.path.join(d, "ranking.json")

        if not os.path.exists(prob_path):
            missing.append(model)
            continue

        all_probs[model]  = np.load(prob_path)
        all_labels[model] = np.load(label_path)
        with open(metric_path) as f:
            all_metrics[model] = json.load(f)
        with open(rank_path) as f:
            ranking = json.load(f)
        if ranking:
            all_rankings[model] = list(ranking.items())

    if missing:
        print(f"Missing model outputs (excluded): {missing}")
    if not all_probs:
        raise RuntimeError("No model outputs found. Run train_model.py for each model first.")

    # Use labels from first available model (all share the same test split)
    test_labels = next(iter(all_labels.values()))

    # Ancestry subgroup evaluation
    subgroups = {}
    if args.ancestry_col and args.covariates:
        splits  = np.load(os.path.join(args.results_dir, "splits.npz"))
        cov_df  = pd.read_csv(args.covariates)
        ancestry = cov_df.iloc[splits["idx_test"]][args.ancestry_col].values
        for model, probs in all_probs.items():
            subgroups[model] = evaluate_subgroups(test_labels, probs, ancestry)

    # Pairwise DeLong tests
    delong = pairwise_delong(all_probs, test_labels)

    # Pathway ranking concordance (models that have rankings)
    spearman = {}
    if len(all_rankings) >= 2:
        spearman = spearman_rank_concordance(
            {m: dict(items) for m, items in all_rankings.items()}
        )

    # Print summary
    print(f"\n{'Model':<20} {'AUROC':>7} {'95% CI':>20}  {'AUPRC':>7}")
    print("-" * 60)
    for model, result in all_metrics.items():
        m  = result["metrics"]
        ci = result["bootstrap_95ci"]["auroc"]
        print(f"{model:<20} {m['auroc']:>7.4f} "
              f"[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]  {m['auprc']:>7.4f}")

    print(f"\nPairwise DeLong tests:")
    for (m1, m2), res in delong.items():
        print(f"  {m1} vs {m2}: z={res['z_stat']:.3f}, p={res['p_value']:.4f}")

    if spearman:
        print(f"\nPathway ranking Spearman concordance:")
        for (m1, m2), res in spearman.items():
            print(f"  {m1} vs {m2}: rho={res['rho']:.3f}, p={res['p_value']:.4f}")

    # Write results.json
    output = {
        "models_included": list(all_probs),
        "models_missing":  missing,
        "eval_results":    {m: {**all_metrics[m], **({"subgroups": subgroups[m]} if m in subgroups else {})}
                            for m in all_metrics},
        "delong_tests":    {f"{m1}_vs_{m2}": v for (m1, m2), v in delong.items()},
        "spearman_concordance": {f"{m1}_vs_{m2}": v for (m1, m2), v in spearman.items()},
        "pathway_rankings":all_rankings,
    }
    out_path = os.path.join(args.results_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {out_path}")

    # Pathway rankings CSV
    if all_rankings:
        ranking_df = pd.DataFrame({m: dict(items) for m, items in all_rankings.items()})
        ranking_df.index.name = "pathway"
        csv_path = os.path.join(args.results_dir, "pathway_rankings.csv")
        ranking_df.to_csv(csv_path)
        print(f"Wrote {csv_path}")

    # Copy mean_attention.npy for plot_interpretability.py
    attn_src = os.path.join(args.results_dir, "transformer", "mean_attention.npy")
    attn_dst = os.path.join(args.results_dir, "mean_attention.npy")
    if os.path.exists(attn_src):
        shutil.copy(attn_src, attn_dst)


if __name__ == "__main__":
    main()
