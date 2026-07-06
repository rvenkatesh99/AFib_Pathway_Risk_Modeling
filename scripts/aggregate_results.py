"""
Aggregate results across all feature sets and models.

Directory structure expected:
  results_dir/
    {feature_set}/
      {model}/
        metrics.json
        ranking.json
        probs_test.npy
        labels_test.npy

Usage:
  python scripts/aggregate_results.py \
      --results_dir /path/to/PATHWAY_MODELING/ \
      [--out_dir /path/to/output/]

Outputs (written to out_dir, defaults to results_dir):
  metrics_summary.csv     — AUROC / AUPRC / Brier per (feature_set, model) + 95% CIs
  ranking_percentile.csv  — within-model percentile rank (0-100) per feature; rows=features, cols=(feature_set, model)
  delong_summary.csv      — pairwise DeLong z/p within each feature set
  ranking_top20.txt       — top-20 features per (feature_set, model) for quick inspection
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics import pairwise_delong, compute_nri, compute_idi, compute_calibration

KNOWN_MODELS = [
    "l1_logistic", "elasticnet",
    "random_forest", "global_attention", "transformer", "gnn",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True,
                   help="Top-level directory containing one subdirectory per feature set.")
    p.add_argument("--out_dir", default=None,
                   help="Where to write output CSVs (defaults to results_dir).")
    return p.parse_args()


def discover_runs(results_dir):
    """Return list of (feature_set, model, run_dir) tuples for completed runs.

    Handles two directory layouts:
      - results_dir/{feature_set}/{model}/metrics.json  (multi-feature-set)
      - results_dir/{model}/metrics.json                (single feature set)
    """
    runs = []
    feature_set_name = os.path.basename(results_dir.rstrip("/"))

    for entry in sorted(os.listdir(results_dir)):
        entry_dir = os.path.join(results_dir, entry)
        if not os.path.isdir(entry_dir):
            continue

        # Single-feature-set layout: results_dir/model/metrics.json
        if os.path.isfile(os.path.join(entry_dir, "metrics.json")):
            runs.append((feature_set_name, entry, entry_dir))
            continue

        # Multi-feature-set layout: results_dir/feature_set/model/metrics.json
        for model in sorted(os.listdir(entry_dir)):
            run_dir = os.path.join(entry_dir, model)
            if os.path.isdir(run_dir) and os.path.isfile(os.path.join(run_dir, "metrics.json")):
                runs.append((entry, model, run_dir))

    return runs


def load_run(run_dir):
    with open(os.path.join(run_dir, "metrics.json")) as f:
        metrics = json.load(f)
    rank_path = os.path.join(run_dir, "ranking.json")
    ranking = {}
    if os.path.isfile(rank_path):
        with open(rank_path) as f:
            ranking = json.load(f)
    probs_path = os.path.join(run_dir, "probs_test.npy")
    labels_path = os.path.join(run_dir, "labels_test.npy")
    probs = np.load(probs_path) if os.path.isfile(probs_path) else None
    labels = np.load(labels_path) if os.path.isfile(labels_path) else None
    return metrics, ranking, probs, labels


def scores_to_percentile_rank(scores: dict) -> dict:
    """Convert raw importance scores to percentile ranks (0-100, higher = more important)."""
    if not scores:
        return {}
    names = list(scores.keys())
    vals = np.array([scores[n] for n in names], dtype=float)
    # rank from 0 to len-1, then scale to 0-100
    order = np.argsort(np.argsort(vals))  # rank of each element
    percentiles = order / max(len(order) - 1, 1) * 100
    return {n: float(p) for n, p in zip(names, percentiles)}


def strip_prefix(name: str) -> str:
    """Remove tissue/source prefix (e.g. grex_AC__ or gwas__) from a feature name."""
    if "__" in name:
        return name.split("__", 1)[1]
    return name


def main():
    args = parse_args()
    out_dir = args.out_dir or args.results_dir
    os.makedirs(out_dir, exist_ok=True)

    runs = discover_runs(args.results_dir)
    if not runs:
        raise RuntimeError(f"No completed runs found under {args.results_dir}")
    print(f"Found {len(runs)} completed runs across "
          f"{len({fs for fs, _, _ in runs})} feature sets and "
          f"{len({m for _, m, _ in runs})} models.")

    # ── Collect everything ────────────────────────────────────────────────────
    metrics_rows = []
    all_probs = {}   # (feature_set, model) -> probs
    all_labels = {}  # feature_set -> labels (shared across models)
    rankings = {}    # (feature_set, model) -> {feature: score}

    for fs, model, run_dir in runs:
        try:
            met, ranking, probs, labels = load_run(run_dir)
        except Exception as e:
            print(f"  WARNING: could not load {fs}/{model}: {e}")
            continue

        m = met["metrics"]
        ci = met.get("bootstrap_95ci", {})
        auroc_ci = ci.get("auroc", {})
        auprc_ci = ci.get("auprc", {})
        f1_ci    = ci.get("f1_at_opt_threshold", {})
        ba_ci    = ci.get("balanced_accuracy", {})
        sen_ci   = ci.get("sensitivity", {})
        spe_ci   = ci.get("specificity", {})

        metrics_rows.append({
            "feature_set": fs,
            "model": model,
            "auroc": m.get("auroc"),
            "auroc_ci_lower": auroc_ci.get("ci_lower"),
            "auroc_ci_upper": auroc_ci.get("ci_upper"),
            "auprc": m.get("auprc"),
            "auprc_ci_lower": auprc_ci.get("ci_lower"),
            "auprc_ci_upper": auprc_ci.get("ci_upper"),
            "brier_score": m.get("brier_score"),
            "threshold": m.get("threshold"),
            "sensitivity": m.get("sensitivity"),
            "sensitivity_ci_lower": sen_ci.get("ci_lower"),
            "sensitivity_ci_upper": sen_ci.get("ci_upper"),
            "specificity": m.get("specificity"),
            "specificity_ci_lower": spe_ci.get("ci_lower"),
            "specificity_ci_upper": spe_ci.get("ci_upper"),
            "balanced_accuracy": m.get("balanced_accuracy"),
            "balanced_accuracy_ci_lower": ba_ci.get("ci_lower"),
            "balanced_accuracy_ci_upper": ba_ci.get("ci_upper"),
            "f1": m.get("f1_at_opt_threshold"),
            "f1_ci_lower": f1_ci.get("ci_lower"),
            "f1_ci_upper": f1_ci.get("ci_upper"),
        })

        if probs is not None:
            all_probs[(fs, model)] = probs
        if labels is not None:
            all_labels[fs] = labels
        if ranking:
            rankings[(fs, model)] = ranking

    # ── Delta AUROC vs covariates-only baseline ───────────────────────────────
    # Baseline: covariates_logistic from covs_only (or best available fallback)
    baseline_auroc = None
    for cov_fs in ["covs_only"]:
        row = next((r for r in metrics_rows
                    if r["feature_set"] == cov_fs and r["model"] == "l1_logistic"), None)
        if row:
            baseline_auroc = row["auroc"]
            print(f"\nBaseline (l1_logistic / {cov_fs}): AUROC = {baseline_auroc:.4f}")
            break

    for row in metrics_rows:
        if baseline_auroc is not None:
            row["delta_auroc_vs_baseline"] = (
                row["auroc"] - baseline_auroc if row["auroc"] is not None else None
            )
        # Per-model delta: improvement over same model run on covs_only
        cov_row = next((r for r in metrics_rows
                        if r["feature_set"] == "covs_only" and r["model"] == row["model"]), None)
        row["delta_auroc_vs_covs_only"] = (
            row["auroc"] - cov_row["auroc"]
            if cov_row and row["auroc"] is not None and cov_row["auroc"] is not None
            else None
        )

    # ── Sparsity for l1_logistic and elasticnet ───────────────────────────────
    SPARSE_MODELS = {"l1_logistic", "elasticnet"}
    sparsity_rows = []
    for fs, model, run_dir in runs:
        if model not in SPARSE_MODELS:
            continue
        rank_path = os.path.join(run_dir, "ranking.json")
        if not os.path.isfile(rank_path):
            continue
        with open(rank_path) as f:
            ranking = json.load(f)
        # Separate pathway features (have __ prefix) from covariates
        pathway_scores = {k: v for k, v in ranking.items() if "__" in k}
        cov_scores     = {k: v for k, v in ranking.items() if "__" not in k}
        n_pw_total     = len(pathway_scores)
        n_pw_nonzero   = sum(1 for v in pathway_scores.values() if v > 0)
        n_cov_nonzero  = sum(1 for v in cov_scores.values() if v > 0)
        sparsity_rows.append({
            "feature_set":       fs,
            "model":             model,
            "n_pathways_total":  n_pw_total,
            "n_pathways_nonzero": n_pw_nonzero,
            "pct_pathways_selected": round(100 * n_pw_nonzero / n_pw_total, 1) if n_pw_total else None,
            "n_covariates_nonzero": n_cov_nonzero,
        })

    if sparsity_rows:
        sparsity_df = pd.DataFrame(sparsity_rows).sort_values(["model", "feature_set"])
        sparsity_path = os.path.join(out_dir, "sparsity_summary.csv")
        sparsity_df.to_csv(sparsity_path, index=False)
        print(f"\nSparsity summary (L1 / elasticnet pathway selection):")
        print(sparsity_df.to_string(index=False))
        print(f"\nWrote {sparsity_path}")

    # ── Metrics summary CSV ───────────────────────────────────────────────────
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = metrics_df.sort_values(["feature_set", "model"])
    metrics_path = os.path.join(out_dir, "metrics_summary.csv")
    metrics_df.to_csv(metrics_path, index=False, float_format="%.4f")
    print(f"\nWrote {metrics_path}")

    # Print to console — wide table: rows=feature_sets, cols=models
    pivot = metrics_df.pivot(index="feature_set", columns="model", values="auroc")
    print(f"\nAUROC by feature set and model:")
    print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))

    if baseline_auroc is not None:
        delta_pivot = metrics_df.pivot(
            index="feature_set", columns="model", values="delta_auroc_vs_baseline"
        )
        print(f"\nDelta AUROC vs l1_logistic baseline:")
        print(delta_pivot.to_string(float_format=lambda x: f"{x:+.4f}"))

    # ── DeLong pairwise within each feature set ───────────────────────────────
    delong_rows = []
    for fs in sorted({fs for fs, _, _ in runs}):
        fs_probs = {m: all_probs[(fs, m)] for (f, m) in all_probs if f == fs}
        if fs not in all_labels or len(fs_probs) < 2:
            continue
        delong = pairwise_delong(fs_probs, all_labels[fs])
        for (m1, m2), res in delong.items():
            delong_rows.append({
                "feature_set": fs,
                "model_1": m1,
                "model_2": m2,
                "z_stat": res["z_stat"],
                "p_value": res["p_value"],
            })

    if delong_rows:
        delong_df = pd.DataFrame(delong_rows).sort_values(["feature_set", "p_value"])
        delong_path = os.path.join(out_dir, "delong_summary.csv")
        delong_df.to_csv(delong_path, index=False, float_format="%.4f")
        print(f"Wrote {delong_path}")

    # ── DeLong cross-feature-set: same model, same labels, different features ──
    # Valid only when all feature sets share the same test-set labels (same splits).
    # Compares every (fs1, model) vs (fs2, model) pair for a fixed model.
    # Skips pairs where labels differ (different splits or test sets).
    cross_delong_rows = []
    all_models_seen = sorted({m for _, m, _ in runs})
    all_fs_seen     = sorted({fs for fs, _, _ in runs})

    for model in all_models_seen:
        # Collect (fs, probs, labels) for this model across all feature sets
        model_entries = [
            (fs, all_probs[(fs, model)], all_labels.get(fs))
            for fs in all_fs_seen
            if (fs, model) in all_probs and all_labels.get(fs) is not None
        ]
        for i, (fs1, probs1, labels1) in enumerate(model_entries):
            for fs2, probs2, labels2 in model_entries[i + 1:]:
                if not np.array_equal(labels1, labels2):
                    continue  # different test sets — DeLong invalid
                res = pairwise_delong({fs1: probs1, fs2: probs2}, labels1)
                for (m1, m2), r in res.items():
                    cross_delong_rows.append({
                        "model":        model,
                        "feature_set_1": m1,
                        "feature_set_2": m2,
                        "z_stat":       r["z_stat"],
                        "p_value":      r["p_value"],
                    })

    if cross_delong_rows:
        cross_df = pd.DataFrame(cross_delong_rows).sort_values(["model", "p_value"])
        cross_path = os.path.join(out_dir, "delong_cross_featureset.csv")
        cross_df.to_csv(cross_path, index=False, float_format="%.4f")
        print(f"\nCross-feature-set DeLong (same model, different feature sets):")
        # Print only comparisons involving the best feature sets for brevity
        key_fs = {"gwas_prs", "gwas_grex_HAA_HLV_prs", "gwas_grex_HAA_HLV_AA_prs",
                  "prs_covs", "covs_only"}
        summary = cross_df[
            cross_df["feature_set_1"].isin(key_fs) & cross_df["feature_set_2"].isin(key_fs)
        ]
        if len(summary):
            print(summary.to_string(index=False))
        print(f"Wrote {cross_path}")

    # ── NRI / IDI / Calibration vs covs_only baseline ────────────────────────
    ref_probs = all_probs.get(("covs_only", "l1_logistic"))
    ref_labels = all_labels.get("covs_only")

    reclassification_rows = []
    calibration_rows = []

    for (fs, model), probs in all_probs.items():
        labels_fs = all_labels.get(fs)
        if labels_fs is None:
            continue

        # Calibration for every model
        cal = compute_calibration(labels_fs, probs)
        calibration_rows.append({
            "feature_set": fs, "model": model,
            **cal,
        })

        # NRI / IDI only vs the covs_only reference, on shared test labels
        if ref_probs is not None and ref_labels is not None and np.array_equal(labels_fs, ref_labels):
            nri = compute_nri(labels_fs, probs, ref_probs)
            idi = compute_idi(labels_fs, probs, ref_probs)
            reclassification_rows.append({
                "feature_set": fs, "model": model,
                **nri, **idi,
            })

    if calibration_rows:
        cal_df = pd.DataFrame(calibration_rows).sort_values(["feature_set", "model"])
        cal_path = os.path.join(out_dir, "calibration_summary.csv")
        cal_df.to_csv(cal_path, index=False, float_format="%.4f")
        print(f"\nCalibration summary (slope≈1, intercept≈0 = well calibrated):")
        print(cal_df.to_string(index=False))
        print(f"Wrote {cal_path}")

    if reclassification_rows:
        reclassify_df = pd.DataFrame(reclassification_rows).sort_values(
            ["feature_set", "model"]
        )
        reclassify_path = os.path.join(out_dir, "nri_idi_summary.csv")
        reclassify_df.to_csv(reclassify_path, index=False, float_format="%.4f")
        print(f"\nNRI / IDI vs covs_only l1_logistic baseline:")
        print(reclassify_df[["feature_set", "model", "nri", "nri_p", "idi", "idi_p"]].to_string(index=False))
        print(f"Wrote {reclassify_path}")

    # ── Ranking percentile table ──────────────────────────────────────────────
    if rankings:
        # Convert raw scores to within-run percentile ranks
        pct_rankings = {key: scores_to_percentile_rank(r) for key, r in rankings.items()}

        # Build a unified index of all feature names (strip prefix for readability)
        all_features = sorted({strip_prefix(f) for r in pct_rankings.values() for f in r})

        # Build multi-column DataFrame: columns = (feature_set, model)
        col_tuples = sorted(pct_rankings.keys())
        data = {}
        for (fs, model) in col_tuples:
            pct = pct_rankings[(fs, model)]
            # map stripped name -> percentile; average if multiple prefixed versions exist
            stripped = {}
            for fname, val in pct.items():
                key = strip_prefix(fname)
                stripped.setdefault(key, []).append(val)
            data[(fs, model)] = {k: float(np.mean(v)) for k, v in stripped.items()}

        ranking_df = pd.DataFrame(data, index=all_features)
        ranking_df.index.name = "feature"
        ranking_df.columns = pd.MultiIndex.from_tuples(
            ranking_df.columns, names=["feature_set", "model"]
        )
        ranking_df = ranking_df.sort_index()
        rank_path = os.path.join(out_dir, "ranking_percentile.csv")
        ranking_df.to_csv(rank_path, float_format="%.1f")
        print(f"Wrote {rank_path}")

        # Top-20 per run for quick inspection
        top_path = os.path.join(out_dir, "ranking_top20.txt")
        with open(top_path, "w") as f:
            for (fs, model) in col_tuples:
                pct = pct_rankings[(fs, model)]
                top = sorted(pct.items(), key=lambda x: -x[1])[:20]
                f.write(f"\n{'='*60}\n{fs} / {model}\n{'='*60}\n")
                for rank, (name, score) in enumerate(top, 1):
                    f.write(f"  {rank:2d}. {strip_prefix(name):<50} {score:5.1f}\n")
        print(f"Wrote {top_path}")

    # ── Save probs / labels for downstream plots ──────────────────────────────
    # plot_interpretability.py loads these to draw ROC, PRC, and calibration curves.
    probs_dir = os.path.join(out_dir, "probs")
    os.makedirs(probs_dir, exist_ok=True)
    for (fs, model), probs in all_probs.items():
        safe = f"{fs}__{model}".replace("/", "_")
        np.save(os.path.join(probs_dir, f"{safe}.npy"), probs)
    for fs, labels in all_labels.items():
        np.save(os.path.join(probs_dir, f"labels__{fs}.npy"), labels)
    print(f"\nWrote probs/labels ({len(all_probs)} runs) to {probs_dir}/")

    print("\nDone.")


if __name__ == "__main__":
    main()
