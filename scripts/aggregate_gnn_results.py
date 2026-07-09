"""
Aggregate GNN results across feature sets and graph methods.

Expected directory layout (one results_dir per GNN run):
  <parent_dir>/
    {feature_set}_{graph_method}/
      gnn/
        metrics.json
        graph_info.json
        probs_test.npy
        labels_test.npy

Usage:
  python scripts/aggregate_gnn_results.py \
      --parent_dir /path/to/PATHWAY_MODELING/ \
      [--out_dir /path/to/output/]

Outputs:
  gnn_metrics_summary.csv    — AUROC + CI per (feature_set, graph_method)
  gnn_auroc_pivot.csv        — feature_set × graph_method AUROC pivot
  gnn_graph_topology.csv     — n_nodes, n_edges, density per run
  gnn_delong_summary.csv     — pairwise DeLong between graph methods per feature set
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metrics import pairwise_delong

GRAPH_METHODS = ["jaccard", "score_correlation", "string", "fully_connected"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--parent_dir", required=True,
                   help="Directory containing one subdir per GNN run "
                        "(named {feature_set}_{graph_method}).")
    p.add_argument("--out_dir", default=None,
                   help="Where to write outputs (defaults to parent_dir).")
    return p.parse_args()


def _strip_method(name):
    """Return (base_feature_set, graph_method) by stripping trailing _<method>."""
    for m in sorted(GRAPH_METHODS, key=len, reverse=True):
        if name.endswith(f"_{m}"):
            return name[: -len(m) - 1], m
    return name, None


def discover_gnn_runs(parent_dir):
    """Walk parent_dir for completed gnn runs. Returns list of dicts."""
    runs = []
    for entry in sorted(os.listdir(parent_dir)):
        entry_dir = os.path.join(parent_dir, entry)
        if not os.path.isdir(entry_dir):
            continue
        gnn_dir = os.path.join(entry_dir, "gnn")
        metrics_path = os.path.join(gnn_dir, "metrics.json")
        if not os.path.isfile(metrics_path):
            continue
        base_fs, method = _strip_method(entry)
        if method is None:
            continue  # not a GNN-tagged dir, skip
        runs.append({
            "feature_set": base_fs,
            "graph_method": method,
            "run_dir": gnn_dir,
        })
    return runs


def load_run(run_dir):
    with open(os.path.join(run_dir, "metrics.json")) as f:
        metrics = json.load(f)

    graph_info = {}
    gi_path = os.path.join(run_dir, "graph_info.json")
    if os.path.isfile(gi_path):
        with open(gi_path) as f:
            graph_info = json.load(f)

    probs_path  = os.path.join(run_dir, "probs_test.npy")
    labels_path = os.path.join(run_dir, "labels_test.npy")
    probs  = np.load(probs_path)  if os.path.isfile(probs_path)  else None
    labels = np.load(labels_path) if os.path.isfile(labels_path) else None

    return metrics, graph_info, probs, labels


def main():
    args = parse_args()
    out_dir = args.out_dir or args.parent_dir
    os.makedirs(out_dir, exist_ok=True)

    runs = discover_gnn_runs(args.parent_dir)
    if not runs:
        raise RuntimeError(f"No completed GNN runs found under {args.parent_dir}")

    feature_sets  = sorted({r["feature_set"]  for r in runs})
    graph_methods = sorted({r["graph_method"] for r in runs},
                           key=lambda m: GRAPH_METHODS.index(m) if m in GRAPH_METHODS else 99)

    print(f"Found {len(runs)} GNN runs: "
          f"{len(feature_sets)} feature sets × {len(graph_methods)} graph methods")
    print(f"  Feature sets : {feature_sets}")
    print(f"  Graph methods: {graph_methods}")

    metrics_rows  = []
    topology_rows = []
    all_probs     = {}   # (fs, method) -> probs
    all_labels    = {}   # fs -> labels

    for r in runs:
        fs, method, run_dir = r["feature_set"], r["graph_method"], r["run_dir"]
        try:
            met, graph_info, probs, labels = load_run(run_dir)
        except Exception as e:
            print(f"  WARNING: could not load {fs}/{method}: {e}")
            continue

        m  = met["metrics"]
        ci = met.get("bootstrap_95ci", {})

        metrics_rows.append({
            "feature_set":      fs,
            "graph_method":     method,
            "auroc":            m.get("auroc"),
            "auroc_ci_lower":   ci.get("auroc", {}).get("ci_lower"),
            "auroc_ci_upper":   ci.get("auroc", {}).get("ci_upper"),
            "auprc":            m.get("auprc"),
            "auprc_ci_lower":   ci.get("auprc", {}).get("ci_lower"),
            "auprc_ci_upper":   ci.get("auprc", {}).get("ci_upper"),
            "brier_score":            m.get("brier_score"),
            "brier_score_ci_lower":   ci.get("brier_score", {}).get("ci_lower"),
            "brier_score_ci_upper":   ci.get("brier_score", {}).get("ci_upper"),
            "sensitivity":            m.get("sensitivity"),
            "sensitivity_ci_lower":   ci.get("sensitivity", {}).get("ci_lower"),
            "sensitivity_ci_upper":   ci.get("sensitivity", {}).get("ci_upper"),
            "specificity":            m.get("specificity"),
            "specificity_ci_lower":   ci.get("specificity", {}).get("ci_lower"),
            "specificity_ci_upper":   ci.get("specificity", {}).get("ci_upper"),
            "balanced_accuracy":      m.get("balanced_accuracy"),
            "balanced_accuracy_ci_lower": ci.get("balanced_accuracy", {}).get("ci_lower"),
            "balanced_accuracy_ci_upper": ci.get("balanced_accuracy", {}).get("ci_upper"),
            "f1":                     m.get("f1_at_opt_threshold"),
            "f1_ci_lower":            ci.get("f1_at_opt_threshold", {}).get("ci_lower"),
            "f1_ci_upper":            ci.get("f1_at_opt_threshold", {}).get("ci_upper"),
        })

        n_nodes = graph_info.get("n_nodes", None)
        n_edges = graph_info.get("n_edges", None)
        density = (2 * n_edges / (n_nodes * (n_nodes - 1))
                   if n_nodes and n_edges and n_nodes > 1 else None)
        topology_rows.append({
            "feature_set":  fs,
            "graph_method": method,
            "n_nodes":      n_nodes,
            "n_edges":      n_edges,
            "graph_density": round(density, 4) if density is not None else None,
            "graph_method_stored": graph_info.get("graph_method"),
        })

        if probs is not None:
            all_probs[(fs, method)] = probs
        if labels is not None:
            all_labels[fs] = labels

    # ── Metrics summary ────────────────────────────────────────────────────────
    metrics_df = pd.DataFrame(metrics_rows).sort_values(["feature_set", "graph_method"])
    metrics_path = os.path.join(out_dir, "gnn_metrics_summary.csv")
    metrics_df.to_csv(metrics_path, index=False, float_format="%.4f")
    print(f"\nWrote {metrics_path}")

    # ── AUROC pivot: rows = feature set, cols = graph method ──────────────────
    pivot = metrics_df.pivot(index="feature_set", columns="graph_method", values="auroc")
    pivot = pivot[[m for m in graph_methods if m in pivot.columns]]  # preserve method order
    print(f"\nAUROC pivot (feature_set × graph_method):")
    print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))

    # Add best-method column
    pivot["best_method"] = pivot.idxmax(axis=1)
    pivot["best_auroc"]  = pivot[graph_methods].max(axis=1)
    pivot_path = os.path.join(out_dir, "gnn_auroc_pivot.csv")
    pivot.to_csv(pivot_path, float_format="%.4f")
    print(f"\nWrote {pivot_path}")

    # ── Graph topology ─────────────────────────────────────────────────────────
    topo_df = pd.DataFrame(topology_rows).sort_values(["feature_set", "graph_method"])
    topo_path = os.path.join(out_dir, "gnn_graph_topology.csv")
    topo_df.to_csv(topo_path, index=False)
    print(f"\nGraph topology:")
    print(topo_df[["feature_set", "graph_method", "n_nodes", "n_edges", "graph_density"]].to_string(index=False))
    print(f"Wrote {topo_path}")

    # ── DeLong pairwise between graph methods per feature set ──────────────────
    delong_rows = []
    for fs in feature_sets:
        fs_probs = {m: all_probs[(fs, m)] for m in graph_methods
                    if (fs, m) in all_probs}
        if fs not in all_labels or len(fs_probs) < 2:
            continue
        delong = pairwise_delong(fs_probs, all_labels[fs])
        for (m1, m2), res in delong.items():
            delong_rows.append({
                "feature_set": fs,
                "method_1":    m1,
                "method_2":    m2,
                "z_stat":      res["z_stat"],
                "p_value":     res["p_value"],
            })

    if delong_rows:
        delong_df = pd.DataFrame(delong_rows).sort_values(["feature_set", "p_value"])
        delong_path = os.path.join(out_dir, "gnn_delong_summary.csv")
        delong_df.to_csv(delong_path, index=False, float_format="%.4f")
        print(f"\nDeLong pairwise (graph methods):")
        print(delong_df.to_string(index=False))
        print(f"Wrote {delong_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
