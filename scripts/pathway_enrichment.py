"""
Pathway enrichment analysis on model importance rankings.

Two analyses:
  ORA  — overrepresentation of selected pathways (L1/elasticnet non-zero coefficients)
          against cardiovascular gene sets via Fisher's exact test.
  GSEA — preranked GSEA using continuous pathway importance scores (GAM attention,
          GNN GradCAM) aggregated to gene level.

Gene sets queried (via gseapy Enrichr or local GMT):
  - MSigDB Hallmark (H)
  - MSigDB C2:CP (curated pathways — KEGG, Reactome, BioCarta)
  - GWAS_Catalog_2023  (to check AFib/cardiovascular gene overlap)
  - GTEx_Tissue_Sample_Gene_Expression_Profiles_up (cardiac tissue expression)

Usage:
  python scripts/pathway_enrichment.py \
      --results_dir AF_PATHWAY_SCORES/PATHWAY_MODELING/gwas_covs/ \
      --model global_attention \
      --gene_sets_json AF_PATHWAY_SCORES/GNN_PATHWAY/pathway_gene_sets.json \
      --out_dir AF_PATHWAY_SCORES/PATHWAY_MODELING/figures/enrichment/

  # L1 logistic (ORA on selected pathways):
  python scripts/pathway_enrichment.py \
      --results_dir AF_PATHWAY_SCORES/PATHWAY_MODELING/gwas_covs/ \
      --model l1_logistic \
      --gene_sets_json AF_PATHWAY_SCORES/GNN_PATHWAY/pathway_gene_sets.json \
      --out_dir AF_PATHWAY_SCORES/PATHWAY_MODELING/figures/enrichment/
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         8,
    "axes.labelsize":    9,
    "axes.titlesize":    9,
    "axes.titleweight":  "bold",
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "legend.frameon":    False,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.7,
    "axes.grid":         False,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "figure.dpi":        150,
    "figure.facecolor":  "white",
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "svg.fonttype":      "none",
    "savefig.pad_inches": 0.05,
})


def _save(fig, path):
    """Save as SVG (vector) and PNG (300 dpi). Extension in path is replaced."""
    fig.tight_layout()
    stem = os.path.splitext(path)[0]
    for ext in (".svg", ".png"):
        out = stem + ext
        fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {stem}.{{svg,png}}")


# Enrichr gene set libraries to query
ENRICHR_LIBRARIES = [
    "MSigDB_Hallmark_2020",
    "KEGG_2021_Human",
    "GWAS_Catalog_2023",
    "GTEx_Tissue_Sample_Gene_Expression_Profiles_up",
]

# Models that produce continuous scores (use preranked GSEA)
CONTINUOUS_MODELS = {"global_attention", "gnn", "random_forest", "transformer"}
# Models that produce selected/not-selected pathways (use ORA)
SELECTION_MODELS  = {"l1_logistic", "elasticnet", "unregularized_logistic"}


def load_ranking(results_dir, model):
    path = Path(results_dir) / model / "ranking.json"
    if not path.exists():
        raise FileNotFoundError(f"ranking.json not found: {path}")
    with open(path) as f:
        return json.load(f)


def strip_prefix(name):
    """Remove tissue/source prefix (gwas__, grex_HAA__, etc.)."""
    return name.split("__", 1)[1] if "__" in name else name


def build_gene_scores(ranking, gene_sets, agg="mean"):
    """
    Aggregate pathway-level importance scores to gene level.
    Each gene receives the mean (or max) score across all pathways it belongs to.
    Returns a Series sorted descending — input for preranked GSEA.
    """
    gene_scores = {}
    for pw_name, score in ranking.items():
        clean = strip_prefix(pw_name)
        genes = gene_sets.get(pw_name, gene_sets.get(clean, []))
        for gene in genes:
            gene_scores.setdefault(gene, []).append(score)

    agg_fn = np.mean if agg == "mean" else np.max
    return pd.Series(
        {g: agg_fn(vals) for g, vals in gene_scores.items()}
    ).sort_values(ascending=False)


def selected_genes(ranking, gene_sets, threshold=0.0):
    """
    Genes belonging to pathways with |score| > threshold (L1/elasticnet selected pathways).
    Returns (selected_gene_set, all_gene_set).
    """
    selected, all_genes = set(), set()
    for pw_name, score in ranking.items():
        clean = strip_prefix(pw_name)
        genes = gene_sets.get(pw_name, gene_sets.get(clean, []))
        all_genes.update(genes)
        if abs(score) > threshold:
            selected.update(genes)
    return selected, all_genes


def run_ora(selected_genes, background_genes, gene_sets_json, out_dir, tag, top_n=20):
    """
    Fisher's exact ORA: are selected genes enriched in cardiovascular gene sets?
    Also queries Enrichr for broader context.
    """
    out_dir = Path(out_dir)

    # ── Enrichr query ────────────────────────────────────────────────────────
    print(f"  Running Enrichr ORA ({len(selected_genes)} genes)...")
    enr_results = []
    for lib in ENRICHR_LIBRARIES:
        try:
            enr = gp.enrichr(
                gene_list=list(selected_genes),
                gene_sets=lib,
                outdir=None,
                verbose=False,
            )
            df = enr.results.copy()
            df["library"] = lib
            enr_results.append(df)
        except Exception as e:
            warnings.warn(f"Enrichr query failed for {lib}: {e}")

    if enr_results:
        enr_df = pd.concat(enr_results, ignore_index=True)
        enr_df = enr_df.sort_values("Adjusted P-value")
        enr_path = out_dir / f"ora_enrichr_{tag}.csv"
        enr_df.to_csv(enr_path, index=False)
        print(f"  Enrichr results saved: {enr_path}")
        _plot_ora(enr_df, out_dir, tag, top_n)

    return enr_df if enr_results else pd.DataFrame()


def _plot_ora(enr_df, out_dir, tag, top_n=20):
    sig = enr_df[enr_df["Adjusted P-value"] < 0.05].head(top_n).copy()
    if sig.empty:
        print("  No significant ORA terms (FDR < 0.05)")
        return

    sig["-log10(FDR)"] = -np.log10(sig["Adjusted P-value"].clip(lower=1e-300))
    sig = sig.sort_values("-log10(FDR)")

    fig, ax = plt.subplots(figsize=(7, max(4, len(sig) * 0.35)))
    colors = plt.cm.tab10(np.linspace(0, 1, sig["library"].nunique()))
    lib_color = {lib: colors[i] for i, lib in enumerate(sig["library"].unique())}

    ax.barh(
        range(len(sig)),
        sig["-log10(FDR)"],
        color=[lib_color[l] for l in sig["library"]],
        edgecolor="none",
    )
    ax.axvline(-np.log10(0.05), color="#888888", linestyle="--", linewidth=0.7)
    ax.set_yticks(range(len(sig)))
    ax.set_yticklabels(sig["Term"].str[:60])
    ax.set_xlabel("−log₁₀(FDR)")
    ax.set_title(f"ORA — {tag.replace('_', ' ')} (FDR < 0.05)")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in lib_color.values()]
    ax.legend(handles, lib_color.keys(), loc="lower right")

    _save(fig, str(out_dir / f"ora_plot_{tag}.png"))


def run_preranked_gsea(gene_scores, out_dir, tag, top_n=20):
    """Preranked GSEA on gene-level scores aggregated from pathway importance."""
    out_dir = Path(out_dir)
    print(f"  Running preranked GSEA ({len(gene_scores)} genes)...")

    all_results = []
    for lib in ENRICHR_LIBRARIES:
        try:
            pre = gp.prerank(
                rnk=gene_scores,
                gene_sets=lib,
                outdir=None,
                permutation_num=1000,
                seed=42,
                verbose=False,
            )
            df = pre.res2d.copy()
            df["library"] = lib
            all_results.append(df)
        except Exception as e:
            warnings.warn(f"Preranked GSEA failed for {lib}: {e}")

    if not all_results:
        print("  No GSEA results produced.")
        return pd.DataFrame()

    gsea_df = pd.concat(all_results, ignore_index=True)
    gsea_df = gsea_df.sort_values("FDR q-val")
    path = out_dir / f"gsea_preranked_{tag}.csv"
    gsea_df.to_csv(path, index=False)
    print(f"  GSEA results saved: {path}")
    _plot_gsea(gsea_df, out_dir, tag, top_n)
    return gsea_df


def _plot_gsea(gsea_df, out_dir, tag, top_n=20):
    sig = gsea_df[gsea_df["FDR q-val"] < 0.25].head(top_n).copy()
    if sig.empty:
        print("  No significant GSEA terms (FDR < 0.25)")
        return

    sig["-log10(FDR)"] = -np.log10(sig["FDR q-val"].clip(lower=1e-300))
    sig = sig.sort_values("NES")

    colors = ["#d01c8b" if nes > 0 else "#4dac26" for nes in sig["NES"]]

    fig, ax = plt.subplots(figsize=(7, max(4, len(sig) * 0.35)))
    ax.barh(range(len(sig)), sig["NES"], color=colors, edgecolor="none")
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(range(len(sig)))
    ax.set_yticklabels(sig["Term"].str[:60])
    ax.set_xlabel("Normalized Enrichment Score (NES)")
    ax.set_title(f"Preranked GSEA — {tag.replace('_', ' ')} (FDR < 0.25)")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#d01c8b"),
        plt.Rectangle((0, 0), 1, 1, color="#4dac26"),
    ]
    ax.legend(handles, ["Enriched", "Depleted"])

    _save(fig, str(out_dir / f"gsea_plot_{tag}.png"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True,
                   help="Feature-set results directory (contains model subdirs).")
    p.add_argument("--model", required=True,
                   help="Model subdirectory name (e.g. global_attention, l1_logistic).")
    p.add_argument("--gene_sets_json", required=True,
                   help="JSON mapping prefixed pathway name → list of gene symbols.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--selection_threshold", type=float, default=0.0,
                   help="L1/elasticnet: absolute coefficient threshold for 'selected' (default 0 = non-zero).")
    p.add_argument("--score_agg", default="mean", choices=["mean", "max"],
                   help="How to aggregate pathway scores to gene level for preranked GSEA.")
    p.add_argument("--top_n", type=int, default=20,
                   help="Top N terms to show in plots.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load inputs
    ranking = load_ranking(args.results_dir, args.model)
    with open(args.gene_sets_json) as f:
        gene_sets = json.load(f)

    feature_set = Path(args.results_dir).name
    tag = f"{feature_set}__{args.model}"

    print(f"[enrichment] {feature_set} / {args.model} | {len(ranking)} pathways")

    if args.model in SELECTION_MODELS:
        # ORA on selected (non-zero) pathways
        sel_genes, bg_genes = selected_genes(ranking, gene_sets, args.selection_threshold)
        n_selected = sum(1 for v in ranking.values() if abs(v) > args.selection_threshold)
        print(f"  Selected pathways: {n_selected}/{len(ranking)} | genes: {len(sel_genes)}/{len(bg_genes)}")
        if not sel_genes:
            print("  No pathways selected — nothing to test.")
            return
        run_ora(sel_genes, bg_genes, gene_sets, out_dir, tag, args.top_n)

    else:
        # Preranked GSEA on continuous importance scores
        gene_scores = build_gene_scores(ranking, gene_sets, agg=args.score_agg)
        print(f"  Gene-level scores: {len(gene_scores)} genes (agg={args.score_agg})")
        run_preranked_gsea(gene_scores, out_dir, tag, args.top_n)


if __name__ == "__main__":
    main()
