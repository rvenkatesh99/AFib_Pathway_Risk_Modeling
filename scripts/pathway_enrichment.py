import argparse
import json
import os
import textwrap
import warnings
from pathlib import Path

import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":          8,
    "axes.labelsize":     9,
    "axes.titlesize":     9,
    "axes.titleweight":   "bold",
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "legend.frameon":     False,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.7,
    "axes.grid":          False,
    "xtick.major.width":  0.7,
    "ytick.major.width":  0.7,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "figure.dpi":         150,
    "figure.facecolor":   "white",
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
    "svg.fonttype":       "none",
    "savefig.pad_inches": 0.05,
})

ENRICHR_LIBRARIES = [
    "MSigDB_Hallmark_2020",
    "GWAS_Catalog_2023",
    "DisGeNET",
    "GO_Molecular_Function_2023",
    "Human_Phenotype_Ontology",
]
# KEGG, Reactome, GO_Biological_Process excluded — circular with input pathway features

CONTINUOUS_MODELS = {"global_attention", "gnn", "random_forest", "transformer"}
SELECTION_MODELS  = {"l1_logistic", "elasticnet", "unregularized_logistic"}

_LIB_SHORT = {
    "MSigDB_Hallmark_2020":       "Hallmark",
    "GWAS_Catalog_2023":          "GWAS",
    "DisGeNET":                   "DisGeNET",
    "GO_Molecular_Function_2023": "GO:MF",
    "Human_Phenotype_Ontology":   "HPO",
}

_LIB_MARKER = {
    "MSigDB_Hallmark_2020":       "o",
    "GWAS_Catalog_2023":          "s",
    "DisGeNET":                   "^",
    "GO_Molecular_Function_2023": "D",
    "Human_Phenotype_Ontology":   "P",
}


def _save(fig, path):
    fig.tight_layout()
    stem = os.path.splitext(path)[0]
    for ext in (".svg", ".png"):
        fig.savefig(stem + ext, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {stem}.{{svg,png}}")


def _gene_count(gene_str):
    if not isinstance(gene_str, str) or not gene_str.strip():
        return 1
    sep = ";" if ";" in gene_str else ","
    return len([g for g in gene_str.split(sep) if g.strip()])


def _size_scale(series, size_min=30, size_max=200):
    rng = series.max() - series.min()
    if rng == 0:
        return pd.Series([size_min + (size_max - size_min) / 2] * len(series))
    return size_min + (series - series.min()) / rng * (size_max - size_min)


def _dot_legends(ax, fig, libs, gc, cmap, norm, size_min=30, size_max=200):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label("−log₁₀(FDR)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    for gc_val, label in zip(
        [gc.min(), int(gc.median()), gc.max()],
        [f"{int(gc.min())} genes", f"{int(gc.median())}", f"{int(gc.max())}"],
    ):
        s = _size_scale(pd.Series([gc.min(), gc_val, gc.max()]), size_min, size_max).iloc[1]
        ax.scatter([], [], s=s, c="#888888", alpha=0.7, label=label)
    size_leg = ax.legend(title="Gene count", fontsize=7, title_fontsize=7,
                         loc="upper right", frameon=True, framealpha=0.8)
    ax.add_artist(size_leg)

    handles = [ax.scatter([], [], marker=_LIB_MARKER.get(l, "o"), c="#888888", s=40, alpha=0.8)
               for l in libs]
    ax.legend(handles, [_LIB_SHORT.get(l, l) for l in libs],
              title="Database", fontsize=7, title_fontsize=7,
              loc="lower right", frameon=True, framealpha=0.8)


def load_ranking(results_dir, model):
    path = Path(results_dir) / model / "ranking.json"
    if not path.exists():
        raise FileNotFoundError(f"ranking.json not found: {path}")
    with open(path) as f:
        return json.load(f)


def strip_prefix(name):
    return name.split("__", 1)[1] if "__" in name else name


def build_gene_scores(ranking, gene_sets, agg="max"):
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
    selected, all_genes = set(), set()
    for pw_name, score in ranking.items():
        clean = strip_prefix(pw_name)
        genes = gene_sets.get(pw_name, gene_sets.get(clean, []))
        all_genes.update(genes)
        if abs(score) > threshold:
            selected.update(genes)
    return selected, all_genes


def run_ora(sel_genes, bg_genes, gene_sets, out_dir, tag, top_n=20):
    out_dir = Path(out_dir)
    print(f"  Running Enrichr ORA ({len(sel_genes)} genes)...")

    results = []
    for lib in ENRICHR_LIBRARIES:
        try:
            enr = gp.enrichr(gene_list=list(sel_genes), gene_sets=lib, outdir=None, verbose=False)
            df = enr.results.copy()
            df["library"] = lib
            results.append(df)
        except Exception as e:
            warnings.warn(f"Enrichr query failed for {lib}: {e}")

    if not results:
        return pd.DataFrame()

    enr_df = pd.concat(results, ignore_index=True).sort_values("Adjusted P-value")
    enr_df.to_csv(out_dir / f"ora_enrichr_{tag}.csv", index=False)
    _plot_ora(enr_df, out_dir, tag, top_n)
    return enr_df


def _plot_ora(enr_df, out_dir, tag, top_n=20):
    sig = enr_df[enr_df["Adjusted P-value"] < 0.05].head(top_n).copy()
    if sig.empty:
        print("  No significant ORA terms (FDR < 0.05)")
        return

    sig["-log10(FDR)"] = -np.log10(sig["Adjusted P-value"].clip(lower=1e-300))
    sig["gene_count"]  = sig["Genes"].apply(_gene_count)
    sig["term_wrapped"] = sig["Term"].apply(lambda t: "\n".join(textwrap.wrap(t, 45)))
    sig = sig.sort_values("-log10(FDR)")

    fig, ax = plt.subplots(figsize=(8, max(4, len(sig) * 0.45)))
    norm = plt.Normalize(sig["-log10(FDR)"].min(), sig["-log10(FDR)"].max())
    cmap = plt.cm.RdBu_r
    sizes = _size_scale(sig["gene_count"])

    for lib in sig["library"].unique():
        sub = sig[sig["library"] == lib]
        ax.scatter(sub["Combined Score"], sub["term_wrapped"],
                   c=sub["-log10(FDR)"], s=sizes[sub.index],
                   cmap=cmap, norm=norm,
                   marker=_LIB_MARKER.get(lib, "o"),
                   alpha=0.85, edgecolors="white", linewidths=0.4, zorder=3)

    ax.set_xlabel("Combined Score")
    ax.set_title(f"ORA — {tag.replace('_', ' ')} (FDR < 0.05)", pad=10)
    ax.grid(axis="x", color="#eeeeee", linewidth=0.5, zorder=0)
    _dot_legends(ax, fig, sig["library"].unique(), sig["gene_count"], cmap, norm)
    _save(fig, str(out_dir / f"ora_plot_{tag}.png"))


def run_preranked_gsea(gene_scores, out_dir, tag, top_n=20):
    out_dir = Path(out_dir)
    print(f"  Running preranked GSEA ({len(gene_scores)} genes)...")

    results = []
    for lib in ENRICHR_LIBRARIES:
        try:
            pre = gp.prerank(rnk=gene_scores, gene_sets=lib, outdir=None,
                             permutation_num=1000, seed=42, verbose=False)
            df = pre.res2d.copy()
            df["library"] = lib
            results.append(df)
        except Exception as e:
            warnings.warn(f"Preranked GSEA failed for {lib}: {e}")

    if not results:
        print("  No GSEA results produced.")
        return pd.DataFrame()

    gsea_df = pd.concat(results, ignore_index=True).sort_values("FDR q-val")
    gsea_df.to_csv(out_dir / f"gsea_preranked_{tag}.csv", index=False)
    _plot_gsea(gsea_df, out_dir, tag, top_n)
    return gsea_df


def _plot_gsea(gsea_df, out_dir, tag, top_n=20):
    sig = gsea_df[gsea_df["FDR q-val"] < 0.25].head(top_n).copy()
    if sig.empty:
        print("  No significant GSEA terms (FDR < 0.25)")
        return

    sig["-log10(FDR)"] = -np.log10(sig["FDR q-val"].clip(lower=1e-300))
    gene_col = next((c for c in ["Lead_genes", "matched_genes", "Genes"] if c in sig.columns), None)
    sig["gene_count"]  = sig[gene_col].apply(_gene_count) if gene_col else 1
    sig["term_wrapped"] = sig["Term"].apply(lambda t: "\n".join(textwrap.wrap(t, 45)))
    sig = sig.sort_values("NES")

    fig, ax = plt.subplots(figsize=(8, max(4, len(sig) * 0.45)))
    norm = plt.Normalize(sig["-log10(FDR)"].min(), sig["-log10(FDR)"].max())
    cmap = plt.cm.RdBu_r
    sizes = _size_scale(sig["gene_count"])

    for lib in sig["library"].unique():
        sub = sig[sig["library"] == lib]
        ax.scatter(sub["NES"], sub["term_wrapped"],
                   c=sub["-log10(FDR)"], s=sizes[sub.index],
                   cmap=cmap, norm=norm,
                   marker=_LIB_MARKER.get(lib, "o"),
                   alpha=0.85, edgecolors="white", linewidths=0.4, zorder=3)

    ax.axvline(0, color="#444444", linewidth=0.8, zorder=2)
    ax.set_xlabel("Normalized Enrichment Score (NES)")
    ax.set_title(f"Preranked GSEA — {tag.replace('_', ' ')} (FDR < 0.25)", pad=10)
    ax.grid(axis="x", color="#eeeeee", linewidth=0.5, zorder=0)
    _dot_legends(ax, fig, sig["library"].unique(), sig["gene_count"], cmap, norm)
    _save(fig, str(out_dir / f"gsea_plot_{tag}.png"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir",         required=True)
    p.add_argument("--model",               required=True)
    p.add_argument("--gene_sets_json",      required=True)
    p.add_argument("--out_dir",             required=True)
    p.add_argument("--selection_threshold", type=float, default=0.0)
    p.add_argument("--score_agg",           default="max", choices=["mean", "max"])
    p.add_argument("--top_n",               type=int, default=20)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ranking = load_ranking(args.results_dir, args.model)
    with open(args.gene_sets_json) as f:
        gene_sets = json.load(f)

    feature_set = Path(args.results_dir).name
    tag = f"{feature_set}__{args.model}"
    print(f"[enrichment] {feature_set} / {args.model} | {len(ranking)} pathways")

    if args.model in SELECTION_MODELS:
        sel_genes, bg_genes = selected_genes(ranking, gene_sets, args.selection_threshold)
        n_sel = sum(1 for v in ranking.values() if abs(v) > args.selection_threshold)
        print(f"  Selected pathways: {n_sel}/{len(ranking)} | genes: {len(sel_genes)}/{len(bg_genes)}")
        if not sel_genes:
            print("  No pathways selected — nothing to test.")
            return
        run_ora(sel_genes, bg_genes, gene_sets, out_dir, tag, args.top_n)
    else:
        gene_scores = build_gene_scores(ranking, gene_sets, agg=args.score_agg)
        print(f"  Gene-level scores: {len(gene_scores)} genes (agg={args.score_agg})")
        run_preranked_gsea(gene_scores, out_dir, tag, args.top_n)


if __name__ == "__main__":
    main()
