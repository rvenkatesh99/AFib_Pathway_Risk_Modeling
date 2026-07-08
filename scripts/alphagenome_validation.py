"""
AlphaGenome post-hoc validation of GWAS+PRS pathway model.

Purpose:
  The gwas_prs global_attention model identifies pathways important for AFib risk.
  This script validates those findings by asking: do the GWAS variants contributing
  to top-ranked pathways have larger predicted regulatory effects in cardiac tissue
  than variants in bottom-ranked pathways?

  A significant regulatory enrichment in cardiac tissue (HAA, HLV) for top-pathway
  variants — but not in control tissues (e.g. whole blood) — supports the biological
  relevance of the model's pathway rankings.

Pipeline:
  1. Load pathway importance scores from the gwas_prs global_attention model
  2. Load the variant-to-pathway mapping (from pathway scoring step)
  3. For each variant, fetch AlphaGenome regulatory effect predictions
     (delta track scores between ref and alt alleles)
  4. Aggregate variant scores to pathway level
  5. Test: do top-ranked pathways have higher cardiac regulatory scores?
  6. Output: scatter plot, violin, and summary CSV

Usage:
  python scripts/alphagenome_validation.py \
      --ranking_json   /path/PATHWAY_MODELING/gwas_prs/global_attention/ranking.json \
      --variant_map    /path/GNN_PATHWAY/gwas_prs_variant_pathway_map.tsv \
      --out_dir        /path/figures/alphagenome/ \
      [--top_n_pathways 50] \
      [--tissues HAA HLV AC WB] \
      [--precomputed_scores /path/alphagenome_scores.tsv]

Input formats:
  ranking.json:
    {"prefixed__PATHWAY_NAME": importance_score, ...}

  variant_map (TSV, one row per variant-pathway pair):
    chrom  pos  ref  alt  rsid  pathway  gwas_beta  gwas_p

  precomputed_scores (TSV, written by this script on first run):
    rsid  chrom  pos  ref  alt  tissue  track  delta_score
    (AlphaGenome outputs; cache to avoid re-querying)

AlphaGenome API note:
  This script calls alphagenome.predict_variant_effects() for each unique variant.
  Install: pip install alphagenome  (requires Google DeepMind access)
  The API accepts (chrom, pos, ref, alt) and returns a VariantEffectPrediction
  object with per-track, per-tissue delta scores.
"""

import argparse
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
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
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "figure.dpi":        150,
    "figure.facecolor":  "white",
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "svg.fonttype":      "none",
    "savefig.pad_inches": 0.05,
})

# Cardiac-relevant tissues and control tissues to query from AlphaGenome.
# These map to AlphaGenome tissue/cell-type identifiers.
CARDIAC_TISSUES = {
    "HAA": "heart_left_atrial_appendage",   # GTEx Heart - Left Atrial Appendage
    "HLV": "heart_left_ventricle",          # GTEx Heart - Left Ventricle
    "AC":  "artery_coronary",               # GTEx Artery - Coronary
}
CONTROL_TISSUES = {
    "WB":  "whole_blood",                   # GTEx Whole Blood (non-cardiac control)
    "SK":  "skin_sun_exposed",              # Skin (distal control)
}

# AlphaGenome track types to aggregate for regulatory effect score.
# ATAC-seq and H3K27ac are the most informative for regulatory activity.
REGULATORY_TRACKS = ["ATAC", "H3K27ac", "H3K4me3", "DNase"]


def _save(fig, path):
    stem = os.path.splitext(path)[0]
    fig.tight_layout()
    for ext in (".svg", ".png"):
        fig.savefig(stem + ext, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {stem}.{{svg,png}}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ranking_json", required=True,
                   help="ranking.json from global_attention/gwas_prs run.")
    p.add_argument("--variant_map", required=True,
                   help="TSV: chrom, pos, ref, alt, rsid, pathway, gwas_beta, gwas_p.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--top_n_pathways", type=int, default=50,
                   help="Number of top/bottom pathways to compare (default 50).")
    p.add_argument("--tissues", nargs="+",
                   default=list(CARDIAC_TISSUES.keys()) + list(CONTROL_TISSUES.keys()),
                   help="Tissue codes to score (default: HAA HLV AC WB SK).")
    p.add_argument("--precomputed_scores", default=None,
                   help="TSV of already-computed AlphaGenome scores. If provided, "
                        "skip the API calls and go straight to analysis.")
    p.add_argument("--max_variants_per_pathway", type=int, default=200,
                   help="Cap variants per pathway to limit API calls (take top by |gwas_beta|).")
    return p.parse_args()


# ── Ranking helpers ───────────────────────────────────────────────────────────

def load_ranking(path):
    with open(path) as f:
        raw = json.load(f)
    # Strip source prefix (gwas__, grex_HAA__, etc.)
    return {(k.split("__", 1)[1] if "__" in k else k): v for k, v in raw.items()}


def split_top_bottom(ranking, n):
    """Return (top_n pathway names, bottom_n pathway names) by importance score."""
    ordered = sorted(ranking, key=ranking.get, reverse=True)
    return set(ordered[:n]), set(ordered[-n:])


# ── Variant map loading ───────────────────────────────────────────────────────

def load_variant_map(path, top_pathways, bottom_pathways,
                     max_per_pathway=200):
    """
    Load TSV mapping variants to pathways. Filters to top/bottom pathway variants only,
    capping each pathway at max_per_pathway variants by |gwas_beta|.

    Required columns: chrom, pos, ref, alt, rsid, pathway
    Optional columns: gwas_beta, gwas_p
    """
    df = pd.read_csv(path, sep="\t")
    required = {"chrom", "pos", "ref", "alt", "pathway"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"variant_map missing columns: {missing}")

    if "rsid" not in df.columns:
        df["rsid"] = df["chrom"].astype(str) + ":" + df["pos"].astype(str)
    if "gwas_beta" not in df.columns:
        df["gwas_beta"] = np.nan

    all_pathways = top_pathways | bottom_pathways
    df = df[df["pathway"].isin(all_pathways)].copy()
    df["tier"] = df["pathway"].apply(
        lambda p: "top" if p in top_pathways else "bottom"
    )

    # Cap per pathway by |gwas_beta| if available, else random
    if df["gwas_beta"].notna().any():
        df["_abs_beta"] = df["gwas_beta"].abs()
        df = (df.sort_values("_abs_beta", ascending=False)
                .groupby("pathway", group_keys=False)
                .head(max_per_pathway)
                .drop(columns="_abs_beta"))
    else:
        df = (df.groupby("pathway", group_keys=False)
                .head(max_per_pathway))

    print(f"Loaded {len(df)} variant-pathway pairs "
          f"({df['rsid'].nunique()} unique variants, "
          f"{df['pathway'].nunique()} pathways)")
    return df


# ── AlphaGenome scoring ───────────────────────────────────────────────────────

def score_variants_alphagenome(variants_df, tissues, out_dir):
    """
    Call AlphaGenome to predict regulatory effect of each unique variant in
    the requested tissues. Returns a DataFrame:
      rsid  chrom  pos  ref  alt  tissue  delta_score

    delta_score = mean |delta| across REGULATORY_TRACKS for that tissue,
    representing the overall regulatory disruption magnitude.

    Results are cached to {out_dir}/alphagenome_raw_scores.tsv to avoid
    re-querying on re-runs.
    """
    cache_path = os.path.join(out_dir, "alphagenome_raw_scores.tsv")
    if os.path.isfile(cache_path):
        print(f"  Loading cached AlphaGenome scores: {cache_path}")
        return pd.read_csv(cache_path, sep="\t")

    try:
        import alphagenome  # noqa: F401
        from alphagenome.models import dna_client
    except ImportError:
        raise ImportError(
            "alphagenome package not found. Install with: pip install alphagenome\n"
            "Or provide --precomputed_scores to skip API calls."
        )

    tissue_ids = {
        t: CARDIAC_TISSUES.get(t) or CONTROL_TISSUES.get(t)
        for t in tissues
        if CARDIAC_TISSUES.get(t) or CONTROL_TISSUES.get(t)
    }

    model = dna_client.DnaClient()
    unique_variants = (variants_df[["rsid", "chrom", "pos", "ref", "alt"]]
                       .drop_duplicates("rsid"))

    print(f"  Scoring {len(unique_variants)} unique variants across "
          f"{len(tissue_ids)} tissues via AlphaGenome...")

    rows = []
    for _, v in unique_variants.iterrows():
        try:
            pred = model.predict_variant_effects(
                chrom=str(v["chrom"]),
                pos=int(v["pos"]),
                ref=v["ref"],
                alt=v["alt"],
            )
            for tissue_code, tissue_id in tissue_ids.items():
                track_deltas = []
                for track in REGULATORY_TRACKS:
                    delta = pred.get_delta(tissue=tissue_id, track=track)
                    if delta is not None:
                        track_deltas.append(abs(float(delta)))
                if track_deltas:
                    rows.append({
                        "rsid":        v["rsid"],
                        "chrom":       v["chrom"],
                        "pos":         v["pos"],
                        "ref":         v["ref"],
                        "alt":         v["alt"],
                        "tissue":      tissue_code,
                        "delta_score": np.mean(track_deltas),
                    })
        except Exception as e:
            warnings.warn(f"AlphaGenome failed for {v['rsid']}: {e}")

    scores_df = pd.DataFrame(rows)
    scores_df.to_csv(cache_path, sep="\t", index=False)
    print(f"  Cached raw scores: {cache_path}")
    return scores_df


# ── Analysis ──────────────────────────────────────────────────────────────────

def aggregate_to_pathway(scores_df, variants_df):
    """
    Join variant scores back to pathway tier labels.
    Aggregate: mean delta_score per (pathway, tissue).
    Returns DataFrame: pathway, tier, tissue, mean_delta, n_variants
    """
    merged = scores_df.merge(
        variants_df[["rsid", "pathway", "tier"]].drop_duplicates(),
        on="rsid", how="inner",
    )
    agg = (merged.groupby(["pathway", "tier", "tissue"])
           .agg(mean_delta=("delta_score", "mean"),
                n_variants=("rsid", "nunique"))
           .reset_index())
    return agg


def test_top_vs_bottom(pathway_scores, tissues):
    """
    Mann-Whitney U test: top-pathway mean_delta > bottom-pathway mean_delta.
    One test per tissue. Returns summary DataFrame with FDR correction.
    """
    rows = []
    for tissue in tissues:
        sub = pathway_scores[pathway_scores["tissue"] == tissue]
        top_vals    = sub[sub["tier"] == "top"]["mean_delta"].dropna().values
        bottom_vals = sub[sub["tier"] == "bottom"]["mean_delta"].dropna().values
        if len(top_vals) < 3 or len(bottom_vals) < 3:
            continue
        stat, p = stats.mannwhitneyu(top_vals, bottom_vals, alternative="greater")
        rows.append({
            "tissue":       tissue,
            "n_top":        len(top_vals),
            "n_bottom":     len(bottom_vals),
            "median_top":   np.median(top_vals),
            "median_bottom": np.median(bottom_vals),
            "fold_change":  np.median(top_vals) / max(np.median(bottom_vals), 1e-9),
            "U_stat":       stat,
            "p_value":      p,
        })

    if not rows:
        return pd.DataFrame()

    result_df = pd.DataFrame(rows)
    _, fdr, _, _ = multipletests(result_df["p_value"], method="fdr_bh")
    result_df["fdr"] = fdr
    return result_df.sort_values("p_value")


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_violin(pathway_scores, tissues, out_dir):
    """Violin plot: delta_score distribution for top vs bottom pathways per tissue."""
    cardiac = [t for t in tissues if t in CARDIAC_TISSUES]
    control = [t for t in tissues if t in CONTROL_TISSUES]
    tissue_order = cardiac + control

    fig, axes = plt.subplots(1, len(tissue_order),
                             figsize=(len(tissue_order) * 2.2, 4),
                             sharey=True, squeeze=False)

    colors = {"top": "#2166ac", "bottom": "#d73027"}

    for ax, tissue in zip(axes[0], tissue_order):
        sub = pathway_scores[pathway_scores["tissue"] == tissue]
        data = [sub[sub["tier"] == tier]["mean_delta"].dropna().values
                for tier in ["top", "bottom"]]
        parts = ax.violinplot(data, positions=[0, 1], showmedians=True,
                              showextrema=False)
        for i, (pc, tier) in enumerate(zip(parts["bodies"], ["top", "bottom"])):
            pc.set_facecolor(colors[tier])
            pc.set_alpha(0.75)
        parts["cmedians"].set_color("#333333")
        parts["cmedians"].set_linewidth(1.2)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Top", "Bottom"])
        ax.set_title(tissue)
        # Shade cardiac vs control
        if tissue in CARDIAC_TISSUES:
            ax.set_facecolor("#f0f4fa")

    axes[0][0].set_ylabel("Mean regulatory delta score")
    fig.suptitle(
        "AlphaGenome regulatory effect: top vs bottom pathways\n"
        "(shaded = cardiac tissues)",
        y=1.02,
    )
    _save(fig, os.path.join(out_dir, "01_violin_top_vs_bottom.png"))


def plot_scatter(pathway_scores, ranking, out_dir, tissue="HAA"):
    """
    Scatter: pathway importance rank (x) vs mean AlphaGenome delta score (y)
    for a single cardiac tissue. One dot per pathway.
    """
    sub = pathway_scores[pathway_scores["tissue"] == tissue].copy()
    if sub.empty:
        return

    ordered = sorted(ranking, key=ranking.get, reverse=True)
    rank_map = {pw: i + 1 for i, pw in enumerate(ordered)}
    sub["rank"] = sub["pathway"].map(rank_map)
    sub = sub.dropna(subset=["rank"])

    rho, p = stats.spearmanr(-sub["rank"], sub["mean_delta"])

    fig, ax = plt.subplots(figsize=(5, 4))
    scatter = ax.scatter(sub["rank"], sub["mean_delta"],
                         c=sub["tier"].map({"top": "#2166ac", "bottom": "#d73027"}),
                         s=20, alpha=0.7, linewidths=0)
    ax.set_xlabel("Pathway importance rank (1 = highest)")
    ax.set_ylabel(f"Mean AlphaGenome delta score ({tissue})")
    ax.set_title(f"Pathway rank vs cardiac regulatory impact ({tissue})")
    ax.text(0.97, 0.05,
            f"Spearman ρ = {rho:.3f}\np = {p:.3g}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=c, markersize=6, label=t)
        for t, c in [("Top pathways", "#2166ac"), ("Bottom pathways", "#d73027")]
    ]
    ax.legend(handles=handles)
    _save(fig, os.path.join(out_dir, f"02_scatter_rank_vs_delta_{tissue}.png"))


def plot_summary_bar(test_df, out_dir):
    """
    Bar chart: -log10(FDR) per tissue for top > bottom Mann-Whitney test.
    Cardiac tissues highlighted. Reference line at FDR = 0.05.
    """
    if test_df.empty:
        return

    cardiac_mask = test_df["tissue"].isin(CARDIAC_TISSUES)
    colors = ["#2166ac" if c else "#aaaaaa" for c in cardiac_mask]

    fig, ax = plt.subplots(figsize=(max(3.5, len(test_df) * 0.8), 3.5))
    y = -np.log10(test_df["fdr"].clip(lower=1e-10))
    ax.bar(range(len(test_df)), y, color=colors, edgecolor="none")
    ax.axhline(-np.log10(0.05), color="#888888", linestyle="--", linewidth=0.7,
               label="FDR = 0.05")
    ax.set_xticks(range(len(test_df)))
    ax.set_xticklabels(test_df["tissue"])
    ax.set_ylabel("−log₁₀(FDR)")
    ax.set_title("Regulatory enrichment: top vs bottom pathways\n(blue = cardiac tissue)")
    ax.legend()
    _save(fig, os.path.join(out_dir, "03_fdr_by_tissue.png"))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load ranking ──────────────────────────────────────────────────────────
    ranking = load_ranking(args.ranking_json)
    print(f"Loaded {len(ranking)} pathway scores")

    top_pathways, bottom_pathways = split_top_bottom(ranking, args.top_n_pathways)
    print(f"Top {args.top_n_pathways} / bottom {args.top_n_pathways} pathways selected")

    # ── Load variant-pathway map ───────────────────────────────────────────────
    variants_df = load_variant_map(
        args.variant_map, top_pathways, bottom_pathways,
        max_per_pathway=args.max_variants_per_pathway,
    )

    # ── Score variants ────────────────────────────────────────────────────────
    if args.precomputed_scores:
        print(f"Loading precomputed scores: {args.precomputed_scores}")
        scores_df = pd.read_csv(args.precomputed_scores, sep="\t")
    else:
        scores_df = score_variants_alphagenome(variants_df, args.tissues, args.out_dir)

    if scores_df.empty:
        raise RuntimeError("No AlphaGenome scores produced — check API access and variant input.")

    # ── Aggregate to pathway level ────────────────────────────────────────────
    pathway_scores = aggregate_to_pathway(scores_df, variants_df)
    pathway_scores.to_csv(
        os.path.join(args.out_dir, "pathway_regulatory_scores.csv"),
        index=False, float_format="%.6f",
    )
    print(f"\nPathway-level scores: {len(pathway_scores)} rows")

    # ── Statistical test ──────────────────────────────────────────────────────
    test_df = test_top_vs_bottom(pathway_scores, args.tissues)
    if not test_df.empty:
        test_path = os.path.join(args.out_dir, "top_vs_bottom_test.csv")
        test_df.to_csv(test_path, index=False, float_format="%.6f")
        print("\nTop vs bottom pathway regulatory enrichment:")
        print(test_df.to_string(index=False))
        print(f"Wrote {test_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nPlotting...")
    tissues_present = pathway_scores["tissue"].unique().tolist()
    plot_violin(pathway_scores, tissues_present, args.out_dir)

    primary_cardiac = next(
        (t for t in ["HAA", "HLV", "AC"] if t in tissues_present), None
    )
    if primary_cardiac:
        plot_scatter(pathway_scores, ranking, args.out_dir, tissue=primary_cardiac)

    if not test_df.empty:
        plot_summary_bar(test_df, args.out_dir)

    print(f"\nDone. Results in {args.out_dir}/")


if __name__ == "__main__":
    main()
