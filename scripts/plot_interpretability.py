"""
Interpretability plots for pathway risk modeling.

Reads output from aggregate_results.py:
  - metrics_summary.csv
  - ranking_percentile.csv

Produces:
  01_auroc_comparison.pdf       — AUROC + 95% CI per model, grouped by feature set
  02_auroc_heatmap.pdf          — AUROC heatmap: rows=feature sets, cols=models
  03_multimodel_dotplot.pdf     — Top-N features by percentile rank across models (per feature set)
  04_cross_tissue_heatmap.pdf   — Per-model heatmap: rows=pathways, cols=GREx tissues
  05_delong_heatmap.pdf         — DeLong p-value heatmap within each feature set

Usage:
  python scripts/plot_interpretability.py \
      --agg_dir  /path/to/aggregated/ \
      --out_dir  /path/to/figures/ \
      [--top_n 30] \
      [--feature_sets gwas_covs grex_AC_covs ...]  # subset to plot; default: all
"""

import argparse
import os
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# ── Shared style ──────────────────────────────────────────────────────────────

MODEL_ORDER = [
    "l1_logistic", "elasticnet",
    "random_forest", "global_attention", "transformer", "gnn",
    "gnn_jaccard", "gnn_score_correlation", "gnn_string", "gnn_fully_connected",
]

MODEL_COLORS = {
    "l1_logistic":            "#b2abd2",
    "elasticnet":             "#8073ac",
    "random_forest":          "#f4a582",
    "global_attention":       "#2166ac",
    "transformer":            "#4dac26",
    "gnn":                    "#d01c8b",
    "gnn_jaccard":            "#d01c8b",
    "gnn_score_correlation":  "#e56fac",
    "gnn_string":             "#f0a0cc",
    "gnn_fully_connected":    "#c0007a",
}

# Logical display order for feature set rows (actual names in metrics CSV).
# Rows not in this list are appended alphabetically at the end.
PREFERRED_ROW_ORDER = [
    "covs_only",
    "prs_only",
    "prs",
    "gwas",
    "gwas_covs",
    "grex_AA_covs",
    "grex_AC_covs",
    "grex_HAA_covs",
    "grex_HLV_covs",
    "grex_WB_covs",
    "grex_all_covs",
    "gwas_grex_HAA_HLV_covs",
    "gwas_grex_HAA_HLV_AA_covs",
    "gwas_prs",
    "gwas_grex_HAA_HLV_prs",
    "gwas_grex_HAA_HLV_AA_prs",
]

# Graph method suffixes added to GNN results directories
_GNN_GRAPH_SUFFIXES = [
    "_fully_connected", "_jaccard", "_score_correlation", "_string",
]


def _normalize_gnn_rows(df):
    """Strip graph method suffix from GNN feature_set names, encode in model column.

    e.g. feature_set="gwas_prs_jaccard", model="gnn"
      -> feature_set="gwas_prs", model="gnn_jaccard"
    """
    df = df.copy()
    for suffix in _GNN_GRAPH_SUFFIXES:
        method = suffix.lstrip("_")
        mask = (df["model"] == "gnn") & df["feature_set"].str.endswith(suffix)
        df.loc[mask, "feature_set"] = df.loc[mask, "feature_set"].str[: -len(suffix)]
        df.loc[mask, "model"] = "gnn_" + method
    return df


def _ordered_feature_sets(all_fs):
    """Return feature sets in logical display order."""
    preferred = [fs for fs in PREFERRED_ROW_ORDER if fs in all_fs]
    extra = sorted(fs for fs in all_fs if fs not in set(PREFERRED_ROW_ORDER))
    return preferred + extra

GREX_TISSUES = ["grex_AC_covs", "grex_AA_covs", "grex_HAA_covs", "grex_HLV_covs", "grex_WB_covs"]
TISSUE_LABELS = {
    "grex_AC_covs":  "Artery\nCoronary",
    "grex_AA_covs":  "Artery\nAorta",
    "grex_HAA_covs": "Heart\nAtrial App.",
    "grex_HLV_covs": "Heart\nLeft Vent.",
    "grex_WB_covs":  "Whole\nBlood",
}


def _style():
    plt.rcParams.update({
        # Fonts
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
        "legend.borderpad":  0.4,
        # Axes
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    0.7,
        "axes.grid":         False,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size":  3,
        "ytick.major.size":  3,
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        # Figure
        "figure.dpi":        150,
        "figure.facecolor":  "white",
        # Font embedding: TrueType in PDF/PS, editable text in SVG
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
        "svg.fonttype":      "none",
        "savefig.pad_inches": 0.05,
    })


def _save(fig, path):
    """Save as SVG (vector) and PNG (300 dpi raster). Extension in path is replaced."""
    fig.tight_layout()
    stem = os.path.splitext(path)[0]
    for ext in (".svg", ".png"):
        out = stem + ext
        fig.savefig(out, bbox_inches="tight", dpi=300)
        print(f"  Saved: {out}")
    plt.close(fig)


def _model_order(models):
    """Sort models in canonical display order."""
    known = [m for m in MODEL_ORDER if m in models]
    extra = [m for m in sorted(models) if m not in MODEL_ORDER]
    return known + extra


# ── 1. AUROC comparison — grouped bar chart ───────────────────────────────────

def plot_auroc_comparison(metrics_df, feature_sets, out_path):
    models = _model_order(metrics_df["model"].unique())
    n_fs   = len(feature_sets)
    n_m    = len(models)

    x       = np.arange(n_fs)
    width   = 0.75 / n_m
    offsets = np.linspace(-(0.75 - width) / 2, (0.75 - width) / 2, n_m)

    fig, ax = plt.subplots(figsize=(max(7, n_fs * 1.3), 4.5))

    for i, model in enumerate(models):
        sub = metrics_df[metrics_df["model"] == model].set_index("feature_set")
        aurocs   = [sub.loc[fs, "auroc"]          if fs in sub.index else np.nan for fs in feature_sets]
        ci_lower = [sub.loc[fs, "auroc_ci_lower"]  if fs in sub.index else np.nan for fs in feature_sets]
        ci_upper = [sub.loc[fs, "auroc_ci_upper"]  if fs in sub.index else np.nan for fs in feature_sets]
        yerr_lo  = np.array([a - l if not (np.isnan(a) or np.isnan(l)) else 0
                             for a, l in zip(aurocs, ci_lower)])
        yerr_hi  = np.array([u - a if not (np.isnan(a) or np.isnan(u)) else 0
                             for a, u in zip(aurocs, ci_upper)])

        ax.bar(x + offsets[i], aurocs, width,
               color=MODEL_COLORS.get(model, "#aaaaaa"),
               edgecolor="none", label=model.replace("_", " "), zorder=2)
        ax.errorbar(x + offsets[i], aurocs,
                    yerr=[yerr_lo, yerr_hi],
                    fmt="none", color="#333333", linewidth=0.7, capsize=1.5, zorder=3)

    ax.axhline(0.5, color="#888888", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([fs.replace("_covs", "").replace("_", "\n") for fs in feature_sets])
    ax.set_ylabel("AUROC")
    finite_auroc = metrics_df["auroc"].dropna()
    ax.set_ylim(bottom=max(0.4, finite_auroc.min() - 0.05),
                top=min(1.0, finite_auroc.max() + 0.04))
    ax.set_title("AUROC by feature set and model (95% CI)")
    ax.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left",
              title="Model", title_fontsize=plt.rcParams["legend.fontsize"])
    _save(fig, out_path)


# ── 2. AUROC heatmap ──────────────────────────────────────────────────────────

def _draw_heatmap(ax, pivot, feature_sets, models, title, vmin=None, vmax=None):
    """Render a single AUROC heatmap panel onto ax."""
    assert pivot.shape[0] == len(feature_sets), (
        f"pivot rows ({pivot.shape[0]}) != feature_sets ({len(feature_sets)})"
    )
    finite = pivot.values[~np.isnan(pivot.values)]
    if len(finite) == 0:
        ax.set_visible(False)
        return None
    if vmin is None:
        vmin = max(0.5, finite.min() - 0.02)
    if vmax is None:
        vmax = min(1.0, finite.max() + 0.02)
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)
    for i in range(len(feature_sets)):
        for j in range(len(models)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                bg = (v - vmin) / (vmax - vmin)
                color = "white" if bg < 0.3 or bg > 0.85 else "black"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=6.5,
                        color=color)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace("_", "\n") for m in models])
    ax.set_yticks(range(len(feature_sets)))
    ax.set_yticklabels([fs.replace("_covs", "").replace("_", " ") for fs in feature_sets])
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel("Feature set")
    # Remove tick marks — cell boundaries serve as guides
    ax.tick_params(length=0)
    return im


def plot_auroc_heatmap(metrics_df, feature_sets, out_path):
    # GNN results are reported in a separate table; exclude from this heatmap.
    GNN_MODELS = {m for m in metrics_df["model"].unique() if m.startswith("gnn")}
    all_models  = _model_order(metrics_df["model"].unique())
    main_models = [m for m in all_models if m not in GNN_MODELS]

    pivot_main = (metrics_df[~metrics_df["model"].isin(GNN_MODELS)]
                  .pivot(index="feature_set", columns="model", values="auroc")
                  .reindex(index=feature_sets, columns=main_models))

    # Drop rows that are entirely NaN (e.g. GNN-only feature sets)
    valid_rows = pivot_main.index[~pivot_main.isnull().all(axis=1)]
    pivot_main = pivot_main.loc[valid_rows]
    feature_sets = list(valid_rows)

    # Colour scale: tight range around the actual data so differences are visible.
    # Clamp lower bound at 0.5 (chance level) so red still means "near chance".
    all_finite = pivot_main.values[~np.isnan(pivot_main.values)]
    if len(all_finite):
        data_min, data_max = all_finite.min(), all_finite.max()
        span = max(data_max - data_min, 0.04)   # at least 4 pp of range
        vmin = max(0.5, data_min - span * 0.1)
        vmax = min(1.0, data_max + span * 0.1)
    else:
        vmin, vmax = 0.5, 1.0

    fig, ax = plt.subplots(
        figsize=(max(6, len(main_models) * 1.3), max(4, len(feature_sets) * 0.55))
    )
    im = _draw_heatmap(ax, pivot_main, feature_sets, main_models,
                       "AUROC by feature set and model", vmin=vmin, vmax=vmax)
    if im is not None:
        cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("AUROC", fontsize=9)
        # Tick marks every 0.01 in the visible range
        ticks = np.round(np.linspace(vmin, vmax, 6), 3)
        cbar.set_ticks(ticks)

    _save(fig, out_path)


def plot_gnn_table(metrics_df, topology_df, out_path):
    """
    Side-by-side summary for GNN models:
      Left : AUROC (feature_set × graph_method heatmap)
      Right: Graph topology (n_edges per feature_set × graph_method)

    metrics_df  — from gnn_metrics_summary.csv (feature_set, graph_method, auroc, …)
    topology_df — from gnn_graph_topology.csv  (feature_set, graph_method, n_edges, …)
                  Pass None to skip the topology panel.
    """
    GNN_METHOD_ORDER = ["jaccard", "score_correlation", "string", "fully_connected"]

    gnn_methods = [m for m in GNN_METHOD_ORDER if m in metrics_df["graph_method"].unique()]
    feature_sets = sorted(metrics_df["feature_set"].unique())

    pivot_auroc = (metrics_df
                   .pivot(index="feature_set", columns="graph_method", values="auroc")
                   .reindex(index=feature_sets, columns=gnn_methods))

    has_topo = topology_df is not None and len(topology_df) > 0
    ncols = 2 if has_topo else 1
    fig, axes = plt.subplots(
        1, ncols,
        figsize=(max(5, len(gnn_methods) * 1.8 + 2) * ncols,
                 max(4, len(feature_sets) * 0.55)),
    )
    if ncols == 1:
        axes = [axes]

    # ── AUROC heatmap ──────────────────────────────────────────────────────
    all_finite = pivot_auroc.values[~np.isnan(pivot_auroc.values)]
    if len(all_finite):
        data_min, data_max = all_finite.min(), all_finite.max()
        span = max(data_max - data_min, 0.04)
        vmin = max(0.5, data_min - span * 0.1)
        vmax = min(1.0, data_max + span * 0.1)
    else:
        vmin, vmax = 0.5, 1.0

    im = _draw_heatmap(axes[0], pivot_auroc, feature_sets, gnn_methods,
                       "GNN AUROC (feature set × graph method)", vmin=vmin, vmax=vmax)
    if im is not None:
        cbar = plt.colorbar(im, ax=axes[0], fraction=0.03, pad=0.02)
        cbar.set_label("AUROC", fontsize=9)
        cbar.set_ticks(np.round(np.linspace(vmin, vmax, 6), 3))

    # ── Topology heatmap ───────────────────────────────────────────────────
    if has_topo:
        pivot_edges = (topology_df
                       .pivot(index="feature_set", columns="graph_method", values="n_edges")
                       .reindex(index=feature_sets, columns=gnn_methods))
        ax2 = axes[1]
        vals = pivot_edges.values.astype(float)
        vmax_e = np.nanmax(vals) if not np.all(np.isnan(vals)) else 1
        im2 = ax2.imshow(vals, aspect="auto", cmap="Blues", vmin=0, vmax=vmax_e)
        ax2.set_xticks(range(len(gnn_methods)))
        ax2.set_xticklabels([m.replace("_", "\n") for m in gnn_methods])
        ax2.set_yticks(range(len(feature_sets)))
        ax2.set_yticklabels([fs.replace("_", " ") for fs in feature_sets])
        ax2.tick_params(length=0)
        ax2.set_title("Graph connectivity (edges)")
        for i in range(len(feature_sets)):
            for j in range(len(gnn_methods)):
                v = vals[i, j]
                if not np.isnan(v):
                    ax2.text(j, i, f"{int(v):,}", ha="center", va="center",
                             fontsize=6.5, color="white" if v > vmax_e * 0.6 else "black")
        plt.colorbar(im2, ax=ax2, fraction=0.03, pad=0.02, label="n_edges")

    plt.tight_layout()
    _save(fig, out_path)


def plot_auroc_dotplot(metrics_df, feature_sets, out_path):
    """
    Forest-plot style dot plot: one row per (feature_set, model),
    x-axis = AUROC with 95% CI whiskers. Feature sets grouped with light shading.
    """
    models = _model_order(metrics_df["model"].unique())

    fig, ax = plt.subplots(figsize=(5, max(3.5, len(feature_sets) * len(models) * 0.18 + 1)))

    y_ticks, y_labels = [], []
    row = 0
    for fi, fs in enumerate(feature_sets):
        sub = metrics_df[metrics_df["feature_set"] == fs].set_index("model")
        group_rows = []
        # Alternate shading every other feature set
        row_start = row
        for model in models:
            if model not in sub.index or pd.isna(sub.loc[model, "auroc"]):
                row += 1
                continue
            r = sub.loc[model]
            auroc = r["auroc"]
            lo, hi = r.get("auroc_ci_lower", np.nan), r.get("auroc_ci_upper", np.nan)
            color = MODEL_COLORS.get(model, "#aaaaaa")
            ax.plot(auroc, row, "o", color=color, markersize=4, zorder=3, clip_on=False)
            if not pd.isna(lo) and not pd.isna(hi):
                ax.plot([lo, hi], [row, row], "-", color=color, linewidth=1.2, zorder=2, alpha=0.8)
            group_rows.append(row)
            row += 1

        if group_rows:
            mid = (group_rows[0] + group_rows[-1]) / 2
            y_ticks.append(mid)
            y_labels.append(fs.replace("_covs", "").replace("_", " "))
            if fi % 2 == 0:
                ax.axhspan(group_rows[0] - 0.5, group_rows[-1] + 0.5,
                           color="#f5f5f5", zorder=0, linewidth=0)
        row += 0.6  # gap between feature-set groups

    finite_auroc = metrics_df["auroc"].dropna()
    ax.axvline(0.5, color="#888888", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.set_xlim(max(0.45, finite_auroc.min() - 0.02),
                min(1.0,  finite_auroc.max() + 0.03))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.invert_yaxis()
    ax.set_xlabel("AUROC (95% CI)")
    ax.set_title("Model performance by feature set")

    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=MODEL_COLORS.get(m, "#aaa"),
                          markersize=5, label=m.replace("_", " ")) for m in models]
    ax.legend(handles=handles, frameon=False,
              bbox_to_anchor=(1.01, 1), loc="upper left",
              title="Model", title_fontsize=plt.rcParams["legend.fontsize"])
    _save(fig, out_path)


# ── 3. Multi-model feature dot plot (per feature set) ─────────────────────────

def plot_multimodel_dotplot(ranking_df, feature_set, models, top_n, out_path):
    """
    ranking_df: ranking_percentile.csv loaded as multi-index columns (feature_set, model)
    Shows top_n features (by max percentile rank across models) for one feature set.
    """
    fs_cols = [(fs, m) for (fs, m) in ranking_df.columns if fs == feature_set and m in models]
    if not fs_cols:
        print(f"  Skipping dot plot for {feature_set}: no ranking data")
        return

    sub = ranking_df[fs_cols].copy()
    sub.columns = [m for (_, m) in fs_cols]
    sub = sub.dropna(how="all")

    # Select top_n features by max percentile across models
    sub["_max"] = sub.max(axis=1)
    top_features = sub.nlargest(top_n, "_max").drop(columns="_max")
    top_features = top_features.sort_values(top_features.columns[0])  # sort by first model

    models_present = list(top_features.columns)
    n_m = len(models_present)
    n_f = len(top_features)
    if n_f == 0:
        return

    fig, ax = plt.subplots(figsize=(max(5, n_m * 1.4), max(5, n_f * 0.32)))

    for xi, model in enumerate(models_present):
        for yi, (feature, row) in enumerate(top_features.iterrows()):
            pct = row[model]
            if np.isnan(pct):
                continue
            size  = max(10, 250 * (pct / 100) ** 2)
            alpha = max(0.2, pct / 100)
            ax.scatter(xi, yi, s=size, color=MODEL_COLORS.get(model, "#888888"),
                       alpha=alpha, linewidths=0, zorder=2)
            if pct >= 80:
                ax.text(xi, yi, f"{pct:.0f}", ha="center", va="center",
                        fontsize=5, color="white")

    ax.set_xticks(range(n_m))
    ax.set_xticklabels([m.replace("_", "\n") for m in models_present])
    ax.set_yticks(range(n_f))
    ax.set_yticklabels([f.replace("_", " ") for f in top_features.index])
    ax.set_xlim(-0.6, n_m - 0.4)
    ax.set_ylim(-0.6, n_f - 0.4)
    ax.tick_params(length=0)
    ax.set_title(
        f"{feature_set.replace('_covs', '').replace('_', ' ')} — top {top_n} pathways\n"
        f"(dot size/opacity = within-model percentile rank)"
    )
    _save(fig, out_path)


# ── 4. Cross-tissue pathway heatmap (per model) ───────────────────────────────

def plot_cross_tissue_heatmap(ranking_df, model, top_n, out_path):
    """
    For a single model, show how pathway percentile ranks vary across GREx tissues.
    Rows = top pathways, columns = tissues.
    """
    tissue_cols = [(fs, m) for (fs, m) in ranking_df.columns
                   if fs in GREX_TISSUES and m == model]
    if len(tissue_cols) < 2:
        print(f"  Skipping cross-tissue heatmap for {model}: fewer than 2 tissues available")
        return

    sub = ranking_df[tissue_cols].copy()
    sub.columns = [TISSUE_LABELS.get(fs, fs) for (fs, _) in tissue_cols]
    sub = sub.dropna(how="all")

    # Top pathways by mean percentile across tissues
    sub["_mean"] = sub.mean(axis=1)
    top = sub.nlargest(top_n, "_mean").drop(columns="_mean")
    top = top.sort_values(top.columns[0])

    if len(top) == 0:
        return

    n_tissues = len(top.columns)
    n_pathways = len(top)
    # Wide enough for labels: left margin + heatmap columns + colorbar
    fig, ax = plt.subplots(figsize=(max(7, n_tissues * 1.8 + 4), max(6, n_pathways * 0.55)))
    im = ax.imshow(top.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)

    for i in range(n_pathways):
        for j in range(n_tissues):
            v = top.values[i, j]
            if not np.isnan(v):
                color = "white" if v > 70 else "black"
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color=color)

    ax.set_xticks(range(n_tissues))
    ax.set_xticklabels(top.columns)
    ax.set_yticks(range(n_pathways))
    ax.tick_params(length=0)

    def _wrap(name, width=45):
        return "\n".join(textwrap.wrap(name.replace("_", " "), width=width))

    ax.set_yticklabels([_wrap(f) for f in top.index])
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Percentile rank")
    ax.set_title(
        f"{model.replace('_', ' ')} — pathway ranks across GREx tissues\n(top {top_n} by mean rank)"
    )
    _save(fig, out_path)


# ── 5. Delta AUROC vs baseline ───────────────────────────────────────────────

def plot_delta_auroc(metrics_df, feature_sets, out_path):
    """
    Grouped bar chart of delta AUROC vs covariates_logistic baseline.
    Positive = improvement over baseline; zero line marked.
    """
    if "delta_auroc_vs_baseline" not in metrics_df.columns:
        print("  Skipping delta AUROC plot: column not found (rerun aggregate_results.py)")
        return

    models = _model_order(metrics_df["model"].unique())
    # Exclude covs_only/covariates_logistic from feature sets shown (delta is trivially 0)
    fs_show = [fs for fs in feature_sets if fs != "covs_only"]
    if not fs_show:
        return

    n_fs = len(fs_show)
    n_m  = len(models)
    x       = np.arange(n_fs)
    width   = 0.8 / n_m
    offsets = np.linspace(-(0.8 - width) / 2, (0.8 - width) / 2, n_m)

    fig, ax = plt.subplots(figsize=(max(8, n_fs * 1.4), 5))

    for i, model in enumerate(models):
        sub = metrics_df[metrics_df["model"] == model].set_index("feature_set")
        deltas = [sub.loc[fs, "delta_auroc_vs_baseline"] if fs in sub.index else np.nan
                  for fs in fs_show]
        ax.bar(x + offsets[i], deltas, width,
               color=MODEL_COLORS.get(model, "#aaaaaa"), alpha=0.85,
               label=model.replace("_", " "), zorder=2)

    ax.axhline(0, color="#444444", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([fs.replace("_covs", "").replace("_", "\n") for fs in fs_show])
    ax.set_ylabel("Δ AUROC vs covariates baseline")
    ax.set_title("Pathway contribution beyond covariate baseline")
    ax.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left",
              title="Model", title_fontsize=plt.rcParams["legend.fontsize"])
    _save(fig, out_path)


# ── 6. Sparsity — pathway selection by L1 / elasticnet ───────────────────────

def plot_sparsity(sparsity_df, out_path):
    """
    Bar chart showing % of pathways selected (non-zero coefficient) per
    (feature_set, model) for l1_logistic and elasticnet.
    """
    if sparsity_df is None or sparsity_df.empty:
        print("  Skipping sparsity plot: no sparsity_summary.csv found")
        return

    models  = sparsity_df["model"].unique()
    fs_list = sorted(sparsity_df["feature_set"].unique())
    n_fs    = len(fs_list)

    fig, axes = plt.subplots(1, len(models), figsize=(max(6, n_fs * 1.0) * len(models), 4),
                              sharey=True, squeeze=False)

    for ax, model in zip(axes[0], models):
        sub = sparsity_df[sparsity_df["model"] == model].set_index("feature_set")
        pcts = [sub.loc[fs, "pct_pathways_selected"] if fs in sub.index else np.nan
                for fs in fs_list]
        colors = [MODEL_COLORS.get(model, "#aaaaaa")] * n_fs
        bars = ax.bar(range(n_fs), pcts, color=colors, alpha=0.8, edgecolor="white")

        for bar, pct in zip(bars, pcts):
            if not np.isnan(pct):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{pct:.1f}%", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(range(n_fs))
        ax.set_xticklabels([fs.replace("_covs", "").replace("_", "\n") for fs in fs_list])
        ax.set_ylabel("% pathways selected")
        ax.set_ylim(0, 105)
        ax.set_title(model.replace("_", " "))

    fig.suptitle("Pathway sparsity — % features with non-zero coefficient", y=1.02)
    _save(fig, out_path)


# ── 7. DeLong p-value heatmap ─────────────────────────────────────────────────

def plot_delong_heatmap(delong_df, feature_set, out_path):
    sub = delong_df[delong_df["feature_set"] == feature_set]
    if sub.empty:
        return

    models = sorted(set(sub["model_1"].tolist() + sub["model_2"].tolist()))
    n = len(models)
    mat = np.full((n, n), np.nan)
    np.fill_diagonal(mat, 1.0)

    for _, row in sub.iterrows():
        i, j = models.index(row["model_1"]), models.index(row["model_2"])
        mat[i, j] = mat[j, i] = row["p_value"]

    fig, ax = plt.subplots(figsize=(max(4, n), max(4, n)))
    # Show -log10(p); cap at 4 (p=0.0001)
    with np.errstate(divide="ignore"):
        log_mat = np.where(np.isnan(mat), np.nan, -np.log10(np.clip(mat, 1e-4, 1.0)))

    im = ax.imshow(log_mat, aspect="auto", cmap="Blues", vmin=0, vmax=4)

    labels = [m.replace("_", "\n") for m in models]
    ax.set_xticks(range(n)); ax.set_xticklabels(labels)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    ax.tick_params(length=0)

    for i in range(n):
        for j in range(n):
            if not np.isnan(mat[i, j]):
                txt = "—" if i == j else f"{mat[i,j]:.3f}"
                color = "white" if (not np.isnan(log_mat[i, j]) and log_mat[i, j] > 2) else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=6.5, color=color)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="−log₁₀(p)")
    ax.set_title(f"DeLong p-values — {feature_set.replace('_covs', '').replace('_', ' ')}")
    _save(fig, out_path)


# ── 8. ROC curves ─────────────────────────────────────────────────────────────

def plot_roc_curves(all_probs, all_labels, feature_sets, models, out_path):
    """One subplot per feature set; one ROC curve per model. Diagonal = chance."""
    fs_with_data = [fs for fs in feature_sets
                    if fs in all_labels and any((fs, m) in all_probs for m in models)]
    if not fs_with_data:
        print("  Skipping ROC curves: no probs/labels found")
        return

    ncols = min(3, len(fs_with_data))
    nrows = int(np.ceil(len(fs_with_data) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.2, nrows * 3.2),
                             squeeze=False)

    for ax in axes.flat:
        ax.set_visible(False)

    for idx, fs in enumerate(fs_with_data):
        ax = axes[idx // ncols][idx % ncols]
        ax.set_visible(True)
        labels = all_labels[fs]
        prevalence = labels.mean()

        for model in models:
            if (fs, model) not in all_probs:
                continue
            probs = all_probs[(fs, model)]
            fpr, tpr, _ = roc_curve(labels, probs)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=MODEL_COLORS.get(model, "#aaaaaa"),
                    linewidth=1.2, label=f"{model.replace('_', ' ')} ({roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "--", color="#aaaaaa", linewidth=0.7)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("1 − Specificity")
        ax.set_ylabel("Sensitivity")
        ax.set_title(fs.replace("_covs", "").replace("_", " "))
        ax.legend(fontsize=6, frameon=False, loc="lower right")
        ax.text(0.97, 0.06, f"prev={prevalence:.1%}", transform=ax.transAxes,
                fontsize=6, ha="right", color="#666666")
        ax.set_aspect("equal")

    _save(fig, out_path)


# ── 9. Precision-recall curves ────────────────────────────────────────────────

def plot_prc_curves(all_probs, all_labels, feature_sets, models, out_path):
    """One subplot per feature set; one PRC per model. Dashed = chance (prevalence)."""
    fs_with_data = [fs for fs in feature_sets
                    if fs in all_labels and any((fs, m) in all_probs for m in models)]
    if not fs_with_data:
        print("  Skipping PRC curves: no probs/labels found")
        return

    ncols = min(3, len(fs_with_data))
    nrows = int(np.ceil(len(fs_with_data) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.2, nrows * 3.2),
                             squeeze=False)

    for ax in axes.flat:
        ax.set_visible(False)

    for idx, fs in enumerate(fs_with_data):
        ax = axes[idx // ncols][idx % ncols]
        ax.set_visible(True)
        labels = all_labels[fs]
        prevalence = labels.mean()

        for model in models:
            if (fs, model) not in all_probs:
                continue
            probs = all_probs[(fs, model)]
            prec, rec, _ = precision_recall_curve(labels, probs)
            ap = average_precision_score(labels, probs)
            ax.plot(rec, prec, color=MODEL_COLORS.get(model, "#aaaaaa"),
                    linewidth=1.2, label=f"{model.replace('_', ' ')} (AP={ap:.3f})")

        ax.axhline(prevalence, color="#aaaaaa", linestyle="--", linewidth=0.7,
                   label=f"chance ({prevalence:.1%})")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(fs.replace("_covs", "").replace("_", " "))
        ax.legend(fontsize=6, frameon=False, loc="upper right")

    _save(fig, out_path)


# ── 10. Calibration reliability diagram ───────────────────────────────────────

def plot_calibration_diagram(all_probs, all_labels, feature_sets, models, out_path,
                             n_bins=10):
    """
    Reliability diagram: mean predicted probability vs observed event rate per decile.
    Perfect calibration = diagonal. One subplot per feature set.
    """
    fs_with_data = [fs for fs in feature_sets
                    if fs in all_labels and any((fs, m) in all_probs for m in models)]
    if not fs_with_data:
        print("  Skipping calibration diagram: no probs/labels found")
        return

    ncols = min(3, len(fs_with_data))
    nrows = int(np.ceil(len(fs_with_data) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.0, nrows * 3.0),
                             squeeze=False)

    for ax in axes.flat:
        ax.set_visible(False)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    for idx, fs in enumerate(fs_with_data):
        ax = axes[idx // ncols][idx % ncols]
        ax.set_visible(True)
        labels = all_labels[fs]

        for model in models:
            if (fs, model) not in all_probs:
                continue
            probs = all_probs[(fs, model)]
            obs_rates = []
            pred_means = []
            for lo, hi in zip(bins[:-1], bins[1:]):
                mask = (probs >= lo) & (probs < hi)
                if mask.sum() == 0:
                    continue
                obs_rates.append(labels[mask].mean())
                pred_means.append(probs[mask].mean())
            if not pred_means:
                continue
            ax.plot(pred_means, obs_rates, "o-",
                    color=MODEL_COLORS.get(model, "#aaaaaa"),
                    markersize=3, linewidth=1.0,
                    label=model.replace("_", " "))

        ax.plot([0, 1], [0, 1], "--", color="#aaaaaa", linewidth=0.7, label="perfect")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed event rate")
        ax.set_title(fs.replace("_covs", "").replace("_", " "))
        ax.legend(fontsize=6, frameon=False, loc="upper left")
        ax.set_aspect("equal")

    _save(fig, out_path)


# ── 11. Pathway lollipop ──────────────────────────────────────────────────────

def plot_pathway_lollipop(ranking_df, feature_set, models, top_n, out_path):
    """
    Horizontal lollipop: one subplot per model, rows = top pathways by percentile rank.
    Models share the same y-axis (pathway order determined by mean rank across models).
    """
    fs_cols = [(fs, m) for (fs, m) in ranking_df.columns
               if fs == feature_set and m in models]
    if not fs_cols:
        print(f"  Skipping lollipop for {feature_set}: no ranking data")
        return

    sub = ranking_df[fs_cols].copy()
    sub.columns = [m for (_, m) in fs_cols]
    sub = sub.dropna(how="all")

    # Order by mean rank across models; take top_n
    sub["_mean"] = sub.mean(axis=1)
    top = sub.nlargest(top_n, "_mean").drop(columns="_mean")
    top = top.sort_values(top.columns[0])  # ascending so highest is at top of plot

    models_present = [m for m in models if m in top.columns]
    n_m = len(models_present)
    if n_m == 0 or len(top) == 0:
        return

    def _wrap(name, width=40):
        return "\n".join(textwrap.wrap(name.replace("_", " "), width=width))

    n_f = len(top)
    # Width: label space + n_m subplots
    label_w = max(2.5, max(len(f) for f in top.index) * 0.055)
    fig, axes = plt.subplots(1, n_m,
                             figsize=(label_w + n_m * 2.8, max(4, n_f * 0.28 + 0.8)),
                             sharey=True)
    if n_m == 1:
        axes = [axes]

    y = np.arange(n_f)
    pathway_labels = [_wrap(f) for f in top.index]

    for ax, model in zip(axes, models_present):
        vals = top[model].values
        color = MODEL_COLORS.get(model, "#aaaaaa")
        ax.hlines(y, 0, vals, color="#dddddd", linewidth=0.8, zorder=1)
        ax.plot(vals, y, "o", color=color, markersize=4, zorder=2)
        ax.axvline(50, color="#aaaaaa", linestyle="--", linewidth=0.6)
        ax.set_xlim(0, 105)
        ax.set_xlabel("Percentile rank")
        ax.set_title(model.replace("_", " "))
        ax.tick_params(left=False)
        ax.spines["left"].set_visible(False)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(pathway_labels, fontsize=7)
    axes[0].invert_yaxis()

    fig.suptitle(
        f"{feature_set.replace('_covs', '').replace('_', ' ')} — top {top_n} pathways",
        y=1.01
    )
    _save(fig, out_path)


# ── Helper: load saved probs/labels ───────────────────────────────────────────

def _load_probs_labels(agg_dir):
    """Load probs/{fs}__{model}.npy and probs/labels__{fs}.npy saved by aggregate_results.py."""
    probs_dir = os.path.join(agg_dir, "probs")
    all_probs, all_labels = {}, {}
    if not os.path.isdir(probs_dir):
        return all_probs, all_labels
    for fname in sorted(os.listdir(probs_dir)):
        if not fname.endswith(".npy"):
            continue
        stem = fname[:-4]
        fpath = os.path.join(probs_dir, fname)
        if stem.startswith("labels__"):
            fs = stem[len("labels__"):]
            all_labels[fs] = np.load(fpath)
        else:
            # format: {fs}__{model} — fs and model names never contain "__"
            parts = stem.split("__", 1)
            if len(parts) == 2:
                all_probs[(parts[0], parts[1])] = np.load(fpath)
    return all_probs, all_labels


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--agg_dir", required=True,
                   help="Directory containing aggregate_results.py outputs.")
    p.add_argument("--out_dir", default="figures/")
    p.add_argument("--top_n", type=int, default=30)
    p.add_argument("--feature_sets", nargs="*", default=None,
                   help="Subset of feature sets to plot. Default: all in metrics_summary.csv.")
    return p.parse_args()


def main():
    args = parse_args()
    _style()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    metrics_path  = os.path.join(args.agg_dir, "metrics_summary.csv")
    rank_path     = os.path.join(args.agg_dir, "ranking_percentile.csv")
    delong_path   = os.path.join(args.agg_dir, "delong_summary.csv")
    sparsity_path = os.path.join(args.agg_dir, "sparsity_summary.csv")

    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(f"metrics_summary.csv not found in {args.agg_dir}")

    metrics_df = pd.read_csv(metrics_path)

    # GNN-specific summaries (written by aggregate_gnn_results.py)
    gnn_metrics_path  = os.path.join(args.agg_dir, "gnn_metrics_summary.csv")
    gnn_topology_path = os.path.join(args.agg_dir, "gnn_graph_topology.csv")
    gnn_metrics_df  = pd.read_csv(gnn_metrics_path)  if os.path.isfile(gnn_metrics_path)  else None
    gnn_topology_df = pd.read_csv(gnn_topology_path) if os.path.isfile(gnn_topology_path) else None

    ranking_df = None
    if os.path.isfile(rank_path):
        ranking_df = pd.read_csv(rank_path, index_col=0, header=[0, 1])

    delong_df = None
    if os.path.isfile(delong_path):
        delong_df = pd.read_csv(delong_path)

    sparsity_df = None
    if os.path.isfile(sparsity_path):
        sparsity_df = pd.read_csv(sparsity_path)

    # Load saved probs/labels (written by aggregate_results.py)
    all_probs, all_labels = _load_probs_labels(args.agg_dir)
    if all_probs:
        print(f"Loaded probs for {len(all_probs)} runs, {len(all_labels)} feature sets")
    else:
        print("No probs/labels found — ROC/PRC/calibration plots will be skipped "
              "(re-run aggregate_results.py to generate them)")

    # Normalize GNN rows: strip graph method suffix from feature_set, encode in model
    metrics_df = _normalize_gnn_rows(metrics_df)

    # Determine feature sets and models to plot
    feature_sets = args.feature_sets or _ordered_feature_sets(metrics_df["feature_set"].unique())
    models       = _model_order(metrics_df["model"].unique())

    print(f"Feature sets : {feature_sets}")
    print(f"Models       : {models}")

    # ── 1. AUROC grouped bar ─────────────────────────────────────────────────
    print("Plotting AUROC comparison...")
    plot_auroc_comparison(
        metrics_df, feature_sets,
        os.path.join(args.out_dir, "01_auroc_comparison.pdf"),
    )

    # ── 1b. GNN summary table ─────────────────────────────────────────────────
    if gnn_metrics_df is not None:
        print("Plotting GNN summary table...")
        plot_gnn_table(
            gnn_metrics_df, gnn_topology_df,
            os.path.join(args.out_dir, "01b_gnn_summary.pdf"),
        )

    # ── 2. AUROC heatmap ──────────────────────────────────────────────────────
    print("Plotting AUROC heatmap...")
    plot_auroc_heatmap(
        metrics_df, feature_sets,
        os.path.join(args.out_dir, "02_auroc_heatmap.pdf"),
    )

    # ── 2b. AUROC forest-plot style dot plot (main paper figure) ──────────────
    print("Plotting AUROC dot plot...")
    plot_auroc_dotplot(
        metrics_df, feature_sets,
        os.path.join(args.out_dir, "02b_auroc_dotplot.pdf"),
    )

    # ── 3. Multi-model dot plot (one per feature set) ─────────────────────────
    if ranking_df is not None:
        for fs in feature_sets:
            if fs == "covs_only":
                continue  # no pathway rankings
            tag = fs.replace("_covs", "").replace("_", "-")
            print(f"Plotting dot plot: {fs}...")
            plot_multimodel_dotplot(
                ranking_df, fs, models, args.top_n,
                os.path.join(args.out_dir, f"03_dotplot_{tag}.pdf"),
            )

    # ── 4. Cross-tissue heatmap (one per model) ───────────────────────────────
    if ranking_df is not None:
        for model in models:
            print(f"Plotting cross-tissue heatmap: {model}...")
            plot_cross_tissue_heatmap(
                ranking_df, model, args.top_n,
                os.path.join(args.out_dir, f"04_cross_tissue_{model}.pdf"),
            )

    # ── 5. Delta AUROC vs baseline ────────────────────────────────────────────
    print("Plotting delta AUROC...")
    plot_delta_auroc(
        metrics_df, feature_sets,
        os.path.join(args.out_dir, "05_delta_auroc.pdf"),
    )

    # ── 6. Sparsity ───────────────────────────────────────────────────────────
    print("Plotting sparsity...")
    plot_sparsity(
        sparsity_df,
        os.path.join(args.out_dir, "06_sparsity.pdf"),
    )

    # ── 7. DeLong heatmap (one per feature set) ───────────────────────────────
    if delong_df is not None:
        for fs in feature_sets:
            tag = fs.replace("_covs", "").replace("_", "-")
            print(f"Plotting DeLong heatmap: {fs}...")
            plot_delong_heatmap(
                delong_df, fs,
                os.path.join(args.out_dir, f"07_delong_{tag}.png"),
            )

    # ── 8. ROC curves ─────────────────────────────────────────────────────────
    if all_probs:
        print("Plotting ROC curves...")
        plot_roc_curves(
            all_probs, all_labels, feature_sets, models,
            os.path.join(args.out_dir, "08_roc_curves.png"),
        )

    # ── 9. Precision-recall curves ────────────────────────────────────────────
    if all_probs:
        print("Plotting PRC curves...")
        plot_prc_curves(
            all_probs, all_labels, feature_sets, models,
            os.path.join(args.out_dir, "09_prc_curves.png"),
        )

    # ── 10. Calibration reliability diagram ───────────────────────────────────
    if all_probs:
        print("Plotting calibration diagrams...")
        plot_calibration_diagram(
            all_probs, all_labels, feature_sets, models,
            os.path.join(args.out_dir, "10_calibration.png"),
        )

    # ── 11. Pathway lollipop (one per feature set) ────────────────────────────
    if ranking_df is not None:
        for fs in feature_sets:
            if fs == "covs_only":
                continue
            tag = fs.replace("_covs", "").replace("_", "-")
            print(f"Plotting lollipop: {fs}...")
            plot_pathway_lollipop(
                ranking_df, fs, models, args.top_n,
                os.path.join(args.out_dir, f"11_lollipop_{tag}.png"),
            )

    print(f"\nAll figures saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
