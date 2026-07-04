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
        "font.family": "sans-serif",
        "font.size":   9,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "figure.dpi": 150,
    })


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


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
    width   = 0.8 / n_m
    offsets = np.linspace(-(0.8 - width) / 2, (0.8 - width) / 2, n_m)

    fig, ax = plt.subplots(figsize=(max(8, n_fs * 1.4), 5))

    for i, model in enumerate(models):
        sub = metrics_df[metrics_df["model"] == model].set_index("feature_set")
        aurocs     = [sub.loc[fs, "auroc"]         if fs in sub.index else np.nan for fs in feature_sets]
        ci_lower   = [sub.loc[fs, "auroc_ci_lower"] if fs in sub.index else np.nan for fs in feature_sets]
        ci_upper   = [sub.loc[fs, "auroc_ci_upper"] if fs in sub.index else np.nan for fs in feature_sets]
        yerr_lo    = np.array([a - l if not np.isnan(a) else 0 for a, l in zip(aurocs, ci_lower)])
        yerr_hi    = np.array([u - a if not np.isnan(a) else 0 for a, u in zip(aurocs, ci_upper)])

        ax.bar(x + offsets[i], aurocs, width,
               color=MODEL_COLORS.get(model, "#aaaaaa"), alpha=0.85,
               label=model.replace("_", " "), zorder=2)
        ax.errorbar(x + offsets[i], aurocs,
                    yerr=[yerr_lo, yerr_hi],
                    fmt="none", color="black", linewidth=0.8, capsize=2, zorder=3)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([fs.replace("_covs", "").replace("_", "\n") for fs in feature_sets],
                       fontsize=8)
    ax.set_ylabel("AUROC")
    ax.set_ylim(bottom=max(0, metrics_df["auroc"].min() - 0.05))
    ax.set_title("AUROC by feature set and model (95% CI)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left",
              title="Model", title_fontsize=8)
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
                color = "white" if bg < 0.45 else "black"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7,
                        color=color, fontweight="bold")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace("_", "\n") for m in models], fontsize=8)
    ax.set_yticks(range(len(feature_sets)))
    ax.set_yticklabels([fs.replace("_covs", "").replace("_", " ") for fs in feature_sets], fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Model", fontsize=9)
    ax.set_ylabel("Feature set", fontsize=9)
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
        tick_step = 0.01
        ticks = np.arange(np.ceil(vmin / tick_step) * tick_step,
                          vmax + tick_step / 2, tick_step)
        ticks = np.round(ticks[ticks <= vmax + 1e-9], 3)
        cbar.set_ticks(ticks)

    _save(fig, out_path)


def plot_auroc_dotplot(metrics_df, feature_sets, out_path):
    """
    Forest-plot style dot plot: one row per feature set, one dot per model,
    x-axis fixed at 0.5–1.0 with 95% CI whiskers. Honest scale, CI visible.
    """
    models = _model_order(metrics_df["model"].unique())
    n_fs = len(feature_sets)
    n_m  = len(models)

    fig, ax = plt.subplots(figsize=(6, max(4, n_fs * n_m * 0.25 + 1)))

    y_ticks, y_labels = [], []
    row = 0
    for fs in feature_sets:
        sub = metrics_df[metrics_df["feature_set"] == fs].set_index("model")
        group_rows = []
        for model in models:
            if model not in sub.index or np.isnan(sub.loc[model, "auroc"]):
                row += 1
                continue
            r = sub.loc[model]
            auroc, lo, hi = r["auroc"], r["auroc_ci_lower"], r["auroc_ci_upper"]
            color = MODEL_COLORS.get(model, "#aaaaaa")
            ax.plot(auroc, row, "o", color=color, markersize=5, zorder=3)
            if not np.isnan(lo) and not np.isnan(hi):
                ax.plot([lo, hi], [row, row], "-", color=color, linewidth=1.5, zorder=2)
            group_rows.append(row)
            row += 1
        if group_rows:
            mid = (group_rows[0] + group_rows[-1]) / 2
            y_ticks.append(mid)
            y_labels.append(fs.replace("_covs", "").replace("_", " "))
            # Separator between feature sets
            ax.axhline(row - 0.5, color="#dddddd", linewidth=0.8)
        row += 0.5  # gap between feature sets

    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="random")
    ax.set_xlim(0.5, min(1.0, metrics_df["auroc"].max() + 0.04))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("AUROC (95% CI)", fontsize=9)
    ax.set_title("Model performance by feature set", fontsize=11, fontweight="bold")

    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=MODEL_COLORS.get(m, "#aaa"),
                          markersize=6, label=m.replace("_", " ")) for m in models]
    ax.legend(handles=handles, fontsize=7, frameon=False,
              bbox_to_anchor=(1.01, 1), loc="upper left", title="Model", title_fontsize=8)
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
    ax.set_xticklabels([m.replace("_", "\n") for m in models_present], fontsize=8)
    ax.set_yticks(range(n_f))
    ax.set_yticklabels([f.replace("_", " ") for f in top_features.index], fontsize=7)
    ax.set_xlim(-0.6, n_m - 0.4)
    ax.set_ylim(-0.6, n_f - 0.4)
    ax.set_title(
        f"{feature_set.replace('_covs', '')} — top {top_n} features\n"
        f"(dot size/number = within-model percentile rank)",
        fontsize=10, fontweight="bold",
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
    ax.set_xticklabels(top.columns, fontsize=9)
    ax.set_yticks(range(n_pathways))

    def _wrap(name, width=45):
        return "\n".join(textwrap.wrap(name.replace("_", " "), width=width))

    ax.set_yticklabels([_wrap(f) for f in top.index], fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Percentile rank")
    ax.set_title(
        f"{model.replace('_', ' ')} — pathway ranks across GREx tissues\n(top {top_n} by mean rank)",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 1.0])
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


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

    ax.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([fs.replace("_covs", "").replace("_", "\n") for fs in fs_show], fontsize=8)
    ax.set_ylabel("Δ AUROC vs covariates baseline")
    ax.set_title("Pathway contribution beyond covariate baseline", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left",
              title="Model", title_fontsize=8)
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
        ax.set_xticklabels([fs.replace("_covs", "").replace("_", "\n") for fs in fs_list],
                           fontsize=7)
        ax.set_ylabel("% pathways selected")
        ax.set_ylim(0, 105)
        ax.set_title(model.replace("_", " "), fontsize=9, fontweight="bold")

    fig.suptitle("Pathway sparsity — % features with non-zero coefficient",
                 fontsize=11, fontweight="bold")
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
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=7)

    for i in range(n):
        for j in range(n):
            if not np.isnan(mat[i, j]):
                txt = "1" if i == j else f"{mat[i,j]:.3f}"
                color = "white" if (not np.isnan(log_mat[i, j]) and log_mat[i, j] > 2) else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=6.5, color=color)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="-log10(p)")
    ax.set_title(
        f"DeLong pairwise p-values\n{feature_set.replace('_covs', '')}",
        fontsize=10, fontweight="bold",
    )
    _save(fig, out_path)


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

    ranking_df = None
    if os.path.isfile(rank_path):
        ranking_df = pd.read_csv(rank_path, index_col=0, header=[0, 1])

    delong_df = None
    if os.path.isfile(delong_path):
        delong_df = pd.read_csv(delong_path)

    sparsity_df = None
    if os.path.isfile(sparsity_path):
        sparsity_df = pd.read_csv(sparsity_path)

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
                os.path.join(args.out_dir, f"07_delong_{tag}.pdf"),
            )

    print(f"\nAll figures saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
