"""
Interpretability plots for pathway risk modeling.

Produces:
  1. Global attention weights — lollipop chart of top-K pathway importance
  2. Cross-model pathway ranking concordance — Spearman correlation heatmap
     + pairwise rank scatter plots
  3. Transformer attention head heatmaps — mean CLS->pathway attention per head/layer
  4. Transformer head specialization — entropy bar chart per head
  5. GNN GradCAM node attribution — lollipop chart + optional network graph
  6. Multi-model pathway rank comparison — top-N pathway dot plot across models

Usage:
  python scripts/plot_interpretability.py \
    --results_dir results/ \
    --output_dir figures/ \
    [--top_n 30] \
    [--pathway_names_file data/pathway_names.txt]

  results_dir must contain results.json (from train_all_models.py).
  Optionally load saved model weights to regenerate attention tensors on the fly.
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr


# ── Shared style ─────────────────────────────────────────────────────────────

PALETTE = {
    "global_attention": "#2166ac",
    "transformer":      "#4dac26",
    "gnn":              "#d01c8b",
    "random_forest":    "#f4a582",
    "l1_logistic":      "#b2abd2",
    "prs_logistic":     "#999999",
}

def _style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "figure.dpi": 150,
    })

def _save(fig, path, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 1. Global attention weight lollipop ──────────────────────────────────────

def plot_global_attention(ranking: list, top_n: int, out_path: str):
    """
    ranking: list of [pathway_name, weight] from results.json
    """
    df = pd.DataFrame(ranking, columns=["pathway", "weight"])
    df = df.sort_values("weight", ascending=False).head(top_n).reset_index(drop=True)
    df = df.sort_values("weight")  # ascending for horizontal lollipop

    fig, ax = plt.subplots(figsize=(6, max(4, top_n * 0.28)))
    y = np.arange(len(df))

    ax.hlines(y, 0, df["weight"], color=PALETTE["global_attention"], linewidth=1.2, alpha=0.7)
    ax.scatter(df["weight"], y, color=PALETTE["global_attention"], s=40, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(df["pathway"], fontsize=8)
    ax.set_xlabel("Softmax attention weight α")
    ax.set_title(f"Global pathway attention weights\n(top {top_n})", fontsize=10, fontweight="bold")
    ax.axvline(1 / len(ranking), color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
               label="Uniform (1/K)")
    ax.legend(fontsize=8, frameon=False)

    _save(fig, out_path)


# ── 2. Cross-model Spearman heatmap ──────────────────────────────────────────

def plot_spearman_heatmap(spearman_data: dict, out_path: str):
    """
    spearman_data: {"model1_vs_model2": {"rho": ..., "p_value": ...}, ...}
    """
    # Collect model names
    pairs = list(spearman_data.keys())
    models = []
    for pair in pairs:
        for m in pair.split("_vs_"):
            if m not in models:
                models.append(m)

    n = len(models)
    mat = np.full((n, n), np.nan)
    for i in range(n):
        mat[i, i] = 1.0
    for pair, res in spearman_data.items():
        m1, m2 = pair.split("_vs_")
        if m1 in models and m2 in models:
            i, j = models.index(m1), models.index(m2)
            mat[i, j] = mat[j, i] = res["rho"]

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r")

    labels = [m.replace("_", "\n") for m in models]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    for i in range(n):
        for j in range(n):
            if not np.isnan(mat[i, j]):
                color = "white" if abs(mat[i, j]) > 0.5 else "black"
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color=color)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman ρ")
    ax.set_title("Cross-model pathway ranking concordance", fontsize=10, fontweight="bold")
    _save(fig, out_path)


# ── 3. Pairwise rank scatter plots ───────────────────────────────────────────

def plot_rank_scatter(rankings: dict, top_n: int, out_path: str):
    """
    rankings: {model_name: [[pathway, score], ...]}
    Scatter plots of pathway ranks between each pair of models.
    """
    # Convert to rank dicts: pathway -> rank (1=top)
    def to_ranks(items):
        sorted_items = sorted(items, key=lambda x: -x[1])
        return {name: r + 1 for r, (name, _) in enumerate(sorted_items)}

    model_ranks = {m: to_ranks(items) for m, items in rankings.items()}
    models = list(model_ranks.keys())

    # Only plot models that have pathway rankings (baseline PRS logistic doesn't)
    n = len(models)
    if n < 2:
        return

    n_pairs = n * (n - 1) // 2
    ncols = min(3, n_pairs)
    nrows = (n_pairs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    pair_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            ax = axes_flat[pair_idx]
            m1, m2 = models[i], models[j]
            r1, r2 = model_ranks[m1], model_ranks[m2]
            common = sorted(set(r1) & set(r2))

            ranks1 = [r1[p] for p in common]
            ranks2 = [r2[p] for p in common]

            # Color top_n in either model
            top_set = {p for p in common if r1[p] <= top_n or r2[p] <= top_n}
            colors = [PALETTE.get(m1, "#aaaaaa") if p in top_set else "#cccccc" for p in common]

            ax.scatter(ranks1, ranks2, c=colors, s=18, alpha=0.7, linewidths=0)

            # Label top pathways
            for p in common:
                if r1[p] <= 10 or r2[p] <= 10:
                    ax.annotate(p, (r1[p], r2[p]), fontsize=5, alpha=0.8,
                                xytext=(3, 3), textcoords="offset points")

            rho, pval = spearmanr(ranks1, ranks2)
            ax.set_xlabel(f"{m1.replace('_', ' ')} rank", fontsize=8)
            ax.set_ylabel(f"{m2.replace('_', ' ')} rank", fontsize=8)
            ax.set_title(f"ρ={rho:.2f}, p={pval:.3g}", fontsize=8)
            ax.invert_xaxis()
            ax.invert_yaxis()
            pair_idx += 1

    for ax in axes_flat[pair_idx:]:
        ax.set_visible(False)

    fig.suptitle("Pairwise pathway rank comparison", fontsize=11, fontweight="bold", y=1.01)
    _save(fig, out_path)


# ── 4. Transformer attention head heatmap ────────────────────────────────────

def plot_transformer_head_heatmap(
    mean_attn_npy_path: str,
    pathway_names: list,
    top_n: int,
    out_path: str,
):
    """
    mean_attn_npy_path: .npy file of shape (n_layers, n_heads, K+1, K+1)
                        saved from PathwayTransformer.compute_mean_attention()
    Shows CLS -> pathway attention (index 0 -> indices 1:) per head per layer.
    """
    if not os.path.exists(mean_attn_npy_path):
        print(f"  Skipping transformer heatmap: {mean_attn_npy_path} not found")
        return

    attn = np.load(mean_attn_npy_path)  # (L, H, K+1, K+1)
    n_layers, n_heads = attn.shape[:2]

    fig, axes = plt.subplots(
        n_layers, n_heads,
        figsize=(4 * n_heads, 3.5 * n_layers),
        squeeze=False,
    )

    for layer in range(n_layers):
        for head in range(n_heads):
            ax = axes[layer][head]
            cls_attn = attn[layer, head, 0, 1:]  # (K,)

            # Show top_n pathways by this head's CLS attention
            top_idx = np.argsort(-cls_attn)[:top_n]
            top_names = [pathway_names[i] for i in top_idx]
            top_vals = cls_attn[top_idx]

            im = ax.imshow(top_vals[np.newaxis, :], aspect="auto",
                           cmap="Blues", vmin=0)
            ax.set_xticks(range(top_n))
            ax.set_xticklabels(top_names, rotation=90, fontsize=6)
            ax.set_yticks([])
            ax.set_title(f"Layer {layer+1}, Head {head+1}", fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Transformer: mean CLS→pathway attention weights", fontsize=11, fontweight="bold")
    _save(fig, out_path)


# ── 5. Transformer head specialization entropy ───────────────────────────────

def plot_head_entropy(
    mean_attn_npy_path: str,
    out_path: str,
):
    """
    Low entropy = head attends to a small set of pathways (specialized).
    High entropy = head attends broadly (uniform).
    """
    if not os.path.exists(mean_attn_npy_path):
        print(f"  Skipping head entropy: {mean_attn_npy_path} not found")
        return

    attn = np.load(mean_attn_npy_path)  # (L, H, K+1, K+1)
    n_layers, n_heads = attn.shape[:2]

    entropies = np.zeros((n_layers, n_heads))
    for l in range(n_layers):
        for h in range(n_heads):
            p = attn[l, h, 0, 1:]
            p = p / (p.sum() + 1e-9)
            entropies[l, h] = -np.sum(p * np.log(p + 1e-9))

    uniform_entropy = np.log(attn.shape[2] - 1)  # log(K)

    fig, ax = plt.subplots(figsize=(max(4, n_heads * 1.2), 3.5))
    x = np.arange(n_heads)
    width = 0.35

    colors = [PALETTE["global_attention"], PALETTE["transformer"]]
    for l in range(n_layers):
        offset = (l - (n_layers - 1) / 2) * width
        bars = ax.bar(x + offset, entropies[l], width, label=f"Layer {l+1}",
                      color=colors[l % len(colors)], alpha=0.8)

    ax.axhline(uniform_entropy, color="gray", linestyle="--", linewidth=0.9,
               label="Uniform entropy")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Head {h+1}" for h in range(n_heads)])
    ax.set_ylabel("Attention entropy (nats)")
    ax.set_title("Transformer head specialization\n(lower = more focused)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, frameon=False)
    _save(fig, out_path)


# ── 6. GNN GradCAM lollipop ──────────────────────────────────────────────────

def plot_gradcam(ranking: list, top_n: int, out_path: str):
    df = pd.DataFrame(ranking, columns=["pathway", "gradcam"])
    df = df.sort_values("gradcam", ascending=False).head(top_n).reset_index(drop=True)
    df = df.sort_values("gradcam")

    fig, ax = plt.subplots(figsize=(6, max(4, top_n * 0.28)))
    y = np.arange(len(df))

    norm = Normalize(vmin=df["gradcam"].min(), vmax=df["gradcam"].max())
    cmap = plt.get_cmap("YlOrRd")
    colors = [cmap(norm(v)) for v in df["gradcam"]]

    ax.hlines(y, 0, df["gradcam"], color="#cccccc", linewidth=1.0)
    for yi, (val, col) in enumerate(zip(df["gradcam"], colors)):
        ax.scatter(val, yi, color=col, s=50, zorder=3)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, label="GradCAM score")

    ax.set_yticks(y)
    ax.set_yticklabels(df["pathway"], fontsize=8)
    ax.set_xlabel("Mean GradCAM node attribution")
    ax.set_title(f"GNN pathway importance (GradCAM)\n(top {top_n})", fontsize=10, fontweight="bold")
    _save(fig, out_path)


# ── 7. Multi-model top-N dot plot ────────────────────────────────────────────

def plot_multimodel_dotplot(rankings: dict, top_n: int, out_path: str):
    """
    Show pathways that appear in the top_n of any model.
    X-axis: model, Y-axis: pathway. Dot size/color = rank.
    """
    # Collect union of top-N pathways across all models
    top_pathways = set()
    model_rank_dicts = {}
    for model, items in rankings.items():
        sorted_items = sorted(items, key=lambda x: -x[1])
        ranks = {name: r + 1 for r, (name, _) in enumerate(sorted_items)}
        model_rank_dicts[model] = ranks
        for name, _ in sorted_items[:top_n]:
            top_pathways.add(name)

    # Order pathways by mean rank across models (ascending = best)
    pathway_list = sorted(
        top_pathways,
        key=lambda p: np.mean([model_rank_dicts[m].get(p, len(top_pathways) + 1)
                               for m in model_rank_dicts]),
    )
    models = list(rankings.keys())

    max_rank = max(
        max(d.values()) for d in model_rank_dicts.values() if d
    )

    fig, ax = plt.subplots(
        figsize=(max(5, len(models) * 1.4), max(5, len(pathway_list) * 0.35))
    )

    for xi, model in enumerate(models):
        ranks = model_rank_dicts[model]
        for yi, pathway in enumerate(pathway_list):
            r = ranks.get(pathway, None)
            if r is None:
                continue
            # Dot size: larger = higher rank (lower number)
            size = max(10, 300 * (1 - (r - 1) / max_rank))
            color = PALETTE.get(model, "#888888")
            alpha = max(0.2, 1 - (r - 1) / max_rank)
            ax.scatter(xi, yi, s=size, color=color, alpha=alpha, linewidths=0)
            if r <= top_n:
                ax.text(xi, yi, str(r), ha="center", va="center", fontsize=5.5,
                        color="white" if alpha > 0.5 else "black")

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace("_", "\n") for m in models], fontsize=8)
    ax.set_yticks(range(len(pathway_list)))
    ax.set_yticklabels(pathway_list, fontsize=7.5)
    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.set_ylim(-0.6, len(pathway_list) - 0.4)
    ax.set_title(
        f"Top-{top_n} pathways across models\n(number = rank, dot size ∝ importance)",
        fontsize=10, fontweight="bold",
    )

    handles = [mpatches.Patch(color=PALETTE.get(m, "#888888"), label=m.replace("_", " "))
               for m in models]
    ax.legend(handles=handles, fontsize=7, frameon=False,
              bbox_to_anchor=(1.01, 1), loc="upper left")

    _save(fig, out_path)


# ── 8. L1 logistic regression coefficient plot ───────────────────────────────

def plot_l1_coefficients(ranking: list, top_n: int, out_path: str):
    """
    Shows L1 logistic regression pathway coefficient magnitudes.
    Distinguishes zero (pruned) vs non-zero (selected) coefficients.
    """
    df = pd.DataFrame(ranking, columns=["pathway", "coef_magnitude"])
    df_nonzero = df[df["coef_magnitude"] > 0].sort_values("coef_magnitude", ascending=False)
    n_zero = (df["coef_magnitude"] == 0).sum()

    df_plot = df_nonzero.head(top_n).sort_values("coef_magnitude")

    fig, ax = plt.subplots(figsize=(6, max(4, len(df_plot) * 0.28)))
    y = np.arange(len(df_plot))

    ax.hlines(y, 0, df_plot["coef_magnitude"], color=PALETTE["l1_logistic"], linewidth=1.2, alpha=0.7)
    ax.scatter(df_plot["coef_magnitude"], y, color=PALETTE["l1_logistic"], s=40, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(df_plot["pathway"], fontsize=8)
    ax.set_xlabel("Coefficient magnitude (L2 norm per pathway)")
    ax.set_title(
        f"L1 logistic: selected pathways (top {top_n})\n"
        f"{len(df_nonzero)} non-zero / {len(df)} total ({n_zero} pruned to zero)",
        fontsize=10, fontweight="bold",
    )
    _save(fig, out_path)


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/")
    p.add_argument("--output_dir", default="figures/")
    p.add_argument("--top_n", type=int, default=30,
                   help="Number of top pathways to show in lollipop/dot plots")
    p.add_argument("--pathway_names_file", default=None,
                   help="Optional plain text file with one pathway name per line "
                        "(needed for transformer heatmap if not in results.json)")
    p.add_argument("--mean_attn_npy", default=None,
                   help="Path to .npy file of mean transformer attention weights "
                        "(shape: n_layers, n_heads, K+1, K+1). "
                        "Default: results_dir/mean_attention.npy")
    return p.parse_args()


def main():
    args = parse_args()
    _style()
    os.makedirs(args.output_dir, exist_ok=True)

    results_path = os.path.join(args.results_dir, "results.json")
    with open(results_path) as f:
        results = json.load(f)

    rankings = results.get("pathway_rankings", {})
    spearman = results.get("spearman_concordance", {})

    # ── 1. Global attention lollipop
    if "global_attention" in rankings:
        print("Plotting global attention weights...")
        plot_global_attention(
            rankings["global_attention"], args.top_n,
            os.path.join(args.output_dir, "01_global_attention_weights.pdf"),
        )

    # ── 2. Spearman concordance heatmap
    if spearman:
        print("Plotting Spearman concordance heatmap...")
        plot_spearman_heatmap(
            spearman,
            os.path.join(args.output_dir, "02_spearman_concordance_heatmap.pdf"),
        )

    # ── 3. Pairwise rank scatter
    if len(rankings) >= 2:
        print("Plotting pairwise rank scatter...")
        plot_rank_scatter(
            rankings, args.top_n,
            os.path.join(args.output_dir, "03_rank_scatter.pdf"),
        )

    # ── 4 & 5. Transformer attention heatmap + head entropy
    attn_npy = args.mean_attn_npy or os.path.join(args.results_dir, "mean_attention.npy")
    pathway_names = None
    if args.pathway_names_file and os.path.exists(args.pathway_names_file):
        with open(args.pathway_names_file) as f:
            pathway_names = [l.strip() for l in f if l.strip()]
    elif "global_attention" in rankings:
        pathway_names = [item[0] for item in rankings["global_attention"]]

    if pathway_names:
        print("Plotting transformer attention heatmaps...")
        plot_transformer_head_heatmap(
            attn_npy, pathway_names, min(args.top_n, 20),
            os.path.join(args.output_dir, "04_transformer_head_heatmap.pdf"),
        )
        print("Plotting head entropy...")
        plot_head_entropy(
            attn_npy,
            os.path.join(args.output_dir, "05_transformer_head_entropy.pdf"),
        )

    # ── 6. GNN GradCAM lollipop
    if "gnn" in rankings:
        print("Plotting GNN GradCAM attribution...")
        plot_gradcam(
            rankings["gnn"], args.top_n,
            os.path.join(args.output_dir, "06_gnn_gradcam.pdf"),
        )

    # ── 7. Multi-model dot plot
    if len(rankings) >= 2:
        print("Plotting multi-model dot plot...")
        plot_multimodel_dotplot(
            rankings, args.top_n,
            os.path.join(args.output_dir, "07_multimodel_dotplot.pdf"),
        )

    # ── 8. L1 coefficient plot
    if "l1_logistic" in rankings:
        print("Plotting L1 logistic coefficients...")
        plot_l1_coefficients(
            rankings["l1_logistic"], args.top_n,
            os.path.join(args.output_dir, "08_l1_coefficients.pdf"),
        )

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
