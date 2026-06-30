"""
Pathway GNN Model.

Architecture:
  - Pathway graph: nodes = pathways, edges from STRING PPI (>=2 shared genes, conf>=400)
    or Reactome hierarchy (parent-child relationships)
  - Two GraphSAGE layers with residual connections
  - Global mean + max pooling -> graph-level representation
  - Covariate fusion -> linear output head
  - Interpretability: GradCAM-based node attribution

Requires: torch_geometric
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool
    from torch_geometric.data import Data, Batch
    HAS_PYGEOMETRIC = True
except ImportError:
    HAS_PYGEOMETRIC = False


def _check_pygeometric():
    if not HAS_PYGEOMETRIC:
        raise ImportError(
            "torch_geometric is required for PathwayGNN. "
            "Install via: pip install torch_geometric"
        )


class GraphSAGEWithResidual(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        _check_pygeometric()
        self.conv = SAGEConv(in_channels, out_channels, aggr="mean")
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x, edge_index):
        out = self.conv(x, edge_index)
        out = self.norm(out)
        out = F.relu(out)
        out = self.dropout(out)
        return out + self.residual(x)


class PathwayGNN(nn.Module):
    """
    Args:
        n_pathways: K (number of pathway nodes)
        pathway_input_dim: T (features per pathway node)
        covariate_dim: C
        embed_dim: d
        n_sage_layers: number of GraphSAGE layers (default 2)
        dropout: dropout rate
        edge_index: (2, E) LongTensor of graph edges; if None, must be passed at forward
    """

    def __init__(
        self,
        n_pathways: int,
        pathway_input_dim: int,
        covariate_dim: int,
        embed_dim: int = 64,
        n_sage_layers: int = 2,
        dropout: float = 0.1,
        edge_index: torch.Tensor = None,
        fully_connected: bool = False,
    ):
        if n_pathways == 0:
            raise ValueError("PathwayGNN requires at least one pathway (n_pathways > 0). "
                             "For covariates-only runs use global_attention or transformer.")
        _check_pygeometric()
        super().__init__()
        self.n_pathways = n_pathways
        self.embed_dim = embed_dim
        # fully_connected: skip message passing, use global pooling only (ablation)
        # Equivalent to fully_connected GNN with mean aggregation but avoids
        # materialising K*(K-1) edges per sample
        self.fully_connected = fully_connected

        if not fully_connected:
            if edge_index is not None:
                self.register_buffer("edge_index", edge_index)
            else:
                self.edge_index = None

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(pathway_input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # GraphSAGE layers
        sage_layers = []
        for _ in range(n_sage_layers):
            sage_layers.append(GraphSAGEWithResidual(embed_dim, embed_dim, dropout))
        self.sage_layers = nn.ModuleList(sage_layers)

        # Covariate encoder
        self.covariate_encoder = nn.Sequential(
            nn.Linear(covariate_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # After global mean + max pooling we get 2*embed_dim from the graph
        self.output_head = nn.Linear(3 * embed_dim, 1)  # 2*embed_dim (pool) + embed_dim (cov)

    def _build_batch_graph(self, pathway_features: torch.Tensor, edge_index: torch.Tensor):
        """
        Convert batch of node feature matrices to a PyG Batch object.
        pathway_features: (batch, K, T)
        Returns PyG Batch with .x (batch*K, T) and .batch (batch*K,)
        """
        batch_size, K, T = pathway_features.shape
        device = pathway_features.device

        # Stack node features
        x = pathway_features.reshape(batch_size * K, T)

        # Build batch-offset edge indices
        offset = torch.arange(batch_size, device=device) * K  # (batch,)
        src = edge_index[0].unsqueeze(0) + offset.unsqueeze(1)  # (batch, E)
        dst = edge_index[1].unsqueeze(0) + offset.unsqueeze(1)
        src = src.reshape(-1)
        dst = dst.reshape(-1)
        batch_edge_index = torch.stack([src, dst], dim=0)

        batch_vec = torch.arange(batch_size, device=device).repeat_interleave(K)

        return x, batch_edge_index, batch_vec

    def forward(self, pathway_features: torch.Tensor, covariates: torch.Tensor, edge_index: torch.Tensor = None):
        """
        pathway_features: (batch, K, T)
        covariates:       (batch, C)
        edge_index:       (2, E) — uses stored edge_index if None; ignored when fully_connected=True
        Returns logits (batch,)
        """
        batch_size, K, T = pathway_features.shape
        device = pathway_features.device

        x_flat = pathway_features.reshape(batch_size * K, T)
        x = self.input_proj(x_flat)  # (batch*K, embed_dim)

        if self.fully_connected:
            # Ablation: all pathways connected — equivalent to global pooling
            # after one mean-aggregation SAGE step, without materialising K*(K-1) edges
            x = x.reshape(batch_size, K, self.embed_dim)
            x_mean = x.mean(dim=1)           # (batch, embed_dim)
            x_max  = x.max(dim=1).values     # (batch, embed_dim)
        else:
            if edge_index is None:
                edge_index = self.edge_index
            if edge_index is None:
                raise ValueError("edge_index must be provided either at init or at forward()")

            x_flat, batch_edge_index, batch_vec = self._build_batch_graph(pathway_features, edge_index)
            x = self.input_proj(x_flat)

            for layer in self.sage_layers:
                x = layer(x, batch_edge_index)

            batch_vec = torch.arange(batch_size, device=device).repeat_interleave(K)
            x_mean = global_mean_pool(x, batch_vec)
            x_max  = global_max_pool(x, batch_vec)

        graph_repr = torch.cat([x_mean, x_max], dim=-1)
        cov_embed  = self.covariate_encoder(covariates)
        combined   = torch.cat([graph_repr, cov_embed], dim=-1)
        return self.output_head(combined).squeeze(-1)

    def get_node_gradcam(
        self,
        pathway_features: torch.Tensor,
        covariates: torch.Tensor,
        edge_index: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        GradCAM node attribution.
        Returns node importance scores of shape (batch, K).
        """
        if edge_index is None:
            edge_index = self.edge_index

        pathway_features = pathway_features.requires_grad_(True)
        x_flat, batch_edge_index, batch_vec = self._build_batch_graph(pathway_features, edge_index)
        x = self.input_proj(x_flat)

        # Track activations after the last SAGE layer
        for layer in self.sage_layers:
            x = layer(x, batch_edge_index)

        activations = x  # (batch*K, embed_dim)
        activations.retain_grad()

        x_mean = global_mean_pool(activations, batch_vec)
        x_max = global_max_pool(activations, batch_vec)
        graph_repr = torch.cat([x_mean, x_max], dim=-1)
        cov_embed = self.covariate_encoder(covariates)
        combined = torch.cat([graph_repr, cov_embed], dim=-1)
        logits = self.output_head(combined).squeeze(-1)

        logits.sum().backward()

        grads = activations.grad  # (batch*K, embed_dim)
        # GradCAM: relu(mean over channels of grad * activation)
        scores = F.relu((grads * activations).mean(dim=-1))  # (batch*K,)
        batch_size = pathway_features.size(0)
        K = self.n_pathways
        return scores.reshape(batch_size, K).detach()


def build_string_edge_index(
    pathway_gene_sets: dict,
    ppi_edges: list,
    min_shared_genes: int = 2,
    pathway_names: list = None,
) -> torch.Tensor:
    """
    Build edge_index from STRING PPI data.

    pathway_gene_sets: {pathway_name: set_of_gene_ids}
    ppi_edges: list of (gene1, gene2, confidence_score) tuples
    min_shared_genes: minimum shared PPI-connected gene pairs for an edge
    pathway_names: ordered list of pathway names (determines node indices)

    Returns edge_index (2, E) LongTensor (undirected, self-loops excluded).
    """
    if pathway_names is None:
        pathway_names = list(pathway_gene_sets.keys())
    name_to_idx = {n: i for i, n in enumerate(pathway_names)}
    K = len(pathway_names)

    # Build gene -> pathway index
    gene_to_pathways = {}
    for name, genes in pathway_gene_sets.items():
        if name not in name_to_idx:
            continue
        idx = name_to_idx[name]
        for g in genes:
            gene_to_pathways.setdefault(g, set()).add(idx)

    # PPI gene pairs (CSV is pre-filtered to desired confidence threshold)
    ppi_gene_pairs = set()
    for g1, g2, conf in ppi_edges:
        ppi_gene_pairs.add((min(g1, g2), max(g1, g2)))

    # Count shared PPI-linked genes between pathway pairs
    shared_counts = {}
    for g1, g2 in ppi_gene_pairs:
        paths1 = gene_to_pathways.get(g1, set())
        paths2 = gene_to_pathways.get(g2, set())
        for p1 in paths1:
            for p2 in paths2:
                if p1 != p2:
                    key = (min(p1, p2), max(p1, p2))
                    shared_counts[key] = shared_counts.get(key, 0) + 1

    src, dst = [], []
    for (p1, p2), count in shared_counts.items():
        if count >= min_shared_genes:
            src.extend([p1, p2])
            dst.extend([p2, p1])

    if not src:
        return torch.zeros(2, 0, dtype=torch.long)

    return torch.tensor([src, dst], dtype=torch.long)


def build_jaccard_edge_index(
    pathway_gene_sets: dict,
    pathway_names: list,
    min_overlap: int = 3,
    min_jaccard: float = 0.1,
) -> torch.Tensor:
    """
    Build edge_index from gene set Jaccard overlap.

    Two pathways are connected if they share >= min_overlap genes AND their
    Jaccard similarity >= min_jaccard.  Uses the same gene sets that defined
    the pathway scores — no external database required.

    pathway_gene_sets: {pathway_name: set_of_gene_symbols}
    pathway_names: ordered list of pathway names (determines node indices)
    """
    name_to_idx = {n: i for i, n in enumerate(pathway_names)}
    sets = {name_to_idx[n]: set(g) for n, g in pathway_gene_sets.items() if n in name_to_idx}
    K = len(pathway_names)

    src, dst = [], []
    idxs = sorted(sets)
    for i, p1 in enumerate(idxs):
        for p2 in idxs[i + 1:]:
            inter = len(sets[p1] & sets[p2])
            if inter < min_overlap:
                continue
            union = len(sets[p1] | sets[p2])
            if union == 0:
                continue
            if inter / union >= min_jaccard:
                src.extend([p1, p2])
                dst.extend([p2, p1])

    if not src:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)


def build_score_correlation_edge_index(
    pw_train: "np.ndarray",
    pathway_names: list,
    threshold: float = 0.3,
) -> torch.Tensor:
    """
    Build edge_index from pairwise Pearson correlation of pathway scores
    across training individuals.  Computed on the training set only to
    avoid leakage.

    Two pathways are connected if |correlation| >= threshold.

    pw_train: (N_train, K) or (N_train, K, T) array of pathway scores
    pathway_names: ordered list of K pathway names
    threshold: absolute correlation cutoff for an edge
    """
    import numpy as np
    if pw_train.ndim == 3:
        # Average across T features per pathway
        pw_train = pw_train.mean(axis=2)

    # (N, K) → correlation matrix (K, K)
    # np.corrcoef expects (K, N)
    corr = np.corrcoef(pw_train.T)
    K = len(pathway_names)

    src, dst = [], []
    for i in range(K):
        for j in range(i + 1, K):
            if abs(corr[i, j]) >= threshold:
                src.extend([i, j])
                dst.extend([j, i])

    if not src:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)


def build_fully_connected_edge_index(n_pathways: int) -> torch.Tensor:
    """
    Fully connected graph (ablation baseline).
    Tests whether graph topology matters at all vs. pure node aggregation.
    """
    idx = torch.arange(n_pathways)
    src = idx.repeat_interleave(n_pathways)
    dst = idx.repeat(n_pathways)
    mask = src != dst
    return torch.stack([src[mask], dst[mask]], dim=0)


