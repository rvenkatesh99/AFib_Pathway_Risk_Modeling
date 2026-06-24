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
    ):
        _check_pygeometric()
        super().__init__()
        self.n_pathways = n_pathways
        self.embed_dim = embed_dim

        # Register graph structure as a buffer so it moves with the model
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
        edge_index:       (2, E) — uses stored edge_index if None
        Returns logits (batch,)
        """
        if edge_index is None:
            edge_index = self.edge_index
        if edge_index is None:
            raise ValueError("edge_index must be provided either at init or at forward()")

        x_flat, batch_edge_index, batch_vec = self._build_batch_graph(pathway_features, edge_index)

        # Input projection
        x = self.input_proj(x_flat)  # (batch*K, embed_dim)

        # GraphSAGE message passing
        for layer in self.sage_layers:
            x = layer(x, batch_edge_index)

        # Global pooling: mean + max -> (batch, 2*embed_dim)
        x_mean = global_mean_pool(x, batch_vec)
        x_max = global_max_pool(x, batch_vec)
        graph_repr = torch.cat([x_mean, x_max], dim=-1)

        # Covariate fusion
        cov_embed = self.covariate_encoder(covariates)
        combined = torch.cat([graph_repr, cov_embed], dim=-1)
        logits = self.output_head(combined).squeeze(-1)
        return logits

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

    # High-confidence PPI gene pairs
    ppi_gene_pairs = set()
    for g1, g2, conf in ppi_edges:
        if conf >= 400:
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


def build_reactome_edge_index(
    hierarchy_edges: list,
    pathway_names: list,
) -> torch.Tensor:
    """
    Build edge_index from Reactome parent-child hierarchy.

    hierarchy_edges: list of (parent_name, child_name) tuples
    pathway_names: ordered list of pathway names

    Returns edge_index (2, E) LongTensor (bidirectional).
    """
    name_to_idx = {n: i for i, n in enumerate(pathway_names)}
    src, dst = [], []
    for parent, child in hierarchy_edges:
        if parent in name_to_idx and child in name_to_idx:
            p, c = name_to_idx[parent], name_to_idx[child]
            src.extend([p, c])
            dst.extend([c, p])
    if not src:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)
