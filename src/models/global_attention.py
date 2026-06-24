"""
Global Pathway Attention Model.

Architecture:
  - Shared per-pathway encoder: Linear -> LayerNorm -> ReLU -> d-dim embedding
  - Learned population-level attention vector a in R^K; alpha = softmax(a)
  - Individual representation: weighted sum of pathway embeddings
  - Separate MLP for covariates
  - Concatenated -> linear output head -> sigmoid

The attention weights alpha are input-independent (same for every individual),
giving a directly interpretable population-level pathway importance ranking.
"""

import torch
import torch.nn as nn


class PathwayEncoder(nn.Module):
    """Shared encoder applied independently to each pathway's T-dim feature vector."""

    def __init__(self, in_features: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (batch, K, T) -> (batch, K, embed_dim)
        return self.net(x)


class CovariateEncoder(nn.Module):
    def __init__(self, in_features: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class GlobalPathwayAttentionModel(nn.Module):
    """
    Args:
        n_pathways: K
        pathway_input_dim: T (features per pathway)
        covariate_dim: C
        embed_dim: d
        dropout: dropout rate
    """

    def __init__(
        self,
        n_pathways: int,
        pathway_input_dim: int,
        covariate_dim: int,
        embed_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_pathways = n_pathways
        self.embed_dim = embed_dim

        self.pathway_encoder = PathwayEncoder(pathway_input_dim, embed_dim, dropout)
        self.covariate_encoder = CovariateEncoder(covariate_dim, embed_dim, dropout)

        # Population-level attention: one scalar per pathway
        self.attention_logits = nn.Parameter(torch.zeros(n_pathways))

        self.output_head = nn.Linear(2 * embed_dim, 1)

    def get_attention_weights(self) -> torch.Tensor:
        """Returns softmax-normalized pathway importance weights alpha (K,)."""
        return torch.softmax(self.attention_logits, dim=0)

    def forward(self, pathway_features: torch.Tensor, covariates: torch.Tensor):
        """
        pathway_features: (batch, K, T)
        covariates:       (batch, C)
        Returns logits (batch,) — use BCEWithLogitsLoss
        """
        # Encode each pathway: (batch, K, embed_dim)
        h = self.pathway_encoder(pathway_features)

        # Population-level attention weights: (K,) -> (1, K, 1) for broadcasting
        alpha = self.get_attention_weights().unsqueeze(0).unsqueeze(-1)

        # Weighted sum over pathways: (batch, embed_dim)
        pathway_context = (alpha * h).sum(dim=1)

        # Covariate embedding: (batch, embed_dim)
        cov_embed = self.covariate_encoder(covariates)

        # Concat and predict
        combined = torch.cat([pathway_context, cov_embed], dim=-1)
        logits = self.output_head(combined).squeeze(-1)
        return logits

    def get_pathway_importance_dict(self, pathway_names: list) -> dict:
        weights = self.get_attention_weights().detach().cpu().numpy()
        return dict(zip(pathway_names, weights))
