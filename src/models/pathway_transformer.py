"""
Pathway Transformer Model.

Architecture:
  - Learned pathway identity embeddings (no positional encoding; pathways have no order)
  - CLS token prepended to the pathway sequence
  - Two transformer encoder layers with multi-head self-attention
  - CLS token output used for prediction after covariate fusion
  - Interpretability: mean test-set attention weights across individuals
"""

import torch
import torch.nn as nn


class PathwayTransformer(nn.Module):
    """
    Args:
        n_pathways: K
        pathway_input_dim: T (features per pathway)
        covariate_dim: C
        embed_dim: d (must be divisible by n_heads)
        n_heads: number of attention heads
        n_layers: number of transformer encoder layers (default 2)
        dropout: dropout rate
    """

    def __init__(
        self,
        n_pathways: int,
        pathway_input_dim: int,
        covariate_dim: int,
        embed_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        feedforward_dim: int = 128,
    ):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        self.n_pathways = n_pathways
        self.embed_dim = embed_dim
        self.n_heads = n_heads

        # Project each pathway's T features to embed_dim
        self.pathway_proj = nn.Sequential(
            nn.Linear(pathway_input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Pathway identity embeddings (replace positional encodings)
        self.pathway_id_embed = nn.Embedding(n_pathways, embed_dim)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Covariate encoder
        self.covariate_encoder = nn.Sequential(
            nn.Linear(covariate_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.output_head = nn.Linear(2 * embed_dim, 1)

        self._pathway_ids = torch.arange(n_pathways)

    def forward(self, pathway_features: torch.Tensor, covariates: torch.Tensor):
        """
        pathway_features: (batch, K, T)
        covariates:       (batch, C)
        Returns logits (batch,)
        """
        batch_size = pathway_features.size(0)

        # Project pathway features: (batch, K, embed_dim)
        h = self.pathway_proj(pathway_features)

        # Add pathway identity embeddings
        ids = self._pathway_ids.to(pathway_features.device)
        h = h + self.pathway_id_embed(ids).unsqueeze(0)  # (batch, K, embed_dim)

        # Prepend CLS token: (batch, K+1, embed_dim)
        cls = self.cls_token.expand(batch_size, -1, -1)
        seq = torch.cat([cls, h], dim=1)

        # Transformer encoder
        out = self.transformer(seq)  # (batch, K+1, embed_dim)

        # CLS token output as sequence representation
        cls_out = out[:, 0, :]  # (batch, embed_dim)

        # Covariate fusion
        cov_embed = self.covariate_encoder(covariates)
        combined = torch.cat([cls_out, cov_embed], dim=-1)
        logits = self.output_head(combined).squeeze(-1)
        return logits

    def get_attention_weights(self, pathway_features: torch.Tensor, covariates: torch.Tensor):
        """
        Extract per-layer per-head attention weights for interpretability.
        Returns list of (batch, n_heads, K+1, K+1) tensors, one per layer.

        NOTE: This requires hooking into the TransformerEncoderLayer internals.
        We re-run the forward pass with attention weight extraction enabled.
        """
        batch_size = pathway_features.size(0)
        h = self.pathway_proj(pathway_features)
        ids = self._pathway_ids.to(pathway_features.device)
        h = h + self.pathway_id_embed(ids).unsqueeze(0)
        cls = self.cls_token.expand(batch_size, -1, -1)
        seq = torch.cat([cls, h], dim=1)

        attn_weights_all_layers = []
        x = seq
        for layer in self.transformer.layers:
            # Pre-norm
            x_norm = layer.norm1(x)
            # Run self-attention with need_weights=True
            attn_out, attn_weights = layer.self_attn(
                x_norm, x_norm, x_norm,
                need_weights=True, average_attn_weights=False,
            )
            # Residual + rest of the layer
            x = x + layer.dropout1(attn_out)
            x2 = layer.norm2(x)
            x = x + layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(x2)))))
            attn_weights_all_layers.append(attn_weights.detach())

        return attn_weights_all_layers  # list[(batch, n_heads, K+1, K+1)]

    def compute_mean_attention(self, data_loader, device: str = "cpu"):
        """
        Compute mean attention weights across the test set.
        Returns (n_layers, n_heads, K+1, K+1) tensor.
        """
        self.eval()
        accum = None
        n_batches = 0
        with torch.no_grad():
            for batch in data_loader:
                pw = batch["pathway_features"].to(device)
                cov = batch["covariates"].to(device)
                weights = self.get_attention_weights(pw, cov)
                batch_mean = torch.stack([w.mean(0) for w in weights])  # (L, H, S, S)
                if accum is None:
                    accum = batch_mean
                else:
                    accum = accum + batch_mean
                n_batches += 1
        return (accum / n_batches).cpu()
