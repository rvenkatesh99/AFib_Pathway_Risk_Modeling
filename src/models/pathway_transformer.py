"""
Pathway Transformer Model.

When n_pathways > 0:
  - Learned pathway identity embeddings + CLS token prepended to the pathway sequence.
  - Two transformer encoder layers with multi-head self-attention.
  - CLS token output fused with covariate embedding -> output head.

When n_pathways == 0 (covariates-only):
  - Transformer is skipped; prediction is from the covariate encoder only.
"""

import torch
import torch.nn as nn


class PathwayTransformer(nn.Module):

    def __init__(self, n_pathways, pathway_input_dim, covariate_dim,
                 embed_dim=64, n_heads=4, n_layers=2, dropout=0.1, feedforward_dim=128):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        self.n_pathways = n_pathways
        self.embed_dim  = embed_dim
        self.n_heads    = n_heads

        if n_pathways > 0:
            self.pathway_proj = nn.Sequential(
                nn.Linear(pathway_input_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.pathway_id_embed = nn.Embedding(n_pathways, embed_dim)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=n_heads, dim_feedforward=feedforward_dim,
                dropout=dropout, batch_first=True, norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self._pathway_ids = torch.arange(n_pathways)

        self.covariate_encoder = nn.Sequential(
            nn.Linear(covariate_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        head_in = (2 if n_pathways > 0 else 1) * embed_dim
        self.output_head = nn.Linear(head_in, 1)

    def forward(self, pathway_features, covariates):
        cov_embed = self.covariate_encoder(covariates)

        if self.n_pathways > 0:
            batch_size = pathway_features.size(0)
            h   = self.pathway_proj(pathway_features)
            ids = self._pathway_ids.to(pathway_features.device)
            h   = h + self.pathway_id_embed(ids).unsqueeze(0)
            seq = torch.cat([self.cls_token.expand(batch_size, -1, -1), h], dim=1)
            cls_out  = self.transformer(seq)[:, 0, :]
            combined = torch.cat([cls_out, cov_embed], dim=-1)
        else:
            combined = cov_embed

        return self.output_head(combined).squeeze(-1)

    def get_attention_weights(self, pathway_features, covariates):
        """Per-layer per-head attention weights. Returns list of (batch, n_heads, K+1, K+1)."""
        if self.n_pathways == 0:
            return []
        batch_size = pathway_features.size(0)
        h   = self.pathway_proj(pathway_features)
        ids = self._pathway_ids.to(pathway_features.device)
        h   = h + self.pathway_id_embed(ids).unsqueeze(0)
        seq = torch.cat([self.cls_token.expand(batch_size, -1, -1), h], dim=1)

        attn_all, x = [], seq
        for layer in self.transformer.layers:
            x_norm = layer.norm1(x)
            attn_out, attn_w = layer.self_attn(
                x_norm, x_norm, x_norm, need_weights=True, average_attn_weights=False)
            x = x + layer.dropout1(attn_out)
            x2 = layer.norm2(x)
            x = x + layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(x2)))))
            attn_all.append(attn_w.detach())
        return attn_all

    def compute_mean_attention(self, data_loader, device="cpu"):
        """Mean attention weights across the test set. Returns (n_layers, n_heads, K+1, K+1)."""
        if self.n_pathways == 0:
            return None
        self.eval()
        accum, n = None, 0
        with torch.no_grad():
            for batch in data_loader:
                pw  = batch["pathway_features"].to(device)
                cov = batch["covariates"].to(device)
                w   = torch.stack([a.mean(0) for a in self.get_attention_weights(pw, cov)])
                accum = w if accum is None else accum + w
                n += 1
        return (accum / n).cpu()
