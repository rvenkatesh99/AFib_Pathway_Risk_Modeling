import torch
import torch.nn as nn

class GlobalPathwayAttentionModel(nn.Module):

    def __init__(self, n_pathways, pathway_input_dim, covariate_dim,
                 embed_dim=64, dropout=0.1):
        super().__init__()
        self.n_pathways = n_pathways
        self.embed_dim  = embed_dim

        if n_pathways > 0:
            self.pathway_encoder = nn.Sequential(
                nn.Linear(pathway_input_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.attention_logits = nn.Parameter(torch.randn(n_pathways) * 0.01)
            self.pathway_context_norm = nn.LayerNorm(embed_dim)

        self.covariate_encoder = nn.Sequential(
            nn.Linear(covariate_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        head_in = (2 if n_pathways > 0 else 1) * embed_dim
        self.output_head = nn.Linear(head_in, 1)

    def get_attention_weights(self):
        if self.n_pathways == 0:
            return None
        return torch.softmax(self.attention_logits, dim=0)

    def forward(self, pathway_features, covariates):
        cov_embed = self.covariate_encoder(covariates)

        if self.n_pathways > 0:
            h     = self.pathway_encoder(pathway_features)
            alpha = self.get_attention_weights().unsqueeze(0).unsqueeze(-1)
            pathway_context = self.pathway_context_norm((alpha * h).sum(dim=1))
            combined = torch.cat([pathway_context, cov_embed], dim=-1)
        else:
            combined = cov_embed

        return self.output_head(combined).squeeze(-1)

    def get_pathway_importance_dict(self, pathway_names):
        weights = self.get_attention_weights()
        if weights is None:
            return {}
        return dict(zip(pathway_names, weights.detach().cpu().numpy()))
