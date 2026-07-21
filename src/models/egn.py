"""EGN: Exemplar Guided Network (WACV 2023)."""

import torch
import torch.nn as nn
import numpy as np


class CrossAttentionBlock(nn.Module):
    """Cross-attention between query and exemplar features."""

    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, exemplars: torch.Tensor) -> torch.Tensor:
        h = self.norm1(query)
        h, _ = self.cross_attn(h, exemplars, exemplars)
        query = query + h
        query = query + self.mlp(self.norm2(query))
        return query


class EGN(nn.Module):
    """
    Exemplar Guided Network: query patch attends to K nearest-neighbor
    'exemplar' patches from a feature bank. Exemplar expressions are fused
    with the query feature via cross-attention.
    """

    def __init__(
        self,
        n_genes: int = 300,
        feature_dim: int = 768,
        n_exemplars: int = 5,
        n_attention_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_exemplars = n_exemplars
        self.feature_dim = feature_dim

        # Project expression to feature space
        self.expr_proj = nn.Linear(n_genes, feature_dim)

        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(feature_dim, n_heads, dropout)
            for _ in range(n_attention_layers)
        ])

        # Final prediction head
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, n_genes),
        )

        # Feature bank (set externally)
        self.feature_bank: torch.Tensor = None
        self.expression_bank: torch.Tensor = None

    def set_feature_bank(self, features: np.ndarray, expressions: np.ndarray):
        """Set the retrieval bank from training data."""
        self.feature_bank = torch.tensor(features, dtype=torch.float32)
        self.expression_bank = torch.tensor(expressions, dtype=torch.float32)

    def retrieve_exemplars(self, query_features: torch.Tensor) -> tuple:
        """Retrieve K nearest neighbors from the feature bank."""
        # Cosine similarity
        query_norm = nn.functional.normalize(query_features, dim=-1)
        bank_norm = nn.functional.normalize(self.feature_bank.to(query_features.device), dim=-1)

        sim = query_norm @ bank_norm.T  # (B, N_bank)
        _, topk_idx = sim.topk(self.n_exemplars, dim=-1)  # (B, K)

        exemplar_feats = self.feature_bank.to(query_features.device)[topk_idx]  # (B, K, D)
        exemplar_exprs = self.expression_bank.to(query_features.device)[topk_idx]  # (B, K, G)

        return exemplar_feats, exemplar_exprs

    def forward(self, query_features: torch.Tensor) -> torch.Tensor:
        """
        query_features: (B, feature_dim) pre-extracted patch features.
        """
        if self.feature_bank is None:
            raise RuntimeError("Call set_feature_bank() before forward.")

        exemplar_feats, exemplar_exprs = self.retrieve_exemplars(query_features)

        # Project exemplar expressions and combine with features
        expr_embedded = self.expr_proj(exemplar_exprs)  # (B, K, D)
        exemplar_tokens = exemplar_feats + expr_embedded

        # Cross-attend query to exemplars
        query = query_features.unsqueeze(1)  # (B, 1, D)
        for layer in self.cross_attn_layers:
            query = layer(query, exemplar_tokens)

        query = query.squeeze(1)  # (B, D)
        return self.head(query)