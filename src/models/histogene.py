"""HisToGene: ViT-based whole-slide token grid for per-spot expression (2022)."""

import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    """Convert image patches to embeddings."""

    def __init__(self, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class HisToGene(nn.Module):
    """
    ViT processes the whole slide image and outputs a token grid.
    Per-spot expression is regressed from the token at the spot's position.
    """

    def __init__(
        self,
        n_genes: int = 300,
        n_spots: int = 1000,
        embed_dim: int = 768,
        n_layers: int = 6,
        n_heads: int = 8,
        patch_size: int = 16,
        img_size: int = 224,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_size, 3, embed_dim)
        n_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, embed_dim) * 0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_genes)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x) + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return tokens.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.get_features(x)
        return self.head(f)

    def forward_with_positions(
        self, x: torch.Tensor, spot_token_indices: torch.Tensor
    ) -> torch.Tensor:
        """Predict expression using the token at each spot's grid position."""
        tokens = self.patch_embed(x) + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)

        # Gather tokens at spot positions
        batch_size = x.shape[0]
        spot_tokens = tokens[torch.arange(batch_size), spot_token_indices]
        return self.head(spot_tokens)