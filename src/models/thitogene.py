"""THItoGene: Multi-scale hierarchical transformer (Briefings in Bioinformatics 2024)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .histogene import TransformerBlock


class MultiScaleEncoder(nn.Module):
    """Process concentric crops at multiple scales (112, 224, 448)."""

    def __init__(self, embed_dim: int = 256, patch_size: int = 16):
        super().__init__()
        self.scale_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(7),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, embed_dim),
            )
            for _ in range(3)  # 3 scales
        ])

    def forward(self, scales: list) -> torch.Tensor:
        """
        scales: list of 3 tensors [(B,3,112,112), (B,3,224,224), (B,3,448,448)]
        Returns: (B, 3, embed_dim)
        """
        embeddings = []
        for encoder, x in zip(self.scale_encoders, scales):
            embeddings.append(encoder(x))
        return torch.stack(embeddings, dim=1)


class THItoGene(nn.Module):
    """
    Multi-scale patches (112/224/448) fed into a hierarchical transformer.
    Scale tokens are fused via cross-scale attention before prediction.
    """

    def __init__(
        self,
        n_genes: int = 300,
        embed_dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.multi_scale = MultiScaleEncoder(embed_dim)

        # Hierarchical transformer: process scale tokens
        self.scale_transformer = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Fusion and prediction
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(embed_dim, n_genes)

    def forward(self, scales: list = None, x: torch.Tensor = None) -> torch.Tensor:
        """
        If scales provided: multi-scale input (list of 3 tensors).
        If x provided: single-scale fallback (B, 3, 224, 224).
        """
        if scales is not None:
            scale_tokens = self.multi_scale(scales)  # (B, 3, D)
        elif x is not None:
            scale_tokens = self.multi_scale([
                F.interpolate(x, size=(112, 112), mode="bilinear", align_corners=False),
                x,
                F.interpolate(x, size=(448, 448), mode="bilinear", align_corners=False),
            ])
        else:
            raise ValueError("Provide either scales or x.")

        # Cross-scale attention
        for block in self.scale_transformer:
            scale_tokens = block(scale_tokens)
        scale_tokens = self.norm(scale_tokens)

        # Flatten and fuse
        fused = self.fusion(scale_tokens.flatten(1))  # (B, D)
        return self.head(fused)

    def get_features(self, scales: list = None, x: torch.Tensor = None) -> torch.Tensor:
        if scales is not None:
            scale_tokens = self.multi_scale(scales)
        elif x is not None:
            scale_tokens = self.multi_scale([
                F.interpolate(x, size=(112, 112), mode="bilinear", align_corners=False),
                x,
                F.interpolate(x, size=(448, 448), mode="bilinear", align_corners=False),
            ])
        else:
            raise ValueError("Provide either scales or x.")

        for block in self.scale_transformer:
            scale_tokens = block(scale_tokens)
        scale_tokens = self.norm(scale_tokens)
        return self.fusion(scale_tokens.flatten(1))