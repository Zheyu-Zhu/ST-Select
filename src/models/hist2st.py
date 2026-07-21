"""Hist2ST: CNN + ViT + GNN for spatial transcriptomics (Briefings in Bioinformatics 2022)."""

import torch
import torch.nn as nn
import torchvision.models as models

from .histogene import TransformerBlock


class SpatialGNNLayer(nn.Module):
    """Graph neural network layer for spatial message passing."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.msg_linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        self_msg = self.linear(x)
        neighbor_msg = adj @ self.msg_linear(x)
        return torch.relu(self_msg + neighbor_msg)


class Hist2ST(nn.Module):
    """
    Three modules in series:
    1. CNN for per-patch features
    2. ViT for cross-spot global context
    3. GNN over 2D spot graph for spatial smoothing
    """

    def __init__(
        self,
        n_genes: int = 300,
        cnn_dim: int = 512,
        embed_dim: int = 256,
        n_transformer_layers: int = 4,
        n_gnn_layers: int = 2,
        n_heads: int = 4,
        pretrained: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        # CNN feature extractor (ResNet-18)
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.cnn_proj = nn.Linear(cnn_dim, embed_dim)

        # Transformer for global context
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, dropout=dropout)
            for _ in range(n_transformer_layers)
        ])
        self.transformer_norm = nn.LayerNorm(embed_dim)

        # GNN for spatial smoothing
        self.gnn_layers = nn.ModuleList([
            SpatialGNNLayer(embed_dim, embed_dim)
            for _ in range(n_gnn_layers)
        ])

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, n_genes),
        )

    def get_features(self, patches: torch.Tensor) -> torch.Tensor:
        """Extract CNN features from patches. patches: (B, 3, 224, 224)"""
        f = self.cnn(patches).flatten(1)
        return self.cnn_proj(f)

    def forward(
        self,
        patches: torch.Tensor,
        adj: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        patches: (N_spots, 3, 224, 224) or (N_spots, embed_dim) if pre-extracted
        adj: (N_spots, N_spots) normalized adjacency matrix
        """
        if patches.dim() == 4:
            features = self.get_features(patches)
        else:
            features = patches

        # Add batch dimension for transformer
        tokens = features.unsqueeze(0)  # (1, N, D)
        for block in self.transformer_blocks:
            tokens = block(tokens)
        tokens = self.transformer_norm(tokens).squeeze(0)  # (N, D)

        # GNN spatial smoothing
        if adj is not None:
            for gnn in self.gnn_layers:
                tokens = gnn(tokens, adj)

        return self.head(tokens)