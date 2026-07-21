"""FeaturePredictor: MLP head over pre-extracted (frozen) patch features.

This is the model for the tutorial's feature-cache fast path (§2.9.4). Frozen
features from a pathology FM (CONCH / UNI / DINOv2 / DenseNet) are extracted
once and cached, then this small head maps them to the HVG expression vector.

Crucially, its `forward` accepts feature vectors (B, feature_dim), so the AL
strategies that call `model(features)` (BADGE, TOD, ...) operate on the same
representation the predictor was trained on — no image-vs-feature mismatch.
"""

import torch
import torch.nn as nn


class FeaturePredictor(nn.Module):
    """MLP mapping cached features (B, feature_dim) -> expression (B, n_genes)."""

    def __init__(
        self,
        n_genes: int = 300,
        feature_dim: int = 1024,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        # `backbone` is identity here: cached features are already the backbone
        # output. Exposing it keeps the (backbone, head) interface that strategies
        # like BADGE probe via hasattr(model, "backbone").
        self.backbone = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_genes),
        )

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))
