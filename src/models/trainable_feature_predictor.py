"""TrainableFeaturePredictor: a learnable projection over frozen FM features.

Motivation
----------
The default FeaturePredictor uses the cached UNI embedding *directly* as the
representation — `backbone` is Identity, so the feature space AL strategies see
is fixed across all acquisition rounds. That is the "frozen features" regime.

Real image->ST models instead train (or fine-tune) their backbone, so the
representation shifts every round. This class is a lightweight, controlled proxy
for that regime: it keeps the (frozen) UNI features as input but inserts a
**learnable projection** as the `backbone`. Because that projection is updated
every training round, `get_features()` returns a *different* embedding each
round — giving feature-based AL strategies (BADGE, CoreSet, TypiClust, entropy,
...) a "live" signal. This isolates the single variable of interest — does a
moving representation change the AL ranking? — without the confound / cost of
back-propagating a full DenseNet over raw image patches.

Interface matches FeaturePredictor: exposes `.backbone` (learnable here) and
`.head`, plus `get_features()`, so all AL strategies and the trainer work
unchanged.
"""

import torch
import torch.nn as nn


class TrainableFeaturePredictor(nn.Module):
    """Learnable projection (backbone) + regression head over frozen features.

    forward:  x (B, feature_dim) -> proj (B, proj_dim) -> expression (B, n_genes)
    The projection is the "trainable backbone"; its output is what AL sees.
    """

    def __init__(
        self,
        n_genes: int = 300,
        feature_dim: int = 1024,
        proj_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        # Learnable representation — updated every round, unlike the frozen
        # Identity backbone in FeaturePredictor. This is what makes the feature
        # space "move" for the AL strategies.
        self.backbone = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, n_genes),
        )

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))
