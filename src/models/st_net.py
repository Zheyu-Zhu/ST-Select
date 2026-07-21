"""ST-Net: DenseNet-121 backbone + MLP head (Nature Biomedical Engineering 2020)."""

import torch
import torch.nn as nn
import torchvision.models as models


class STNet(nn.Module):
    """
    ImageNet-pretrained DenseNet-121, global average pool,
    MLP head mapping 1024-dim features to HVG expression vector.
    """

    def __init__(
        self,
        n_genes: int = 300,
        pretrained: bool = True,
        frozen_backbone: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        densenet = models.densenet121(
            weights=models.DenseNet121_Weights.DEFAULT if pretrained else None
        )
        self.backbone = nn.Sequential(*list(densenet.features.children()))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.backbone_dim = 1024

        if frozen_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(self.backbone_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_genes),
        )

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)
        f = self.pool(f).flatten(1)
        return f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.get_features(x)
        return self.head(f)