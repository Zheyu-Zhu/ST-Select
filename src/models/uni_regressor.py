"""UNIRegressor: UNI (ViT-L/16 pathology FM) backbone + MLP head, fine-tunable end-to-end.

The image-input counterpart of the frozen-UNI fast path: instead of caching UNI
features once, the UNI backbone is trained (or frozen) jointly with the head over
raw 224x224 H&E patches. Gradient checkpointing keeps ViT-L within a 10 GB GPU.

Interface matches STNet (`.backbone`, `.head`, `get_features`), so ALTrainer and
the AL strategies work unchanged. UNI is gated on HuggingFace (needs access +
`hf auth login`); ImageNet normalization is applied by PatchImageDataset.
"""

import torch
import torch.nn as nn


class UNIRegressor(nn.Module):
    def __init__(
        self,
        n_genes: int = 300,
        frozen_backbone: bool = False,
        grad_checkpointing: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError("UNIRegressor needs timm: pip install 'timm>=0.9'") from e
        # num_classes=0 -> backbone(x) returns the pooled 1024-d embedding.
        self.backbone = timm.create_model(
            "hf-hub:MahmoodLab/UNI", pretrained=True,
            init_values=1e-5, dynamic_img_size=True, num_classes=0,
        )
        self.backbone_dim = 1024

        if frozen_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        elif grad_checkpointing and hasattr(self.backbone, "set_grad_checkpointing"):
            # Recompute activations in backward -> big activation-memory saving,
            # which is what lets ViT-L fine-tune on a 10 GB card.
            self.backbone.set_grad_checkpointing(True)

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
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))
