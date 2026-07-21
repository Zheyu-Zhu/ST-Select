"""ConfiDNet: Learned confidence for failure prediction and AL (NeurIPS 2019)."""

from typing import Dict, List, Optional

import numpy as np
import torch
from ...utils.reproducibility import resolve_device as _resolve_device
import torch.nn as nn

from ..base import ActiveLearningStrategy
from ..registry import register
from ..base import model_device


class ConfidenceHead(nn.Module):
    """Auxiliary head predicting calibrated confidence."""

    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def confidence_loss(
    predicted_conf: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """TCP loss: confidence should predict 1 - normalized_error."""
    error = (predictions - targets).pow(2).mean(dim=-1)
    max_error = error.max().detach() + 1e-8
    true_conf = 1.0 - (error / max_error)
    return nn.functional.mse_loss(predicted_conf, true_conf.detach())


@register
class ConfiDNet(ActiveLearningStrategy):
    """Acquire samples with lowest predicted confidence."""

    name = "confidnet"
    family = "medical"
    requires_model = True
    requires_features = True

    def __init__(self, device: str = "cuda", batch_size: int = 256):
        self.device = _resolve_device(device)
        self.batch_size = batch_size

    def select(
        self,
        candidate_indices: List[int],
        selected_indices: List[int],
        k: int,
        features: Optional[np.ndarray] = None,
        positions: Optional[np.ndarray] = None,
        model: Optional[torch.nn.Module] = None,
        dataloader: Optional[torch.utils.data.DataLoader] = None,
        extras: Optional[Dict] = None,
    ) -> List[int]:
        conf_head = extras.get("confidence_head") if extras else None
        if conf_head is None:
            raise ValueError("ConfiDNet requires extras['confidence_head'].")

        scores = self._score_candidates(model, conf_head, features, candidate_indices)
        # Low confidence = most informative
        top_k = np.argsort(scores)[:k]
        return [candidate_indices[i] for i in top_k]

    def _score_candidates(self, model, conf_head, features, candidates):
        model.eval()
        self.device = model_device(model, self.device)
        conf_head.eval()

        x_all = torch.tensor(features[candidates], dtype=torch.float32)
        scores = []

        for i in range(0, len(x_all), self.batch_size):
            batch = x_all[i : i + self.batch_size].to(self.device)
            with torch.no_grad():
                if hasattr(model, "backbone"):
                    feat = model.backbone(batch)
                else:
                    feat = batch
                conf = conf_head(feat)
            scores.append(conf.cpu().numpy())

        return np.concatenate(scores)