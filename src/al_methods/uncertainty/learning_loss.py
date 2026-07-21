"""Learning Loss for Active Learning (CVPR 2019)."""

from typing import Dict, List, Optional

import numpy as np
import torch
from ...utils.reproducibility import resolve_device as _resolve_device
import torch.nn as nn

from ..base import ActiveLearningStrategy
from ..registry import register
from ..base import model_device


class LossPredictor(nn.Module):
    """Auxiliary head that predicts per-sample task loss."""

    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def loss_prediction_loss(
    predicted_loss: torch.Tensor, true_loss: torch.Tensor, margin: float = 1.0
) -> torch.Tensor:
    """Pairwise margin-ranking loss between predicted and actual losses."""
    n = len(predicted_loss)
    if n < 2:
        return torch.tensor(0.0, device=predicted_loss.device)

    # Create pairs
    idx = torch.randperm(n, device=predicted_loss.device)
    i, j = idx[: n // 2], idx[n // 2 : n // 2 * 2]

    pred_diff = predicted_loss[i] - predicted_loss[j]
    true_diff = true_loss[i] - true_loss[j]
    target = torch.sign(true_diff)

    return nn.functional.margin_ranking_loss(
        predicted_loss[i], predicted_loss[j], target, margin=margin
    )


@register
class LearningLoss(ActiveLearningStrategy):
    """Acquire samples with highest predicted loss from the auxiliary head."""

    name = "learning_loss"
    family = "uncertainty"
    requires_model = True

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
        loss_head = extras.get("loss_head") if extras else None
        if loss_head is None:
            raise ValueError("LearningLoss requires extras['loss_head'] (LossPredictor).")

        scores = self._score_candidates(model, loss_head, candidate_indices, features)
        top_k = np.argsort(scores)[-k:]
        return [candidate_indices[i] for i in top_k]

    def _score_candidates(self, model, loss_head, candidates, features):
        model.eval()
        self.device = model_device(model, self.device)
        loss_head.eval()

        x_all = torch.tensor(features[candidates], dtype=torch.float32)
        all_scores = []

        for i in range(0, len(x_all), self.batch_size):
            batch = x_all[i : i + self.batch_size].to(self.device)
            with torch.no_grad():
                feats = model.backbone(batch) if hasattr(model, "backbone") else model.get_features(batch)
                pred_loss = loss_head(feats)
            all_scores.append(pred_loss.cpu().numpy())

        return np.concatenate(all_scores)