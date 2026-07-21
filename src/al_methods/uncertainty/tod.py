"""Temporal Output Discrepancy (ICCV 2021)."""

from typing import Dict, List, Optional

import numpy as np
import torch
from ...utils.reproducibility import resolve_device as _resolve_device

from ..base import ActiveLearningStrategy
from ..registry import register
from ..base import model_device


@register
class TemporalOutputDiscrepancy(ActiveLearningStrategy):
    """
    Score = discrepancy between predictions from two recent checkpoints.
    No auxiliary network needed — just keep two snapshots of the model.
    """

    name = "tod"
    family = "uncertainty"
    requires_model = True

    def __init__(self, device: str = "cuda", batch_size: int = 256, seed: int = 42):
        self.device = _resolve_device(device)
        self.batch_size = batch_size
        self.seed = seed
        self.rng = np.random.default_rng(seed)

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
        prev_model = extras.get("prev_model") if extras else None
        if prev_model is None:
            # No previous checkpoint yet (first AL round): temporal discrepancy
            # is undefined, so fall back to a random acquisition rather than
            # scoring against a copy of the current model (which would yield an
            # identically-zero, meaningless signal).
            chosen = self.rng.choice(len(candidate_indices), size=k, replace=False)
            return [candidate_indices[int(i)] for i in chosen]

        scores = self._compute_discrepancy(
            model, prev_model, candidate_indices, features
        )
        top_k = np.argsort(scores)[-k:]
        return [candidate_indices[i] for i in top_k]

    def _compute_discrepancy(self, model_curr, model_prev, candidates, features):
        model_curr.eval()
        self.device = model_device(model_curr, self.device)
        model_prev.eval()

        x_all = torch.tensor(features[candidates], dtype=torch.float32)
        all_scores = []

        for i in range(0, len(x_all), self.batch_size):
            batch = x_all[i : i + self.batch_size].to(self.device)
            with torch.no_grad():
                pred_curr = model_curr(batch)
                pred_prev = model_prev(batch)
            discrepancy = (pred_curr - pred_prev).norm(dim=-1)
            all_scores.append(discrepancy.cpu().numpy())

        return np.concatenate(all_scores)