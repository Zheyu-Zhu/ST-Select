"""CALD: Consistency-based Active Learning for Detection (CVPR 2021)."""

from typing import Dict, List, Optional

import numpy as np
import torch
from ...utils.reproducibility import resolve_device as _resolve_device

from ..base import ActiveLearningStrategy
from ..registry import register
from ..base import model_device


@register
class CALD(ActiveLearningStrategy):
    """
    Uncertainty from prediction inconsistency under augmentations.
    Score = average L2 distance between predictions of original and augmented inputs.
    """

    name = "cald"
    family = "medical"
    requires_model = True
    requires_features = True

    def __init__(
        self,
        n_augmentations: int = 5,
        noise_std: float = 0.1,
        device: str = "cuda",
        batch_size: int = 256,
    ):
        self.n_augmentations = n_augmentations
        self.noise_std = noise_std
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
        scores = self._compute_consistency_scores(model, features, candidate_indices)
        top_k = np.argsort(scores)[-k:]
        return [candidate_indices[i] for i in top_k]

    def _compute_consistency_scores(self, model, features, candidates):
        model.eval()
        self.device = model_device(model, self.device)
        x_all = torch.tensor(features[candidates], dtype=torch.float32)
        all_scores = []

        for i in range(0, len(x_all), self.batch_size):
            batch = x_all[i : i + self.batch_size].to(self.device)

            with torch.no_grad():
                pred_orig = model(batch)

            discrepancies = []
            for _ in range(self.n_augmentations):
                noise = torch.randn_like(batch) * self.noise_std
                augmented = batch + noise
                with torch.no_grad():
                    pred_aug = model(augmented)
                disc = (pred_aug - pred_orig).norm(dim=-1)
                discrepancies.append(disc)

            # Average inconsistency across augmentations
            avg_disc = torch.stack(discrepancies).mean(dim=0)
            all_scores.append(avg_disc.cpu().numpy())

        return np.concatenate(all_scores)