"""BADGE: Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds (ICLR 2020)."""

from typing import Dict, List, Optional

import numpy as np
import torch
from ...utils.reproducibility import resolve_device as _resolve_device

from ..base import ActiveLearningStrategy
from ..registry import register
from ..base import model_device


@register
class BADGE(ActiveLearningStrategy):
    """
    Gradient embedding + k-means++ (D^2 sampling).
    For regression: g = concat(backbone_features, predictions).
    Magnitude captures uncertainty, direction captures diversity.
    """

    name = "badge"
    family = "hybrid"
    requires_model = True
    requires_features = True

    def __init__(self, device: str = "cuda", batch_size: int = 256, seed: int = 42):
        self.device = _resolve_device(device)
        self.batch_size = batch_size
        self.seed = seed

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
        grad_embeddings = self._compute_gradient_embeddings(
            model, candidate_indices, features
        )
        picks = self._kmeans_pp(grad_embeddings, k)
        return [candidate_indices[i] for i in picks]

    def _compute_gradient_embeddings(self, model, candidates, features):
        model.eval()
        self.device = model_device(model, self.device)
        x_all = torch.tensor(features[candidates], dtype=torch.float32)
        embeddings = []

        for i in range(0, len(x_all), self.batch_size):
            batch = x_all[i : i + self.batch_size].to(self.device)
            with torch.no_grad():
                if hasattr(model, "backbone") and hasattr(model, "head"):
                    feat = model.backbone(batch)
                    pred = model.head(feat)
                else:
                    pred = model(batch)
                    feat = batch  # fallback: use input features
                emb = torch.cat([feat, pred], dim=-1)
            embeddings.append(emb.cpu().numpy())

        return np.concatenate(embeddings, axis=0)

    def _kmeans_pp(self, embeddings: np.ndarray, k: int) -> List[int]:
        """K-means++ initialization (D^2 sampling) on gradient embeddings."""
        rng = np.random.default_rng(self.seed)
        n = len(embeddings)

        first = rng.integers(0, n)
        centers = [first]
        min_dist = np.linalg.norm(embeddings - embeddings[first], axis=1) ** 2

        for _ in range(k - 1):
            probs = min_dist / min_dist.sum()
            new_center = rng.choice(n, p=probs)
            centers.append(int(new_center))
            new_dist = np.linalg.norm(embeddings - embeddings[new_center], axis=1) ** 2
            min_dist = np.minimum(min_dist, new_dist)

        return centers