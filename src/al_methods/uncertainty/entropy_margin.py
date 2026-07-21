"""Classical uncertainty baselines: Entropy, Margin, Least Confidence.

Adapted for regression (ST): uncertainty = per-spot prediction variance or output magnitude.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
from ...utils.reproducibility import resolve_device as _resolve_device

from ..base import ActiveLearningStrategy
from ..registry import register
from ..base import model_device


@register
class EntropySampling(ActiveLearningStrategy):
    """For regression: score = sum of absolute predictions (proxy for output magnitude)."""

    name = "entropy"
    family = "uncertainty"
    requires_model = True

    def __init__(self, device: str = "cuda"):
        self.device = _resolve_device(device)

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
        model.eval()
        self.device = model_device(model, self.device)
        scores = self._compute_scores(model, candidate_indices, features, dataloader, extras)
        top_k = np.argsort(scores)[-k:]
        return [candidate_indices[i] for i in top_k]

    def _compute_scores(self, model, candidates, features, dataloader, extras):
        if features is not None:
            x = torch.tensor(features[candidates], dtype=torch.float32).to(self.device)
            with torch.no_grad():
                preds = model(x)
            return preds.abs().sum(dim=-1).cpu().numpy()

        scores = []
        if dataloader is not None:
            with torch.no_grad():
                for batch in dataloader:
                    x = batch["image"].to(self.device)
                    preds = model(x)
                    scores.append(preds.abs().sum(dim=-1).cpu().numpy())
            return np.concatenate(scores)

        raise ValueError("Either features or dataloader must be provided.")


@register
class MarginSampling(ActiveLearningStrategy):
    """For regression: score = variance across output genes (high spread = uncertain)."""

    name = "margin"
    family = "uncertainty"
    requires_model = True

    def __init__(self, device: str = "cuda"):
        self.device = _resolve_device(device)

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
        model.eval()
        self.device = model_device(model, self.device)
        x = torch.tensor(features[candidate_indices], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = model(x)
        scores = preds.var(dim=-1).cpu().numpy()
        top_k = np.argsort(scores)[-k:]
        return [candidate_indices[i] for i in top_k]


@register
class LeastConfidence(ActiveLearningStrategy):
    """For regression: score = L2 norm of prediction (farthest from zero = most uncertain)."""

    name = "least_confidence"
    family = "uncertainty"
    requires_model = True

    def __init__(self, device: str = "cuda"):
        self.device = _resolve_device(device)

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
        model.eval()
        self.device = model_device(model, self.device)
        x = torch.tensor(features[candidate_indices], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = model(x)
        scores = preds.norm(dim=-1).cpu().numpy()
        top_k = np.argsort(scores)[-k:]
        return [candidate_indices[i] for i in top_k]