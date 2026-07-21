"""CDAL: Contextual Diversity for Active Learning (ECCV 2020)."""

from typing import Dict, List, Optional

import numpy as np
import torch
from ...utils.reproducibility import resolve_device as _resolve_device

from ..base import ActiveLearningStrategy
from ..registry import register
from ..base import model_device


@register
class CDAL(ActiveLearningStrategy):
    """
    Measure diversity by prediction divergence between sample pairs,
    then greedily pick samples maximizing total pairwise divergence.
    Adapted for regression: use L2 distance between prediction vectors as divergence.
    """

    name = "cdal"
    family = "diversity"
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
        predictions = self._get_predictions(model, candidate_indices, features)
        picks = self._greedy_contextual_diversity(predictions, k)
        return [candidate_indices[i] for i in picks]

    def _get_predictions(self, model, candidates, features):
        model.eval()
        self.device = model_device(model, self.device)
        x_all = torch.tensor(features[candidates], dtype=torch.float32)
        preds = []

        for i in range(0, len(x_all), self.batch_size):
            batch = x_all[i : i + self.batch_size].to(self.device)
            with torch.no_grad():
                pred = model(batch)
            preds.append(pred.cpu().numpy())

        return np.concatenate(preds, axis=0)

    def _greedy_contextual_diversity(self, predictions: np.ndarray, k: int) -> List[int]:
        n = len(predictions)
        picks = []
        available = set(range(n))

        # Precompute norms for fast distance computation
        norms = np.linalg.norm(predictions, axis=1)

        for _ in range(k):
            if not available:
                break

            best_idx = -1
            best_score = -np.inf

            avail_list = sorted(available)
            if not picks:
                # First pick: sample with highest norm (most extreme prediction)
                for idx in avail_list:
                    if norms[idx] > best_score:
                        best_score = norms[idx]
                        best_idx = idx
            else:
                # Subsequent picks: maximize min-distance to already picked
                picked_preds = predictions[picks]
                for idx in avail_list:
                    dists = np.linalg.norm(picked_preds - predictions[idx], axis=1)
                    score = dists.min()
                    if score > best_score:
                        best_score = score
                        best_idx = idx

            picks.append(best_idx)
            available.remove(best_idx)

        return picks