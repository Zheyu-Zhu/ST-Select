"""ProbCover: Active Learning Through a Covering Lens (NeurIPS 2022)."""

from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import pairwise_distances

from ..base import ActiveLearningStrategy
from ..registry import register


@register
class ProbCover(ActiveLearningStrategy):
    """
    Define a delta-ball graph on feature space, then greedy max-cover:
    repeatedly pick the node covering the most still-uncovered neighbors.
    """

    name = "probcover"
    family = "diversity"
    requires_features = True

    def __init__(self, delta: Optional[float] = None, delta_percentile: float = 0.1):
        self.delta = delta
        self.delta_percentile = delta_percentile

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
        cand_feats = features[candidate_indices]
        picks = self._greedy_max_cover(cand_feats, k)
        return [candidate_indices[i] for i in picks]

    def _greedy_max_cover(self, features: np.ndarray, k: int) -> List[int]:
        n = len(features)

        # Compute pairwise distances
        dist_matrix = pairwise_distances(features)

        # Determine delta (radius for coverage)
        delta = self.delta
        if delta is None:
            delta = float(np.percentile(dist_matrix.ravel(), self.delta_percentile * 100))

        # Build adjacency: node i covers node j if dist(i,j) < delta
        adjacency = dist_matrix < delta

        # Greedy max-cover
        covered = np.zeros(n, dtype=bool)
        picks = []

        for _ in range(k):
            # Count how many uncovered nodes each candidate would cover
            cover_counts = adjacency[:, ~covered].sum(axis=1)
            cover_counts[covered] = -1  # don't re-pick covered nodes as centers

            best = int(np.argmax(cover_counts))
            picks.append(best)
            covered |= adjacency[best]

        return picks