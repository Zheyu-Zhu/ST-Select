"""MaxHerding: Generalized Coverage (NeurIPS 2024)."""

from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics.pairwise import rbf_kernel, pairwise_distances

from ..base import ActiveLearningStrategy
from ..registry import register


@register
class MaxHerding(ActiveLearningStrategy):
    """
    Continuous Gaussian-kernel coverage maximization.
    cov(S) = sum_j max_{i in S} K(x_j, x_i).
    Greedy maximization gives provably better worst-case coverage than k-Center.
    """

    name = "maxherding"
    family = "diversity"
    requires_features = True

    def __init__(self, sigma: Optional[float] = None, subsample_for_sigma: int = 1000):
        self.sigma = sigma
        self.subsample_for_sigma = subsample_for_sigma

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
        picks = self._greedy_maxherding(cand_feats, k)
        return [candidate_indices[i] for i in picks]

    def _greedy_maxherding(self, features: np.ndarray, k: int) -> List[int]:
        n = len(features)

        sigma = self.sigma
        if sigma is None:
            subset_size = min(self.subsample_for_sigma, n)
            sigma = float(np.median(
                pairwise_distances(features[:subset_size])
            ))

        gamma = 1.0 / (2.0 * sigma ** 2)
        K = rbf_kernel(features, gamma=gamma)

        max_sim = np.zeros(n)
        picks = []

        for _ in range(k):
            # Gain from adding each candidate
            gain = np.maximum(K, max_sim[np.newaxis, :]).sum(axis=1) - max_sim.sum()
            i = int(np.argmax(gain))
            picks.append(i)
            max_sim = np.maximum(max_sim, K[i])

        return picks