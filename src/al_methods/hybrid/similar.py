"""SIMILAR / DISTIL: Submodular Information Measures for AL (NeurIPS 2021)."""

from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import pairwise_distances

from ..base import ActiveLearningStrategy
from ..registry import register


@register
class SIMILAR(ActiveLearningStrategy):
    """
    Submodular acquisition using Facility Location or Graph Cut objectives.
    Subsumes diversity, representation, and uncertainty as different submodular measures.
    """

    name = "similar"
    family = "hybrid"
    requires_features = True

    def __init__(
        self,
        objective: str = "facility_location",
        metric: str = "euclidean",
    ):
        self.objective = objective
        self.metric = metric

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

        if self.objective == "facility_location":
            picks = self._facility_location(cand_feats, k)
        elif self.objective == "graph_cut":
            picks = self._graph_cut(cand_feats, k)
        elif self.objective == "log_determinant":
            picks = self._log_det(cand_feats, k)
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

        return [candidate_indices[i] for i in picks]

    def _facility_location(self, features: np.ndarray, k: int) -> List[int]:
        """Greedy facility location: maximize sum of max-similarities to selected set."""
        n = len(features)
        # Compute similarity (negative distance)
        sim = -pairwise_distances(features, metric=self.metric)

        max_sim = np.full(n, -np.inf)
        picks = []

        for _ in range(k):
            # Marginal gain of adding each candidate
            gains = np.maximum(sim, max_sim[:, np.newaxis]).sum(axis=0) - max_sim.sum()
            # Don't re-pick
            for p in picks:
                gains[p] = -np.inf
            best = int(np.argmax(gains))
            picks.append(best)
            max_sim = np.maximum(max_sim, sim[:, best])

        return picks

    def _graph_cut(self, features: np.ndarray, k: int) -> List[int]:
        """Greedy graph cut: maximize inter-set similarity minus intra-set similarity."""
        n = len(features)
        sim = -pairwise_distances(features, metric=self.metric)
        # Normalize similarity to [0, 1]
        sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)

        lam = 1.0 / k  # Penalty for intra-set similarity
        picks = []
        selected_set = set()

        for _ in range(k):
            best_gain = -np.inf
            best_idx = -1

            for i in range(n):
                if i in selected_set:
                    continue
                # Gain: sum of similarity to non-selected minus lambda * sum to selected
                gain_out = sim[i, list(set(range(n)) - selected_set - {i})].sum()
                gain_in = sim[i, list(selected_set)].sum() if selected_set else 0
                gain = gain_out - lam * gain_in

                if gain > best_gain:
                    best_gain = gain
                    best_idx = i

            picks.append(best_idx)
            selected_set.add(best_idx)

        return picks

    def _log_det(self, features: np.ndarray, k: int) -> List[int]:
        """Greedy log-determinant: maximize log det of kernel submatrix."""
        from sklearn.metrics.pairwise import rbf_kernel

        sigma = float(np.median(pairwise_distances(features[:min(500, len(features))])))
        K = rbf_kernel(features, gamma=1.0 / (2 * sigma ** 2))
        K += 1e-6 * np.eye(len(K))  # regularize

        picks = []
        available = set(range(len(features)))

        for _ in range(k):
            best_gain = -np.inf
            best_idx = -1

            for i in available:
                test_set = picks + [i]
                sub_K = K[np.ix_(test_set, test_set)]
                gain = np.linalg.slogdet(sub_K)[1]
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i

            picks.append(best_idx)
            available.remove(best_idx)

        return picks