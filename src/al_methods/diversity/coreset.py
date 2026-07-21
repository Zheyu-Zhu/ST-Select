"""Core-Set active learning (ICLR 2018) — greedy k-Center."""

from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import pairwise_distances

from ..base import ActiveLearningStrategy
from ..registry import register


@register
class CoreSet(ActiveLearningStrategy):
    """Greedy k-Center: pick the unlabeled sample farthest from any selected sample."""

    name = "coreset"
    family = "diversity"
    requires_features = True

    def __init__(self, metric: str = "euclidean", batch_compute: bool = True):
        self.metric = metric
        self.batch_compute = batch_compute

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
        if self.batch_compute and len(candidate_indices) > 50000:
            return self._batched_k_center(features, candidate_indices, selected_indices, k)
        return self._greedy_k_center(features, candidate_indices, selected_indices, k)

    def _greedy_k_center(self, features, candidates, selected, k):
        cand_feats = features[candidates]
        sel_feats = features[selected] if selected else cand_feats[:1]

        dist_to_selected = pairwise_distances(
            cand_feats, sel_feats, metric=self.metric
        ).min(axis=1)

        picks = []
        for _ in range(k):
            i = int(np.argmax(dist_to_selected))
            picks.append(i)
            new_feat = cand_feats[i : i + 1]
            new_dist = pairwise_distances(
                cand_feats, new_feat, metric=self.metric
            ).ravel()
            dist_to_selected = np.minimum(dist_to_selected, new_dist)

        return [candidates[i] for i in picks]

    def _batched_k_center(self, features, candidates, selected, k, chunk_size=10000):
        """Memory-efficient version for large pools."""
        cand_feats = features[candidates]
        sel_feats = features[selected] if selected else cand_feats[:1]

        # Compute initial min-distances in chunks
        dist_to_selected = np.full(len(candidates), np.inf)
        for i in range(0, len(sel_feats), chunk_size):
            chunk = sel_feats[i : i + chunk_size]
            d = pairwise_distances(cand_feats, chunk, metric=self.metric).min(axis=1)
            dist_to_selected = np.minimum(dist_to_selected, d)

        picks = []
        for _ in range(k):
            i = int(np.argmax(dist_to_selected))
            picks.append(i)
            new_dist = np.linalg.norm(cand_feats - cand_feats[i], axis=1)
            dist_to_selected = np.minimum(dist_to_selected, new_dist)

        return [candidates[i] for i in picks]