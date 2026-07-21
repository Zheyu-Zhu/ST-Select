"""Poisson-Disk Sampling for spatially uniform coverage."""

from typing import Dict, List, Optional

import numpy as np
import torch

from ..base import ActiveLearningStrategy
from ..registry import register


@register
class PoissonDiskSampling(ActiveLearningStrategy):
    """
    Poisson-disk sampling on spot physical coordinates.
    Any two selected spots are at least r apart.
    Needs no features and no model — maximally uniform spatial coverage.
    """

    name = "poisson_disk"
    family = "spatial"
    requires_positions = True

    def __init__(self, radius: Optional[float] = None, seed: int = 42):
        self.radius = radius
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
        cand_positions = positions[candidate_indices]
        picks = self._poisson_disk(cand_positions, k)
        return [candidate_indices[i] for i in picks]

    def _poisson_disk(self, positions: np.ndarray, k: int) -> List[int]:
        n = len(positions)
        r = self.radius
        if r is None:
            x_range = positions[:, 0].max() - positions[:, 0].min()
            y_range = positions[:, 1].max() - positions[:, 1].min()
            area = max(x_range * y_range, 1e-6)
            r = np.sqrt(area / k) * 0.8

        # Greedy Poisson-disk: iteratively pick random point from valid candidates
        selected = []
        available = np.ones(n, dtype=bool)

        # Start from a random point
        order = self.rng.permutation(n)

        for idx in order:
            if not available[idx]:
                continue

            # Check distance to all already-selected
            if selected:
                sel_positions = positions[selected]
                dists = np.linalg.norm(sel_positions - positions[idx], axis=1)
                if dists.min() < r:
                    continue

            selected.append(idx)
            if len(selected) >= k:
                break

            # Mark nearby points as unavailable (optimization)
            all_dists = np.linalg.norm(positions - positions[idx], axis=1)
            too_close = all_dists < r
            available[too_close] = False
            available[idx] = False  # already selected

        # If we haven't reached k (radius too large), relax and fill remaining
        if len(selected) < k:
            remaining = [i for i in range(n) if i not in set(selected)]
            self.rng.shuffle(remaining)
            selected.extend(remaining[: k - len(selected)])

        return selected[:k]