"""Spatial-Stratified Sampling: grid-based equal allocation."""

from typing import Dict, List, Optional

import numpy as np
import torch

from ..base import ActiveLearningStrategy
from ..registry import register


@register
class SpatialStratified(ActiveLearningStrategy):
    """
    Partition the tissue area into a grid and draw equal counts from each cell.
    Surprisingly strong baseline for ST that beats random in most settings.
    """

    name = "spatial_stratified"
    family = "spatial"
    requires_positions = True

    def __init__(self, grid_size: Optional[int] = None, seed: int = 42):
        self.grid_size = grid_size
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
        picks = self._stratified_sample(cand_positions, k)
        return [candidate_indices[i] for i in picks]

    def _stratified_sample(self, positions: np.ndarray, k: int) -> List[int]:
        grid_size = self.grid_size
        if grid_size is None:
            grid_size = max(2, int(np.sqrt(k)))

        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()

        x_edges = np.linspace(x_min, x_max + 1e-6, grid_size + 1)
        y_edges = np.linspace(y_min, y_max + 1e-6, grid_size + 1)

        # Assign each spot to a grid cell
        x_bins = np.digitize(positions[:, 0], x_edges) - 1
        y_bins = np.digitize(positions[:, 1], y_edges) - 1
        cell_ids = x_bins * grid_size + y_bins

        # Group spots by cell
        cells = {}
        for i, cell_id in enumerate(cell_ids):
            cells.setdefault(cell_id, []).append(i)

        # Allocate budget proportionally to cell population, minimum 1 per non-empty cell
        non_empty_cells = {cid: members for cid, members in cells.items() if members}
        n_cells = len(non_empty_cells)

        if n_cells == 0:
            return []

        base_per_cell = max(1, k // n_cells)
        remainder = k - base_per_cell * n_cells

        picks = []
        for cid, members in non_empty_cells.items():
            n_from_cell = min(base_per_cell, len(members))
            if remainder > 0:
                extra = min(1, len(members) - n_from_cell)
                n_from_cell += extra
                remainder -= extra
            chosen = self.rng.choice(members, size=min(n_from_cell, len(members)), replace=False)
            picks.extend(chosen.tolist())

        # If we have too many (due to rounding), trim
        if len(picks) > k:
            picks = self.rng.choice(picks, size=k, replace=False).tolist()

        # If we have too few, fill randomly from remaining
        if len(picks) < k:
            remaining = [i for i in range(len(positions)) if i not in set(picks)]
            extra = self.rng.choice(remaining, size=k - len(picks), replace=False)
            picks.extend(extra.tolist())

        return picks[:k]