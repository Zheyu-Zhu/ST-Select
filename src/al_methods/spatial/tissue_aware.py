"""Tissue-Aware Sampling: filter background spots before sampling."""

from typing import Dict, List, Optional

import numpy as np
import torch

from ..base import ActiveLearningStrategy
from ..registry import register


@register
class TissueAwareSampling(ActiveLearningStrategy):
    """
    Mask out background spots (low tissue area / low gene count) before
    applying any AL strategy. Wraps another strategy.
    """

    name = "tissue_aware"
    family = "spatial"

    def __init__(
        self,
        inner_strategy: Optional["ActiveLearningStrategy"] = None,
        min_gene_count: float = 100.0,
        seed: int = 42,
    ):
        self.inner_strategy = inner_strategy
        self.min_gene_count = min_gene_count
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
        # Filter candidates by tissue mask
        tissue_mask = extras.get("tissue_mask") if extras else None
        expression_sums = extras.get("expression_sums") if extras else None

        filtered_candidates = candidate_indices
        if tissue_mask is not None:
            filtered_candidates = [
                i for i in candidate_indices if tissue_mask[i]
            ]
        elif expression_sums is not None:
            filtered_candidates = [
                i for i in candidate_indices
                if expression_sums[i] >= self.min_gene_count
            ]

        if len(filtered_candidates) < k:
            filtered_candidates = candidate_indices

        # Delegate to inner strategy
        if self.inner_strategy is not None:
            return self.inner_strategy.select(
                candidate_indices=filtered_candidates,
                selected_indices=selected_indices,
                k=k,
                features=features,
                positions=positions,
                model=model,
                dataloader=dataloader,
                extras=extras,
            )

        # Default: random from filtered
        chosen = self.rng.choice(filtered_candidates, size=k, replace=False)
        return chosen.tolist()