"""Random sampling baseline."""

from typing import Dict, List, Optional

import numpy as np
import torch

from ..base import ActiveLearningStrategy
from ..registry import register


@register
class RandomSampling(ActiveLearningStrategy):
    name = "random"
    family = "baseline"

    def __init__(self, seed: int = 42):
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
        chosen = self.rng.choice(candidate_indices, size=k, replace=False)
        return chosen.tolist()