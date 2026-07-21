"""CEAL: Cost-Effective Active Learning (CVPR 2017)."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from ...utils.reproducibility import resolve_device as _resolve_device

from ..base import ActiveLearningStrategy
from ..registry import register
from ..base import model_device


@register
class CEAL(ActiveLearningStrategy):
    """
    SSL+AL hybrid: low-confidence samples go to annotator,
    high-confidence ones get pseudo-labels and join training.
    Returns both query set and pseudo-labeled set.
    """

    name = "ceal"
    family = "medical"
    requires_model = True
    requires_features = True

    def __init__(
        self,
        confidence_threshold: float = 0.8,
        uncertainty_threshold: float = 0.3,
        device: str = "cuda",
        batch_size: int = 256,
    ):
        self.confidence_threshold = confidence_threshold
        self.uncertainty_threshold = uncertainty_threshold
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
        query_indices, pseudo_indices = self._split_candidates(
            model, features, candidate_indices, k
        )

        # Store pseudo-labeled indices in extras for downstream use
        if extras is not None:
            extras["pseudo_labeled"] = pseudo_indices

        return query_indices

    def _split_candidates(
        self, model, features, candidates, k
    ) -> Tuple[List[int], List[int]]:
        model.eval()
        self.device = model_device(model, self.device)
        x_all = torch.tensor(features[candidates], dtype=torch.float32)
        all_preds = []

        for i in range(0, len(x_all), self.batch_size):
            batch = x_all[i : i + self.batch_size].to(self.device)
            with torch.no_grad():
                pred = model(batch)
            all_preds.append(pred.cpu())

        preds = torch.cat(all_preds, dim=0)

        # For regression: confidence = inverse of prediction norm (low norm = confident)
        # uncertainty = high prediction norm
        uncertainty = preds.norm(dim=-1).numpy()
        max_unc = uncertainty.max()
        normalized_unc = uncertainty / (max_unc + 1e-8)

        # High confidence (low uncertainty) -> pseudo-label
        pseudo_mask = normalized_unc < (1.0 - self.confidence_threshold)
        # High uncertainty -> query
        query_mask = normalized_unc > self.uncertainty_threshold

        query_indices_local = np.where(query_mask)[0]
        pseudo_indices_local = np.where(pseudo_mask)[0]

        # Sort by uncertainty descending and take top-k for query
        sorted_query = query_indices_local[np.argsort(normalized_unc[query_indices_local])[::-1]]
        query_picks = sorted_query[:k].tolist()

        query_indices = [candidates[i] for i in query_picks]
        pseudo_indices = [candidates[i] for i in pseudo_indices_local]

        return query_indices, pseudo_indices