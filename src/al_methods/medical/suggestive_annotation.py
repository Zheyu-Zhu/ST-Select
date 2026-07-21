"""Suggestive Annotation: Ensemble disagreement for medical imaging (MICCAI 2017)."""

from typing import Dict, List, Optional

import numpy as np
import torch
from ...utils.reproducibility import resolve_device as _resolve_device

from ..base import ActiveLearningStrategy
from ..registry import register
from ..base import model_device


@register
class SuggestiveAnnotation(ActiveLearningStrategy):
    """
    Ensemble of T models trained with bootstrapping.
    Score = prediction disagreement (variance across ensemble members).
    """

    name = "suggestive_annotation"
    family = "medical"
    requires_model = True

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
        ensemble = extras.get("ensemble") if extras else None
        if ensemble is None:
            raise ValueError("SuggestiveAnnotation requires extras['ensemble'] (list of models).")

        scores = self._compute_disagreement(ensemble, candidate_indices, features)
        top_k = np.argsort(scores)[-k:]
        return [candidate_indices[i] for i in top_k]

    def _compute_disagreement(self, ensemble, candidates, features):
        all_preds = []

        for model in ensemble:
            model.eval()
            self.device = model_device(model, self.device)
            preds = []
            x_all = torch.tensor(features[candidates], dtype=torch.float32)

            for i in range(0, len(x_all), self.batch_size):
                batch = x_all[i : i + self.batch_size].to(self.device)
                with torch.no_grad():
                    pred = model(batch)
                preds.append(pred.cpu())
            all_preds.append(torch.cat(preds, dim=0))

        stacked = torch.stack(all_preds, dim=0)  # (T, N, G)
        disagreement = stacked.var(dim=0).sum(dim=-1)  # (N,)
        return disagreement.numpy()