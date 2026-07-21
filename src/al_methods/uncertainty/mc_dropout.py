"""MC Dropout uncertainty estimation for active learning."""

from typing import Dict, List, Optional

import numpy as np
import torch
from ...utils.reproducibility import resolve_device as _resolve_device

from ..base import ActiveLearningStrategy
from ..registry import register
from ..base import model_device


@register
class MCDropout(ActiveLearningStrategy):
    """Score = sum of per-gene predictive variance over T stochastic forward passes."""

    name = "mc_dropout"
    family = "uncertainty"
    requires_model = True

    def __init__(self, T: int = 20, device: str = "cuda", batch_size: int = 256):
        self.T = T
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
        scores = self._compute_variance(model, candidate_indices, features, dataloader)
        top_k = np.argsort(scores)[-k:]
        return [candidate_indices[i] for i in top_k]

    def _compute_variance(self, model, candidates, features, dataloader):
        model.train()  # keep dropout active

        if features is not None:
            return self._variance_from_features(model, features, candidates)

        if dataloader is not None:
            return self._variance_from_dataloader(model, dataloader)

        raise ValueError("Either features or dataloader must be provided.")

    def _variance_from_features(self, model, features, candidates):
        all_preds = []
        x_all = torch.tensor(features[candidates], dtype=torch.float32)

        for _ in range(self.T):
            preds_t = []
            for i in range(0, len(x_all), self.batch_size):
                batch = x_all[i : i + self.batch_size].to(self.device)
                with torch.no_grad():
                    pred = model(batch)
                preds_t.append(pred.cpu())
            all_preds.append(torch.cat(preds_t, dim=0))

        stacked = torch.stack(all_preds, dim=0)  # (T, N, G)
        variance = stacked.var(dim=0).sum(dim=-1)  # (N,)
        return variance.numpy()

    def _variance_from_dataloader(self, model, dataloader):
        all_preds = []
        for _ in range(self.T):
            preds_t = []
            for batch in dataloader:
                x = batch["image"].to(self.device)
                with torch.no_grad():
                    pred = model(x)
                preds_t.append(pred.cpu())
            all_preds.append(torch.cat(preds_t, dim=0))

        stacked = torch.stack(all_preds, dim=0)
        variance = stacked.var(dim=0).sum(dim=-1)
        return variance.numpy()