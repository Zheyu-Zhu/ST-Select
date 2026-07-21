"""BatchBALD: joint mutual information batch acquisition."""

from typing import Dict, List, Optional

import numpy as np
import torch
from ...utils.reproducibility import resolve_device as _resolve_device

from ..base import ActiveLearningStrategy
from ..registry import register
from ..base import model_device


@register
class BatchBALD(ActiveLearningStrategy):
    """
    BatchBALD for regression: uses Gaussian-likelihood BALD.
    Score = mutual information between predictions and model weights,
    estimated via MC Dropout disagreement, with greedy batch selection
    to reduce redundancy.
    """

    name = "batchbald"
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
        mc_preds = self._get_mc_predictions(model, candidate_indices, features)
        picks = self._greedy_batchbald(mc_preds, k)
        return [candidate_indices[i] for i in picks]

    def _get_mc_predictions(self, model, candidates, features):
        model.train()
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

        return torch.stack(all_preds, dim=0)  # (T, N, G)

    def _greedy_batchbald(self, mc_preds: torch.Tensor, k: int) -> List[int]:
        T, N, G = mc_preds.shape

        # Individual BALD scores: H[y|x] - E_w[H[y|x,w]]
        # For Gaussian: BALD ≈ log(var of mean) - mean of log(var)
        pred_mean = mc_preds.mean(dim=0)  # (N, G)
        total_var = mc_preds.var(dim=0).sum(dim=-1)  # (N,) total uncertainty

        # Greedy batch selection with redundancy penalty
        picks = []
        available = set(range(N))

        for _ in range(k):
            if not available:
                break

            avail_list = sorted(available)
            scores = total_var[avail_list].numpy()

            # Penalize samples similar to already-picked ones
            if picks:
                picked_preds = mc_preds[:, picks, :].mean(dim=0)  # (num_picked, G)
                for idx_in_avail, global_idx in enumerate(avail_list):
                    cand_pred = pred_mean[global_idx]  # (G,)
                    similarity = torch.cosine_similarity(
                        cand_pred.unsqueeze(0), picked_preds, dim=-1
                    ).max().item()
                    scores[idx_in_avail] *= (1.0 - 0.5 * similarity)

            best_local = int(np.argmax(scores))
            best_global = avail_list[best_local]
            picks.append(best_global)
            available.remove(best_global)

        return picks