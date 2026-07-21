"""ALFA-Mix: Active Learning by Feature Mixing (CVPR 2022)."""

from typing import Dict, List, Optional

import numpy as np
import torch
from ...utils.reproducibility import resolve_device as _resolve_device

from ..base import ActiveLearningStrategy
from ..registry import register
from ..base import model_device


@register
class ALFAMix(ActiveLearningStrategy):
    """
    Interpolate features of labeled and unlabeled samples:
    alpha * f_u + (1 - alpha) * f_l
    Find the alpha that flips the prediction. Samples needing smallest alpha are most informative.
    """

    name = "alfa_mix"
    family = "hybrid"
    requires_features = True
    requires_model = True

    def __init__(
        self,
        n_alpha_steps: int = 20,
        n_labeled_anchors: int = 10,
        device: str = "cuda",
        batch_size: int = 256,
    ):
        self.n_alpha_steps = n_alpha_steps
        self.n_labeled_anchors = n_labeled_anchors
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
        scores = self._compute_flip_scores(
            model, features, candidate_indices, selected_indices
        )
        top_k = np.argsort(scores)[:k]  # smaller alpha = more informative
        return [candidate_indices[i] for i in top_k]

    def _compute_flip_scores(self, model, features, candidates, selected):
        model.eval()
        self.device = model_device(model, self.device)

        cand_feats = torch.tensor(features[candidates], dtype=torch.float32).to(self.device)

        # Sample labeled anchors
        n_anchors = min(self.n_labeled_anchors, len(selected))
        anchor_idx = np.random.choice(selected, n_anchors, replace=False)
        anchor_feats = torch.tensor(features[anchor_idx], dtype=torch.float32).to(self.device)

        # Get predictions for candidates
        with torch.no_grad():
            cand_preds = model(cand_feats)  # (N_cand, G)

        # For each candidate, find smallest alpha that causes significant prediction change
        alphas = torch.linspace(0, 1, self.n_alpha_steps, device=self.device)
        flip_scores = np.ones(len(candidates))  # default: 1.0 (no flip)

        for i in range(0, len(cand_feats), self.batch_size):
            batch_feats = cand_feats[i : i + self.batch_size]
            batch_preds = cand_preds[i : i + self.batch_size]
            batch_size = len(batch_feats)

            min_alphas = torch.ones(batch_size, device=self.device)

            for anchor in anchor_feats:
                for alpha in alphas:
                    # Interpolate
                    mixed = alpha * batch_feats + (1 - alpha) * anchor.unsqueeze(0)
                    with torch.no_grad():
                        mixed_preds = model(mixed)

                    # Check for significant change (> threshold in L2 norm)
                    change = (mixed_preds - batch_preds).norm(dim=-1)
                    threshold = batch_preds.norm(dim=-1) * 0.5
                    flipped = change > threshold

                    # Update min alpha for flipped samples
                    min_alphas = torch.where(
                        flipped & (alpha < min_alphas),
                        torch.full_like(min_alphas, alpha.item()),
                        min_alphas,
                    )

            flip_scores[i : i + batch_size] = min_alphas.cpu().numpy()

        return flip_scores