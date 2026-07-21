"""CCAL: Contrastive Coding Active Learning (NeurIPS 2021)."""

from typing import Dict, List, Optional

import numpy as np
import torch
from ...utils.reproducibility import resolve_device as _resolve_device
import torch.nn as nn
import torch.optim as optim

from ..base import ActiveLearningStrategy
from ..registry import register


class ContrastiveEncoder(nn.Module):
    """Projection head for contrastive learning."""

    def __init__(self, input_dim: int, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, proj_dim),
        )

    def forward(self, x):
        return nn.functional.normalize(self.net(x), dim=-1)


@register
class CCAL(ActiveLearningStrategy):
    """
    Train semantic + OOD-aware contrastive encoders.
    Their disagreement identifies informative, in-distribution samples.
    """

    name = "ccal"
    family = "adversarial"
    requires_features = True

    def __init__(
        self,
        proj_dim: int = 128,
        train_epochs: int = 30,
        lr: float = 1e-3,
        temperature: float = 0.07,
        device: str = "cuda",
        batch_size: int = 256,
    ):
        self.proj_dim = proj_dim
        self.train_epochs = train_epochs
        self.lr = lr
        self.temperature = temperature
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
        input_dim = features.shape[1]
        enc_semantic = ContrastiveEncoder(input_dim, self.proj_dim).to(self.device)
        enc_ood = ContrastiveEncoder(input_dim, self.proj_dim).to(self.device)

        self._train_encoders(enc_semantic, enc_ood, features, selected_indices)
        scores = self._score_by_disagreement(
            enc_semantic, enc_ood, features, candidate_indices
        )

        # High disagreement = informative and in-distribution
        top_k = np.argsort(scores)[-k:]
        return [candidate_indices[i] for i in top_k]

    def _train_encoders(self, enc_sem, enc_ood, features, selected):
        opt_sem = optim.Adam(enc_sem.parameters(), lr=self.lr)
        opt_ood = optim.Adam(enc_ood.parameters(), lr=self.lr)

        labeled_feats = torch.tensor(features[selected], dtype=torch.float32)

        for _ in range(self.train_epochs):
            idx = np.random.choice(len(labeled_feats), min(self.batch_size, len(labeled_feats)), replace=False)
            x = labeled_feats[idx].to(self.device)

            # Add noise for augmentation pairs
            x_aug = x + 0.1 * torch.randn_like(x)

            # Semantic encoder: attract augmented pairs
            z1 = enc_sem(x)
            z2 = enc_sem(x_aug)
            sim = torch.mm(z1, z2.t()) / self.temperature
            labels = torch.arange(len(z1), device=self.device)
            loss_sem = nn.functional.cross_entropy(sim, labels)

            opt_sem.zero_grad()
            loss_sem.backward()
            opt_sem.step()

            # OOD encoder: trained with different augmentation strategy (stronger noise)
            x_strong = x + 0.5 * torch.randn_like(x)
            z1_ood = enc_ood(x)
            z2_ood = enc_ood(x_strong)
            sim_ood = torch.mm(z1_ood, z2_ood.t()) / self.temperature
            loss_ood = nn.functional.cross_entropy(sim_ood, labels)

            opt_ood.zero_grad()
            loss_ood.backward()
            opt_ood.step()

    def _score_by_disagreement(self, enc_sem, enc_ood, features, candidates):
        enc_sem.eval()
        enc_ood.eval()
        scores = []

        x_all = torch.tensor(features[candidates], dtype=torch.float32)
        for i in range(0, len(x_all), self.batch_size):
            batch = x_all[i : i + self.batch_size].to(self.device)
            with torch.no_grad():
                z_sem = enc_sem(batch)
                z_ood = enc_ood(batch)
            # Disagreement = 1 - cosine similarity
            agreement = (z_sem * z_ood).sum(dim=-1)
            disagreement = 1.0 - agreement
            scores.append(disagreement.cpu().numpy())

        return np.concatenate(scores)