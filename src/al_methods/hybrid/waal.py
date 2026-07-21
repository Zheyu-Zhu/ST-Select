"""WAAL: Wasserstein Adversarial Active Learning (AISTATS 2020)."""

from typing import Dict, List, Optional

import numpy as np
import torch
from ...utils.reproducibility import resolve_device as _resolve_device
import torch.nn as nn
import torch.optim as optim

from ..base import ActiveLearningStrategy
from ..registry import register


class WassersteinDiscriminator(nn.Module):
    """Critic for Wasserstein distance estimation."""

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


@register
class WAAL(ActiveLearningStrategy):
    """
    Wasserstein-distance regularizer to align labeled/unlabeled distributions.
    The discriminator output serves as the query score.
    """

    name = "waal"
    family = "adversarial"
    requires_features = True

    def __init__(
        self,
        train_epochs: int = 50,
        lr: float = 1e-3,
        clip_value: float = 0.01,
        n_critic: int = 5,
        device: str = "cuda",
        batch_size: int = 256,
    ):
        self.train_epochs = train_epochs
        self.lr = lr
        self.clip_value = clip_value
        self.n_critic = n_critic
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
        critic = WassersteinDiscriminator(input_dim).to(self.device)

        self._train_critic(critic, features, candidate_indices, selected_indices)
        scores = self._score_candidates(critic, features, candidate_indices)

        # Higher Wasserstein score = more different from labeled = more informative
        top_k = np.argsort(scores)[-k:]
        return [candidate_indices[i] for i in top_k]

    def _train_critic(self, critic, features, candidates, selected):
        opt = optim.RMSprop(critic.parameters(), lr=self.lr)

        labeled_feats = torch.tensor(features[selected], dtype=torch.float32)
        unlabeled_feats = torch.tensor(features[candidates], dtype=torch.float32)

        for _ in range(self.train_epochs):
            for _ in range(self.n_critic):
                l_idx = np.random.choice(len(labeled_feats), min(self.batch_size, len(labeled_feats)), replace=False)
                u_idx = np.random.choice(len(unlabeled_feats), min(self.batch_size, len(unlabeled_feats)), replace=False)

                x_l = labeled_feats[l_idx].to(self.device)
                x_u = unlabeled_feats[u_idx].to(self.device)

                # Wasserstein loss: maximize E[D(labeled)] - E[D(unlabeled)]
                loss = -(critic(x_l).mean() - critic(x_u).mean())

                opt.zero_grad()
                loss.backward()
                opt.step()

                # Weight clipping for Lipschitz constraint
                for p in critic.parameters():
                    p.data.clamp_(-self.clip_value, self.clip_value)

    def _score_candidates(self, critic, features, candidates):
        critic.eval()
        scores = []

        x_all = torch.tensor(features[candidates], dtype=torch.float32)
        for i in range(0, len(x_all), self.batch_size):
            batch = x_all[i : i + self.batch_size].to(self.device)
            with torch.no_grad():
                score = critic(batch).squeeze(-1)
            scores.append(score.cpu().numpy())

        return np.concatenate(scores)