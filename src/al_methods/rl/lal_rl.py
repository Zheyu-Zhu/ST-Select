"""LAL-RL: Reinforcement-Learned Active Learning (NeurIPS 2018)."""

from typing import Dict, List, Optional

import numpy as np
import torch
from ...utils.reproducibility import resolve_device as _resolve_device
import torch.nn as nn
import torch.optim as optim

from ..base import ActiveLearningStrategy
from ..registry import register


class ALPolicy(nn.Module):
    """Simple policy network mapping state to per-sample scores."""

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


@register
class LALRL(ActiveLearningStrategy):
    """
    Treat AL as an MDP: policy outputs query scores per sample.
    Requires a pre-trained policy (from simulated AL episodes) or
    trains online with REINFORCE.
    """

    name = "lal_rl"
    family = "rl"
    requires_features = True

    def __init__(
        self,
        state_dim: Optional[int] = None,
        hidden_dim: int = 128,
        device: str = "cuda",
        batch_size: int = 256,
    ):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.device = _resolve_device(device)
        self.batch_size = batch_size
        self.policy: Optional[ALPolicy] = None

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
        # Use pre-trained policy if available
        if extras and "policy" in extras:
            self.policy = extras["policy"]

        if self.policy is None:
            # Fallback: use feature norm as a simple heuristic
            scores = np.linalg.norm(features[candidate_indices], axis=1)
            top_k = np.argsort(scores)[-k:]
            return [candidate_indices[i] for i in top_k]

        # Build state representation for each candidate
        states = self._build_states(features, candidate_indices, selected_indices)
        scores = self._score_with_policy(states)
        top_k = np.argsort(scores)[-k:]
        return [candidate_indices[i] for i in top_k]

    def _build_states(self, features, candidates, selected):
        """
        State per candidate: [candidate_feature, pool_statistics].
        Pool statistics: mean and std of selected features.
        """
        cand_feats = features[candidates]

        if selected:
            sel_feats = features[selected]
            sel_mean = sel_feats.mean(axis=0)
            sel_std = sel_feats.std(axis=0)
        else:
            sel_mean = np.zeros(features.shape[1])
            sel_std = np.ones(features.shape[1])

        # State = [candidate_feat, distance_to_mean, sel_mean, sel_std]
        dist_to_mean = np.linalg.norm(cand_feats - sel_mean, axis=1, keepdims=True)
        states = np.hstack([
            cand_feats,
            dist_to_mean,
            np.tile(sel_mean, (len(candidates), 1)),
            np.tile(sel_std, (len(candidates), 1)),
        ])
        return states

    def _score_with_policy(self, states):
        self.policy.eval()
        x = torch.tensor(states, dtype=torch.float32)
        scores = []

        for i in range(0, len(x), self.batch_size):
            batch = x[i : i + self.batch_size].to(self.device)
            with torch.no_grad():
                score = self.policy(batch)
            scores.append(score.cpu().numpy())

        return np.concatenate(scores)

    def train_policy(
        self,
        episodes: List[Dict],
        feature_dim: int,
        epochs: int = 100,
        lr: float = 1e-3,
    ) -> ALPolicy:
        """
        Train the policy from simulated AL episodes using REINFORCE.
        Each episode: {'states': np.ndarray, 'actions': np.ndarray, 'rewards': np.ndarray}
        """
        state_dim = feature_dim * 3 + 1  # feat + dist + mean + std
        self.policy = ALPolicy(state_dim, self.hidden_dim).to(self.device)
        optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        for _ in range(epochs):
            total_loss = 0.0
            for episode in episodes:
                states = torch.tensor(episode["states"], dtype=torch.float32).to(self.device)
                rewards = torch.tensor(episode["rewards"], dtype=torch.float32).to(self.device)

                scores = self.policy(states)
                log_probs = torch.log_softmax(scores, dim=0)

                # REINFORCE: maximize expected reward
                loss = -(log_probs * rewards).mean()
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self.policy