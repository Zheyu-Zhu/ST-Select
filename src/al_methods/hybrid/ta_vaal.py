"""TA-VAAL: Task-Aware Variational Adversarial Active Learning (CVPR 2021)."""

from typing import Dict, List, Optional

import numpy as np
import torch
from ...utils.reproducibility import resolve_device as _resolve_device
import torch.nn as nn
import torch.optim as optim

from ..base import ActiveLearningStrategy
from ..registry import register
from ..base import model_device
from .vaal import VAE, Discriminator


class TaskAwareDiscriminator(nn.Module):
    """Discriminator augmented with task loss ranking signal."""

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),  # +1 for loss signal
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, z, loss_signal):
        x = torch.cat([z, loss_signal.unsqueeze(-1)], dim=-1)
        return self.net(x)


@register
class TAVAAL(ActiveLearningStrategy):
    """VAAL extended with task-aware loss ranking signal in the discriminator."""

    name = "ta_vaal"
    family = "adversarial"
    requires_features = True
    requires_model = True

    def __init__(
        self,
        latent_dim: int = 32,
        train_epochs: int = 50,
        lr: float = 5e-4,
        device: str = "cuda",
        batch_size: int = 256,
    ):
        self.latent_dim = latent_dim
        self.train_epochs = train_epochs
        self.lr = lr
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
        vae = VAE(input_dim, self.latent_dim).to(self.device)
        disc = TaskAwareDiscriminator(self.latent_dim).to(self.device)

        # Get loss predictions from a loss head if available
        loss_head = extras.get("loss_head") if extras else None

        loss_signals = self._compute_loss_signals(
            model, loss_head, features, candidate_indices, selected_indices
        )

        self._train_ta_vaal(
            vae, disc, features, candidate_indices, selected_indices, loss_signals
        )
        scores = self._score_candidates(vae, disc, features, candidate_indices, loss_signals)

        top_k = np.argsort(scores)[:k]
        return [candidate_indices[i] for i in top_k]

    def _compute_loss_signals(self, model, loss_head, features, candidates, selected):
        all_indices = candidates + selected
        x = torch.tensor(features[all_indices], dtype=torch.float32).to(self.device)

        model.eval()

        self.device = model_device(model, self.device)
        with torch.no_grad():
            if loss_head is not None:
                if hasattr(model, "backbone"):
                    feat = model.backbone(x)
                else:
                    feat = x
                signals = loss_head(feat).cpu().numpy()
            else:
                preds = model(x)
                signals = preds.norm(dim=-1).cpu().numpy()

        # Normalize to [0, 1]
        signals = (signals - signals.min()) / (signals.max() - signals.min() + 1e-8)
        return signals

    def _train_ta_vaal(self, vae, disc, features, candidates, selected, loss_signals):
        opt_vae = optim.Adam(vae.parameters(), lr=self.lr)
        opt_disc = optim.Adam(disc.parameters(), lr=self.lr)

        n_cand = len(candidates)
        labeled_feats = torch.tensor(features[selected], dtype=torch.float32)
        unlabeled_feats = torch.tensor(features[candidates], dtype=torch.float32)
        labeled_loss = torch.tensor(loss_signals[n_cand:], dtype=torch.float32)
        unlabeled_loss = torch.tensor(loss_signals[:n_cand], dtype=torch.float32)

        for _ in range(self.train_epochs):
            l_idx = np.random.choice(len(labeled_feats), min(self.batch_size, len(labeled_feats)), replace=False)
            u_idx = np.random.choice(len(unlabeled_feats), min(self.batch_size, len(unlabeled_feats)), replace=False)

            x_l = labeled_feats[l_idx].to(self.device)
            x_u = unlabeled_feats[u_idx].to(self.device)
            ls_l = labeled_loss[l_idx].to(self.device)
            ls_u = unlabeled_loss[u_idx].to(self.device)

            # VAE forward
            recon_l, mu_l, logvar_l, z_l = vae(x_l)
            recon_u, mu_u, logvar_u, z_u = vae(x_u)

            recon_loss = nn.functional.mse_loss(recon_l, x_l) + nn.functional.mse_loss(recon_u, x_u)
            kl_loss = -0.5 * (
                torch.mean(1 + logvar_l - mu_l.pow(2) - logvar_l.exp())
                + torch.mean(1 + logvar_u - mu_u.pow(2) - logvar_u.exp())
            )

            d_l = disc(z_l, ls_l)
            d_u = disc(z_u, ls_u)
            vae_adv = nn.functional.binary_cross_entropy(d_l, torch.zeros_like(d_l)) + \
                      nn.functional.binary_cross_entropy(d_u, torch.ones_like(d_u))

            vae_loss = recon_loss + kl_loss + vae_adv
            opt_vae.zero_grad()
            vae_loss.backward()
            opt_vae.step()

            # Discriminator
            with torch.no_grad():
                _, _, _, z_l = vae(x_l)
                _, _, _, z_u = vae(x_u)

            d_l = disc(z_l.detach(), ls_l)
            d_u = disc(z_u.detach(), ls_u)
            disc_loss = nn.functional.binary_cross_entropy(d_l, torch.ones_like(d_l)) + \
                        nn.functional.binary_cross_entropy(d_u, torch.zeros_like(d_u))

            opt_disc.zero_grad()
            disc_loss.backward()
            opt_disc.step()

    def _score_candidates(self, vae, disc, features, candidates, loss_signals):
        vae.eval()
        disc.eval()
        n_cand = len(candidates)
        scores = []

        x_all = torch.tensor(features[candidates], dtype=torch.float32)
        ls_all = torch.tensor(loss_signals[:n_cand], dtype=torch.float32)

        for i in range(0, len(x_all), self.batch_size):
            batch = x_all[i : i + self.batch_size].to(self.device)
            ls_batch = ls_all[i : i + self.batch_size].to(self.device)
            with torch.no_grad():
                _, _, _, z = vae(batch)
                score = disc(z, ls_batch).squeeze(-1)
            scores.append(score.cpu().numpy())

        return np.concatenate(scores)