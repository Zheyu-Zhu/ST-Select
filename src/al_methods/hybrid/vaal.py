"""VAAL: Variational Adversarial Active Learning (ICCV 2019)."""

from typing import Dict, List, Optional

import numpy as np
import torch
from ...utils.reproducibility import resolve_device as _resolve_device
import torch.nn as nn
import torch.optim as optim

from ..base import ActiveLearningStrategy
from ..registry import register


class VAE(nn.Module):
    """Simple VAE for VAAL."""

    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z


class Discriminator(nn.Module):
    """Discriminator: classifies labeled vs. unlabeled."""

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


@register
class VAAL(ActiveLearningStrategy):
    """
    Train VAE + Discriminator. Unlabeled samples the discriminator
    confidently labels as 'unlabeled' are the most under-represented.
    """

    name = "vaal"
    family = "adversarial"
    requires_features = True

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
        disc = Discriminator(self.latent_dim).to(self.device)

        self._train_vaal(vae, disc, features, candidate_indices, selected_indices)
        scores = self._score_candidates(vae, disc, features, candidate_indices)

        # Lower discriminator score = more "unlabeled-like" = more informative
        top_k = np.argsort(scores)[:k]
        return [candidate_indices[i] for i in top_k]

    def _train_vaal(self, vae, disc, features, candidates, selected):
        opt_vae = optim.Adam(vae.parameters(), lr=self.lr)
        opt_disc = optim.Adam(disc.parameters(), lr=self.lr)

        labeled_feats = torch.tensor(features[selected], dtype=torch.float32)
        unlabeled_feats = torch.tensor(features[candidates], dtype=torch.float32)

        for _ in range(self.train_epochs):
            # Sample batches
            l_idx = np.random.choice(len(labeled_feats), min(self.batch_size, len(labeled_feats)), replace=False)
            u_idx = np.random.choice(len(unlabeled_feats), min(self.batch_size, len(unlabeled_feats)), replace=False)

            x_l = labeled_feats[l_idx].to(self.device)
            x_u = unlabeled_feats[u_idx].to(self.device)

            # Train VAE
            recon_l, mu_l, logvar_l, z_l = vae(x_l)
            recon_u, mu_u, logvar_u, z_u = vae(x_u)

            recon_loss = nn.functional.mse_loss(recon_l, x_l) + nn.functional.mse_loss(recon_u, x_u)
            kl_loss = -0.5 * (
                torch.mean(1 + logvar_l - mu_l.pow(2) - logvar_l.exp())
                + torch.mean(1 + logvar_u - mu_u.pow(2) - logvar_u.exp())
            )

            # VAE tries to fool discriminator
            d_l = disc(z_l)
            d_u = disc(z_u)
            vae_adv = nn.functional.binary_cross_entropy(d_l, torch.zeros_like(d_l)) + \
                      nn.functional.binary_cross_entropy(d_u, torch.ones_like(d_u))

            vae_loss = recon_loss + kl_loss + vae_adv
            opt_vae.zero_grad()
            vae_loss.backward()
            opt_vae.step()

            # Train Discriminator
            with torch.no_grad():
                _, _, _, z_l = vae(x_l)
                _, _, _, z_u = vae(x_u)

            d_l = disc(z_l.detach())
            d_u = disc(z_u.detach())
            disc_loss = nn.functional.binary_cross_entropy(d_l, torch.ones_like(d_l)) + \
                        nn.functional.binary_cross_entropy(d_u, torch.zeros_like(d_u))

            opt_disc.zero_grad()
            disc_loss.backward()
            opt_disc.step()

    def _score_candidates(self, vae, disc, features, candidates):
        vae.eval()
        disc.eval()
        scores = []

        x_all = torch.tensor(features[candidates], dtype=torch.float32)
        for i in range(0, len(x_all), self.batch_size):
            batch = x_all[i : i + self.batch_size].to(self.device)
            with torch.no_grad():
                _, _, _, z = vae(batch)
                score = disc(z).squeeze(-1)
            scores.append(score.cpu().numpy())

        return np.concatenate(scores)