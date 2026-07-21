"""Model trainer for ST expression prediction."""

from typing import Dict, Optional

import numpy as np
import torch
from ..utils.reproducibility import resolve_device as _resolve_device
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset


class ALTrainer:
    """Train ST prediction models within the AL loop."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        epochs: int = 50,
        batch_size: int = 64,
        device: str = "cuda",
        loss_fn: str = "mse",
        scheduler: str = "cosine",
        patience: int = 10,
        num_workers: int = 0,
    ):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = _resolve_device(device)
        self.loss_fn_name = loss_fn
        self.scheduler_name = scheduler
        self.patience = patience
        self.num_workers = num_workers

        self.model.to(self.device)

    def _get_loss_fn(self):
        if self.loss_fn_name == "mse":
            return nn.MSELoss()
        elif self.loss_fn_name == "l1":
            return nn.L1Loss()
        elif self.loss_fn_name == "huber":
            return nn.HuberLoss()
        else:
            raise ValueError(f"Unknown loss: {self.loss_fn_name}")

    def train(
        self,
        dataset,
        selected_indices: list,
        val_dataset=None,
        val_indices: Optional[list] = None,
    ) -> Dict[str, list]:
        """Train model on selected subset."""
        train_subset = Subset(dataset, selected_indices)
        train_loader = DataLoader(
            train_subset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=(self.device == "cuda"),
        )

        val_loader = None
        if val_dataset is not None and val_indices is not None:
            val_subset = Subset(val_dataset, val_indices)
            val_loader = DataLoader(
                val_subset, batch_size=self.batch_size, shuffle=False,
                num_workers=self.num_workers,
            )

        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        if self.scheduler_name == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        elif self.scheduler_name == "step":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        else:
            scheduler = None

        loss_fn = self._get_loss_fn()
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                images = batch["image"].to(self.device)
                expressions = batch["expression"].to(self.device)

                optimizer.zero_grad()
                predictions = self.model(images)
                loss = loss_fn(predictions, expressions)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            history["train_loss"].append(avg_train_loss)

            if scheduler:
                scheduler.step()

            # Validation
            if val_loader is not None:
                val_loss = self._validate(val_loader, loss_fn)
                history["val_loss"].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        return history

    def _validate(self, val_loader, loss_fn) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(self.device)
                expressions = batch["expression"].to(self.device)
                predictions = self.model(images)
                loss = loss_fn(predictions, expressions)
                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def predict(self, dataloader: DataLoader) -> np.ndarray:
        """Get predictions for all samples in the dataloader."""
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(self.device)
                preds = self.model(images)
                all_preds.append(preds.cpu().numpy())

        return np.concatenate(all_preds, axis=0)

    def get_features(self, dataloader: DataLoader) -> np.ndarray:
        """Extract backbone features for all samples."""
        self.model.eval()
        all_feats = []

        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(self.device)
                if hasattr(self.model, "get_features"):
                    feats = self.model.get_features(images)
                elif hasattr(self.model, "backbone"):
                    feats = self.model.backbone(images)
                    if feats.dim() > 2:
                        feats = feats.mean(dim=[-2, -1])
                else:
                    feats = self.model(images)
                all_feats.append(feats.cpu().numpy())

        return np.concatenate(all_feats, axis=0)
