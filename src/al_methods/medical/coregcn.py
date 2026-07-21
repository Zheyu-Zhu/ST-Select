"""CoreGCN: Sequential GCN for Active Learning (CVPR 2021)."""

from typing import Dict, List, Optional

import numpy as np
import torch
from ...utils.reproducibility import resolve_device as _resolve_device
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import kneighbors_graph

from ..base import ActiveLearningStrategy
from ..registry import register


class GCNLayer(nn.Module):
    """Simple Graph Convolutional Layer."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # adj: (N, N) normalized adjacency
        h = self.linear(x)
        return torch.relu(adj @ h)


class QueryGCN(nn.Module):
    """GCN that predicts query value per node."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.gcn1(x, adj)
        h = self.gcn2(h, adj)
        return self.fc(h).squeeze(-1)


@register
class CoreGCN(ActiveLearningStrategy):
    """
    Build a feature-similarity graph over all samples, train a GCN to predict
    query value per node. Especially suited for ST data with natural spatial structure.
    """

    name = "coregcn"
    family = "medical"
    requires_features = True

    def __init__(
        self,
        n_neighbors: int = 10,
        hidden_dim: int = 128,
        train_epochs: int = 100,
        lr: float = 1e-3,
        device: str = "cuda",
        use_spatial_graph: bool = False,
    ):
        self.n_neighbors = n_neighbors
        self.hidden_dim = hidden_dim
        self.train_epochs = train_epochs
        self.lr = lr
        self.device = _resolve_device(device)
        self.use_spatial_graph = use_spatial_graph

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
        all_indices = candidate_indices + selected_indices
        all_feats = features[all_indices]

        # Build graph
        if self.use_spatial_graph and positions is not None:
            adj = self._build_spatial_graph(positions[all_indices])
        else:
            adj = self._build_feature_graph(all_feats)

        # Train GCN
        gcn = self._train_gcn(all_feats, adj, len(candidate_indices), len(selected_indices))

        # Score candidates
        scores = self._score_candidates(gcn, all_feats, adj, len(candidate_indices))
        top_k = np.argsort(scores)[-k:]
        return [candidate_indices[i] for i in top_k]

    def _build_feature_graph(self, features):
        adj_sparse = kneighbors_graph(features, self.n_neighbors, mode="connectivity")
        adj = adj_sparse.toarray()
        adj = adj + adj.T
        adj = (adj > 0).astype(float)
        # Add self-loops and normalize
        adj += np.eye(len(adj))
        degree = adj.sum(axis=1, keepdims=True)
        adj_norm = adj / degree
        return torch.tensor(adj_norm, dtype=torch.float32).to(self.device)

    def _build_spatial_graph(self, positions):
        adj_sparse = kneighbors_graph(positions, self.n_neighbors, mode="connectivity")
        adj = adj_sparse.toarray()
        adj = adj + adj.T
        adj = (adj > 0).astype(float)
        adj += np.eye(len(adj))
        degree = adj.sum(axis=1, keepdims=True)
        adj_norm = adj / degree
        return torch.tensor(adj_norm, dtype=torch.float32).to(self.device)

    def _train_gcn(self, features, adj, n_candidates, n_selected):
        input_dim = features.shape[1]
        gcn = QueryGCN(input_dim, self.hidden_dim).to(self.device)
        optimizer = optim.Adam(gcn.parameters(), lr=self.lr)

        x = torch.tensor(features, dtype=torch.float32).to(self.device)

        # Labels: 1 for labeled (selected), 0 for unlabeled (candidates)
        labels = torch.zeros(n_candidates + n_selected, device=self.device)
        labels[n_candidates:] = 1.0

        # Train mask: only supervise on nodes with known labels
        train_mask = torch.ones(n_candidates + n_selected, dtype=torch.bool, device=self.device)

        for _ in range(self.train_epochs):
            gcn.train()
            scores = gcn(x, adj)
            loss = nn.functional.binary_cross_entropy_with_logits(
                scores[train_mask], labels[train_mask]
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return gcn

    def _score_candidates(self, gcn, features, adj, n_candidates):
        gcn.eval()
        x = torch.tensor(features, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            scores = gcn(x, adj)
        # Return scores for candidate nodes only (first n_candidates)
        # Lower score = more "unlabeled-like" = more informative
        return (1.0 - torch.sigmoid(scores[:n_candidates])).cpu().numpy()