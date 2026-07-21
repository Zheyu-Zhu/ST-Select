"""TypiClust: Active Learning on a Budget (ICML 2022)."""

from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from ..base import ActiveLearningStrategy
from ..registry import register


@register
class TypiClust(ActiveLearningStrategy):
    """
    K-Means the feature space into k clusters, then from each cluster
    pick the most typical sample (closest to cluster center).
    At low budgets, typical samples beat uncertain ones.
    """

    name = "typiclust"
    family = "diversity"
    requires_features = True

    def __init__(self, pca_dim: int = 128, seed: int = 42):
        self.pca_dim = pca_dim
        self.seed = seed

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
        cand_feats = features[candidate_indices]
        picks = self._typiclust(cand_feats, k)
        return [candidate_indices[i] for i in picks]

    def _reduce(self, features: np.ndarray) -> np.ndarray:
        """PCA-reduce features, robust to SVD non-convergence.

        Frozen FM features can be near-collinear / contain near-duplicate rows
        at small budgets, which makes the default randomized-SVD PCA raise
        ``LinAlgError: SVD did not converge``. Fall back to the full LAPACK
        solver, then to no reduction, so acquisition never crashes the run.
        Cast to float64 first — float32 (esp. from an MPS path) is what tips
        the SVD into non-convergence.
        """
        feats = np.ascontiguousarray(features, dtype=np.float64)
        pca_dim = min(self.pca_dim, feats.shape[1], feats.shape[0])
        if pca_dim < 1:
            return feats
        for solver in ("randomized", "full"):
            try:
                return PCA(
                    n_components=pca_dim, svd_solver=solver, random_state=self.seed
                ).fit_transform(feats)
            except np.linalg.LinAlgError:
                continue
        return feats  # last resort: cluster on the raw features

    def _typiclust(self, features: np.ndarray, k: int) -> List[int]:
        reduced = self._reduce(features)

        kmeans = KMeans(n_clusters=k, random_state=self.seed, n_init=10)
        labels = kmeans.fit_predict(reduced)

        picks = []
        for c in range(k):
            members = np.where(labels == c)[0]
            if len(members) == 0:
                continue
            center = reduced[members].mean(axis=0)
            dists = np.linalg.norm(reduced[members] - center, axis=1)
            most_typical = members[np.argmin(dists)]
            picks.append(int(most_typical))

        return picks