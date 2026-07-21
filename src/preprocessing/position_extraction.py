"""Spot position extraction and spatial neighbor graph construction."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc


class PositionExtractor:
    """Extract spot pixel coordinates from HEST-formatted h5ad files."""

    @staticmethod
    def extract(h5ad_path: str, save_path: Optional[str] = None) -> pd.DataFrame:
        adata = sc.read_h5ad(h5ad_path)
        coords = adata.obsm["spatial"]

        df = pd.DataFrame(
            {"x": coords[:, 0], "y": coords[:, 1]},
            index=[f"spot_{i}" for i in range(len(coords))],
        )

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path)

        return df

    @staticmethod
    def get_scalefactors(adata) -> dict:
        spatial_keys = list(adata.uns.get("spatial", {}).keys())
        if not spatial_keys:
            return {}
        lib_id = spatial_keys[0]
        return adata.uns["spatial"][lib_id].get("scalefactors", {})


class NeighborGraphBuilder:
    """Build spatial kNN graphs for spot neighborhoods."""

    def __init__(self, n_neighbors: int = 7):
        self.n_neighbors = n_neighbors

    def build(self, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        from sklearn.neighbors import NearestNeighbors

        knn = NearestNeighbors(n_neighbors=self.n_neighbors).fit(positions)
        distances, indices = knn.kneighbors(positions)
        return distances, indices

    def build_radius(
        self, positions: np.ndarray, radius: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        from sklearn.neighbors import NearestNeighbors

        if radius is None:
            from scipy.spatial.distance import pdist

            radius = np.median(pdist(positions[:min(500, len(positions))]))

        knn = NearestNeighbors(radius=radius).fit(positions)
        distances, indices = knn.radius_neighbors(positions)
        return distances, indices