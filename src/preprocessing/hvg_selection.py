"""Highly Variable Gene (HVG) selection across slides."""

from pathlib import Path
from typing import List, Optional

import numpy as np
import scanpy as sc
from anndata import AnnData


class HVGSelector:
    """Joint HVG selection across all slides in a dataset."""

    def __init__(
        self,
        n_top_genes: int = 1000,
        target_sum: float = 1e4,
        flavor: str = "seurat",
    ):
        self.n_top_genes = n_top_genes
        self.target_sum = target_sum
        self.flavor = flavor
        self.hvg_names: Optional[List[str]] = None
        self.combined: Optional[AnnData] = None

    def fit(self, h5ad_paths: List[str], slide_ids: Optional[List[str]] = None) -> "HVGSelector":
        if slide_ids is None:
            slide_ids = [Path(p).stem for p in h5ad_paths]

        adatas = [sc.read_h5ad(p) for p in h5ad_paths]
        self.combined = sc.concat(adatas, label="_slide_id", keys=slide_ids)
        # Stash the raw counts before any normalization so the regression target
        # can be built either as log1p(CP10k) (default) or log1p(raw) — the latter
        # retains per-spot depth/density signal that images predict well and can
        # roughly double per-gene PCC (see extract_expression_matrices target=).
        self.combined.layers["counts"] = self.combined.X.copy()

        if self.flavor == "seurat_v3":
            # seurat_v3 expects raw counts and ranks genes *before* normalization.
            try:
                sc.pp.highly_variable_genes(
                    self.combined, n_top_genes=self.n_top_genes, flavor="seurat_v3"
                )
            except (ImportError, ValueError) as e:
                raise RuntimeError(
                    "flavor='seurat_v3' requires the 'scikit-misc' package (loess). "
                    "Install it (`pip install scikit-misc`) or use flavor='seurat'. "
                    f"Underlying error: {e}"
                ) from e
            sc.pp.normalize_total(self.combined, target_sum=self.target_sum)
            sc.pp.log1p(self.combined)
        else:
            # seurat / cell_ranger expect normalized, log1p data.
            sc.pp.normalize_total(self.combined, target_sum=self.target_sum)
            sc.pp.log1p(self.combined)
            sc.pp.highly_variable_genes(
                self.combined, n_top_genes=self.n_top_genes, flavor=self.flavor
            )

        # Rank HVGs so the list is reproducible. seurat/cell_ranger expose
        # 'dispersions_norm'; seurat_v3 exposes 'highly_variable_rank'.
        var = self.combined.var
        hvg_mask = var["highly_variable"]
        if "highly_variable_rank" in var.columns:
            ranked = var.loc[hvg_mask, "highly_variable_rank"].sort_values(ascending=True)
        else:
            ranked = var.loc[hvg_mask, "dispersions_norm"].sort_values(ascending=False)
        self.hvg_names = ranked.index.tolist()
        return self

    def get_hvg_names(self, top_n: Optional[int] = None) -> List[str]:
        if self.hvg_names is None:
            raise RuntimeError("Call fit() first.")
        if top_n is None:
            return self.hvg_names
        return self.hvg_names[:top_n]

    def extract_expression_matrices(
        self, slide_ids: List[str], top_n: int = 300, save_dir: Optional[str] = None,
        target: str = "norm",
    ) -> dict:
        """Build the per-slide regression target for the selected HVGs.

        target="norm" (default): log1p(CP10k) — the normalized, log-transformed
            expression (self.combined.X after fit's normalize_total + log1p).
        target="raw": log1p(raw counts) — retains per-spot sequencing depth /
            tissue-density signal. Empirically ~2x higher per-gene PCC because
            images predict that signal well (but it is less depth-corrected).
        HVG *selection* is unchanged (always on normalized data); only the target
        column values differ.
        """
        if self.combined is None or self.hvg_names is None:
            raise RuntimeError("Call fit() first.")

        hvg_subset = self.hvg_names[:top_n]
        gene_idx = [self.combined.var_names.get_loc(g) for g in hvg_subset]

        if target == "raw":
            X = self.combined.layers["counts"]
            X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
            X = np.log1p(X)
        elif target == "norm":
            X = self.combined.X
            if hasattr(X, "toarray"):
                X = X.toarray()
        else:
            raise ValueError(f"target must be 'norm' or 'raw', got {target!r}")

        results = {}
        for sid in slide_ids:
            mask = (self.combined.obs["_slide_id"] == sid).values
            expr = X[mask][:, gene_idx].astype("float32")
            # Per-row spot barcodes (obs_names), so downstream code can join
            # expression to image patches by barcode rather than by position.
            barcodes = self.combined.obs_names[mask].to_numpy()
            results[sid] = {
                "gene_names": hvg_subset,
                "expression": expr,
                "barcodes": barcodes,
            }

            if save_dir:
                out_dir = Path(save_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                np.savez(
                    out_dir / f"{sid}.npz",
                    gene_names=hvg_subset,
                    expression=expr,
                    barcodes=barcodes,
                )

        return results