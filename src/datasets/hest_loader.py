"""HEST-1k dataset loader and downloader."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scanpy as sc


# HEST-1k dataset tags for the five paper datasets (+ two 10x extras).
# Aliases (he_breast, crc_hd) point at the same tag so either name resolves.
HEST_TAGS = {
    "her2_breast": "andersson2021spatial",   # HER2+ breast (Andersson 2021)
    "he_breast_10x": "he2020integrating",     # He/Breast-ST (He 2020)
    "he_breast": "he2020integrating",
    "cscc": "ji2020multimodal",               # skin SCC (Ji 2020)
    "dlpfc": "maynard2021transcriptome",      # brain DLPFC (Maynard 2021)
    "crc_hd": "oliveira2025crc",              # colorectal HD (Oliveira 2025, Visium HD)
    "crc_hd_8um": "oliveira2025crc",
    "crc_hd_16um": "oliveira2025crc",
    "10x_breast": "10xgenomics_breast",
    "10x_mouse_brain": "10xgenomics_mouse_brain",
}


class HESTLoader:
    """Load and manage HEST-1k datasets."""

    def __init__(self, data_dir: str = "./hest_data"):
        self.data_dir = Path(data_dir)

    def download(self, dataset_name: str, tag: Optional[str] = None) -> None:
        """Download a HEST dataset from Hugging Face."""
        if tag is None:
            tag = HEST_TAGS.get(dataset_name, dataset_name)

        import subprocess
        subprocess.run([
            "huggingface-cli", "download", "MahmoodLab/hest",
            "--include", f"{tag}*",
            "--local-dir", str(self.data_dir),
        ], check=True)

    def list_slides(self, dataset_name: str) -> List[str]:
        """List all slide IDs for a dataset."""
        tag = HEST_TAGS.get(dataset_name, dataset_name)
        pattern = f"{tag}*"
        h5ad_files = list(self.data_dir.glob(f"**/{pattern}/*.h5ad")) + \
                     list(self.data_dir.glob(f"**/*{tag}*.h5ad"))
        return [f.stem for f in h5ad_files]

    def load_slide(self, slide_id: str) -> Tuple[str, str]:
        """Load paths for a single slide (h5ad, tif)."""
        h5ad_files = list(self.data_dir.rglob(f"*{slide_id}*.h5ad"))
        tif_files = list(self.data_dir.rglob(f"*{slide_id}*.tif"))

        if not h5ad_files:
            raise FileNotFoundError(f"No h5ad file found for slide {slide_id}")
        if not tif_files:
            raise FileNotFoundError(f"No TIF file found for slide {slide_id}")

        return str(h5ad_files[0]), str(tif_files[0])

    def load_all_h5ads(self, dataset_name: str) -> Dict[str, str]:
        """Get all h5ad paths for a dataset."""
        tag = HEST_TAGS.get(dataset_name, dataset_name)
        h5ad_files = list(self.data_dir.rglob(f"*{tag}*.h5ad"))
        return {f.stem: str(f) for f in h5ad_files}

    def get_patient_mapping(self, dataset_name: str) -> Dict[str, str]:
        """
        Get slide -> patient mapping for fold splitting.
        Reads from h5ad obs metadata where available.
        """
        h5ad_paths = self.load_all_h5ads(dataset_name)
        mapping = {}

        for slide_id, path in h5ad_paths.items():
            adata = sc.read_h5ad(path)
            if "patient_id" in adata.obs.columns:
                patient = adata.obs["patient_id"].iloc[0]
            elif "donor" in adata.obs.columns:
                patient = adata.obs["donor"].iloc[0]
            else:
                patient = slide_id.split("_")[0]
            mapping[slide_id] = patient

        return mapping
