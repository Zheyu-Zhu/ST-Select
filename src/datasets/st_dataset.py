"""PyTorch datasets for ST data."""

from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class STDataset(Dataset):
    """
    Dataset for ST gene expression prediction.
    Supports both pre-dumped patches and on-the-fly cropping.
    """

    def __init__(
        self,
        records: List[Dict],
        slide_images: Optional[Dict[str, Image.Image]] = None,
        transform: Optional[Callable] = None,
        patch_size: int = 224,
    ):
        self.records = records
        self.slide_images = slide_images
        self.transform = transform
        self.patch_size = patch_size
        self.half = patch_size // 2

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]

        # Load or crop patch
        if "patch_path" in record and Path(record["patch_path"]).exists():
            image = Image.open(record["patch_path"]).convert("RGB")
        elif self.slide_images is not None:
            slide = self.slide_images[record["sample_id"]]
            x, y = record["x"], record["y"]
            image = slide.crop((
                x - self.half, y - self.half,
                x + self.half, y + self.half,
            ))
        else:
            # Return a blank patch as fallback
            image = Image.new("RGB", (self.patch_size, self.patch_size), (255, 255, 255))

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0

        expression = torch.tensor(record["expression"], dtype=torch.float32)

        return {
            "image": image,
            "expression": expression,
            "sample_id": record["sample_id"],
            "spot_id": record.get("spot_id", f"spot_{idx}"),
            "x": record["x"],
            "y": record["y"],
        }


class STFeatureDataset(Dataset):
    """
    Dataset using pre-extracted features (no image loading needed).
    Much faster for AL experiments where features are cached.
    """

    def __init__(
        self,
        features: np.ndarray,
        expressions: np.ndarray,
        positions: Optional[np.ndarray] = None,
        metadata: Optional[List[Dict]] = None,
    ):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.expressions = torch.tensor(expressions, dtype=torch.float32)
        self.positions = positions
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "image": self.features[idx],  # "image" key for compatibility with trainer
            "expression": self.expressions[idx],
        }
        if self.positions is not None:
            item["position"] = torch.tensor(self.positions[idx], dtype=torch.float32)
        if self.metadata is not None:
            item["metadata"] = self.metadata[idx]
        return item
