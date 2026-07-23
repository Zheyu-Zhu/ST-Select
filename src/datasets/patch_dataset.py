"""Image-patch dataset for end-to-end backbone training.

Unlike STFeatureDataset (which serves pre-extracted frozen features), this
dataset serves the raw 224x224 H&E patch images so a trainable backbone
(e.g. st_net / DenseNet-121) can be fine-tuned end-to-end over pixels.

Patches come from HEST's pre-extracted `patches/<slide>.h5` files (array `img`,
shape (N,224,224,3) uint8). Each cache row carries the slide id and the row
index into that slide's patch array (`patch_pos`), so a patch is fetched
lazily on __getitem__. Slide patch arrays are memoized in a module-level cache
so the h5 files are read once, not once per (fold, seed, epoch).

Images are ImageNet-normalized to match the DenseNet-121 pretrained weights
(and the normalization used when the frozen feature cache was built).
"""

import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# LRU cache of {slide_id: uint8 (N,224,224,3)} so each patch h5 is read once and
# shared across datasets built during a benchmark — but BOUNDED in bytes so a
# large dataset (e.g. He-breast, ~9.5 GB of patches over 68 slides) cannot
# accumulate every slide in RAM and trigger swapping. A single CV fold's train
# slides (~3-4 GB for He-breast) fit under the cap, so there is no intra-fold
# thrashing; cross-fold growth evicts least-recently-used slides.
_SLIDE_IMG_CACHE: "OrderedDict[str, np.ndarray]" = OrderedDict()
_CACHE_BYTES = 0
# Override with PATCH_CACHE_GB; default 5 GB leaves headroom on a 16 GB machine
# alongside torch/CUDA host allocations.
_CACHE_CAP_BYTES = int(float(os.environ.get("PATCH_CACHE_GB", "5")) * 1024 ** 3)


def _get_slide_imgs(slide_id: str, patches_dir: Path) -> np.ndarray:
    global _CACHE_BYTES
    arr = _SLIDE_IMG_CACHE.get(slide_id)
    if arr is not None:
        _SLIDE_IMG_CACHE.move_to_end(slide_id)  # mark most-recently-used
        return arr
    with h5py.File(patches_dir / f"{slide_id}.h5", "r") as f:
        arr = f["img"][:]  # (N,224,224,3) uint8
    _SLIDE_IMG_CACHE[slide_id] = arr
    _CACHE_BYTES += arr.nbytes
    # Evict least-recently-used slides until under the cap (keep >=1 resident).
    while _CACHE_BYTES > _CACHE_CAP_BYTES and len(_SLIDE_IMG_CACHE) > 1:
        _, old = _SLIDE_IMG_CACHE.popitem(last=False)
        _CACHE_BYTES -= old.nbytes
    return arr


def clear_slide_cache() -> None:
    global _CACHE_BYTES
    _SLIDE_IMG_CACHE.clear()
    _CACHE_BYTES = 0


class PatchImageDataset(Dataset):
    """Serves ImageNet-normalized (3,224,224) patches + HVG expression vectors.

    Parameters mirror the sub-array layout run_experiment already uses for
    STFeatureDataset: pass the per-row slices for the split you want.
    """

    def __init__(
        self,
        sample_ids: np.ndarray,
        patch_pos: np.ndarray,
        expressions: np.ndarray,
        patches_dir: str,
        positions: Optional[np.ndarray] = None,
    ):
        self.sample_ids = np.asarray(sample_ids)
        self.patch_pos = np.asarray(patch_pos).astype(int)
        self.expressions = torch.tensor(np.asarray(expressions), dtype=torch.float32)
        self.patches_dir = Path(patches_dir)
        self.positions = positions

    def __len__(self) -> int:
        return len(self.patch_pos)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        slide = str(self.sample_ids[idx])
        imgs = _get_slide_imgs(slide, self.patches_dir)
        patch = imgs[self.patch_pos[idx]]  # (224,224,3) uint8
        img = torch.from_numpy(np.ascontiguousarray(patch)).float().permute(2, 0, 1) / 255.0
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD

        item = {"image": img, "expression": self.expressions[idx]}
        if self.positions is not None:
            item["position"] = torch.tensor(self.positions[idx], dtype=torch.float32)
        return item
