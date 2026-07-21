"""Image cropping utilities for extracting H&E patches from WSIs."""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import PIL.Image

PIL.Image.MAX_IMAGE_PIXELS = None


class TIFConverter:
    """Convert multi-frame WSI TIF files to single-frame images."""

    @staticmethod
    def tif_to_png(tif_path: str, out_path: str, level: int = 0) -> None:
        img = PIL.Image.open(tif_path)
        if hasattr(img, "n_frames") and img.n_frames > 1:
            img.seek(level)
        img.convert("RGB").save(out_path)

    @staticmethod
    def read_level0_openslide(tif_path: str):
        import openslide

        return openslide.OpenSlide(tif_path)


class PatchCropper:
    """Crop per-spot patches from whole slide images."""

    def __init__(self, patch_size: int = 224, backend: str = "pil"):
        self.patch_size = patch_size
        self.half = patch_size // 2
        self.backend = backend

    def crop_from_pil(
        self, image: PIL.Image.Image, x: int, y: int
    ) -> PIL.Image.Image:
        w, h = image.size
        x0 = max(0, x - self.half)
        y0 = max(0, y - self.half)
        x1 = min(w, x + self.half)
        y1 = min(h, y + self.half)

        patch = image.crop((x0, y0, x1, y1))

        if patch.size != (self.patch_size, self.patch_size):
            padded = PIL.Image.new("RGB", (self.patch_size, self.patch_size), (255, 255, 255))
            paste_x = self.half - (x - x0)
            paste_y = self.half - (y - y0)
            padded.paste(patch, (paste_x, paste_y))
            return padded
        return patch

    def crop_from_openslide(self, slide, x: int, y: int) -> PIL.Image.Image:
        region = slide.read_region(
            (x - self.half, y - self.half), 0, (self.patch_size, self.patch_size)
        )
        return region.convert("RGB")

    def crop_all_spots(
        self,
        image_source,
        spot_coords: np.ndarray,
        save_dir: Optional[str] = None,
        slide_id: str = "slide",
    ) -> List[PIL.Image.Image]:
        patches = []
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        for i, (x, y) in enumerate(spot_coords):
            x, y = int(x), int(y)
            if self.backend == "openslide":
                patch = self.crop_from_openslide(image_source, x, y)
            else:
                patch = self.crop_from_pil(image_source, x, y)

            if save_dir:
                patch.save(Path(save_dir) / f"{slide_id}_spot_{i}.png")
            patches.append(patch)

        return patches


class HESTPatcher:
    """Use HEST's built-in patching utilities."""

    def __init__(self, patch_size: int = 224):
        self.patch_size = patch_size

    def dump_patches(self, slide_dir: str, output_dir: str) -> None:
        from hest import HESTData

        hest = HESTData(slide_dir)
        hest.dump_patches(patch_save_dir=output_dir, target_patch_size=self.patch_size)