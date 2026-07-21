"""On-disk data format for the ST AL pipeline."""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class SpotDataFormatter:
    """Create and load the per-spot dict format expected by all downstream code."""

    @staticmethod
    def create_spot_records(
        slide_id: str,
        coords: np.ndarray,
        expression: np.ndarray,
        patient_id: Optional[str] = None,
        fold: Optional[int] = None,
        patch_dir: Optional[str] = None,
    ) -> List[Dict]:
        records = []
        for i in range(len(coords)):
            record = {
                "sample_id": slide_id,
                "spot_id": f"spot_{i}",
                "x": int(coords[i, 0]),
                "y": int(coords[i, 1]),
                "expression": expression[i].astype("float32"),
            }
            if patient_id is not None:
                record["patient_id"] = patient_id
            if fold is not None:
                record["fold"] = fold
            if patch_dir is not None:
                record["patch_path"] = f"{patch_dir}/{slide_id}/spot_{i}.png"
            records.append(record)
        return records

    @staticmethod
    def save(records: List[Dict], save_path: str) -> None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_path, data=records)

    @staticmethod
    def load(path: str) -> List[Dict]:
        loaded = np.load(path, allow_pickle=True)
        return loaded["data"].tolist()

    @staticmethod
    def merge_slides(slide_records: List[List[Dict]], assign_global_ids: bool = True) -> List[Dict]:
        merged = []
        global_idx = 0
        for records in slide_records:
            for record in records:
                if assign_global_ids:
                    record["global_idx"] = global_idx
                    global_idx += 1
                merged.append(record)
        return merged