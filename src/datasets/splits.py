"""Data splitting strategies for ST experiments."""

from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import KFold


class PatientKFold:
    """
    Patient-level k-fold cross-validation.
    All slides of one patient stay in the same fold.
    """

    def __init__(self, n_folds: int = 4, seed: int = 42):
        self.n_folds = n_folds
        self.seed = seed

    def split(
        self,
        records: List[Dict],
        patient_key: str = "patient_id",
    ) -> List[Tuple[List[int], List[int]]]:
        """
        Returns list of (train_indices, test_indices) tuples.
        """
        # Group records by patient
        patient_to_indices = {}
        for i, record in enumerate(records):
            patient = record.get(patient_key, record.get("sample_id", str(i)))
            patient_to_indices.setdefault(patient, []).append(i)

        patients = sorted(patient_to_indices.keys())
        if self.n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {self.n_folds}.")
        if self.n_folds > len(patients):
            raise ValueError(
                f"n_folds={self.n_folds} exceeds the number of distinct patients "
                f"({len(patients)}: {patients}). Patient-level CV needs at least "
                f"one patient per fold — reduce n_folds or add more patients/slides."
            )
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

        folds = []
        for train_patients, test_patients in kf.split(patients):
            train_indices = []
            test_indices = []
            for pi in train_patients:
                train_indices.extend(patient_to_indices[patients[pi]])
            for pi in test_patients:
                test_indices.extend(patient_to_indices[patients[pi]])
            folds.append((train_indices, test_indices))

        return folds


class SlideHoldout:
    """
    Slide-level holdout: specific slides designated for train/test.
    Common pattern: middle slice for training, outer slices for testing.
    """

    def __init__(self):
        pass

    def split(
        self,
        records: List[Dict],
        train_slides: List[str],
        test_slides: List[str],
    ) -> Tuple[List[int], List[int]]:
        """Split by explicit slide assignment."""
        train_set = set(train_slides)
        test_set = set(test_slides)

        train_indices = [
            i for i, r in enumerate(records)
            if r.get("sample_id") in train_set
        ]
        test_indices = [
            i for i, r in enumerate(records)
            if r.get("sample_id") in test_set
        ]

        return train_indices, test_indices

    @staticmethod
    def split_by_ratio(
        records: List[Dict],
        train_ratio: float = 0.8,
        seed: int = 42,
    ) -> Tuple[List[int], List[int]]:
        """Random slide-level split."""
        rng = np.random.default_rng(seed)

        slides = list(set(r.get("sample_id", str(i)) for i, r in enumerate(records)))
        rng.shuffle(slides)

        n_train = int(len(slides) * train_ratio)
        train_slides = set(slides[:n_train])

        train_indices = [
            i for i, r in enumerate(records)
            if r.get("sample_id") in train_slides
        ]
        test_indices = [
            i for i, r in enumerate(records)
            if r.get("sample_id") not in train_slides
        ]

        return train_indices, test_indices


class BudgetMasker:
    """Apply mask ratios to simulate different annotation budgets."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def mask(self, indices: List[int], ratio: float) -> List[int]:
        """Return a subset of indices at the given ratio."""
        k = max(1, int(len(indices) * ratio))
        return self.rng.choice(indices, size=k, replace=False).tolist()

    def mask_sweep(
        self, indices: List[int], ratios: List[float] = None
    ) -> Dict[float, List[int]]:
        """Generate masked sets for multiple budget ratios."""
        if ratios is None:
            ratios = [0.05, 0.10, 0.15, 0.30, 0.50, 0.75]
        return {r: self.mask(indices, r) for r in ratios}
