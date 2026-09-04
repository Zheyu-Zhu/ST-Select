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


class WithinSlideSplit:
    """Within-slide k-fold: train and test spots come from the SAME slides.

    This is the acquisition question the cost argument actually rests on: given a
    slide you are going to profile, can you measure a fraction of its spots and
    predict the rest? Every other split in this codebase (PatientKFold) answers a
    different question - can a model trained on some patients generalise to a NEW
    patient - and the two are not comparable. Numbers here are far higher, because
    the model has already seen the slide's morphology, staining and batch effects
    and only has to fill in gaps.

    Why a fixed reserved test set rather than "predict whatever you did not
    select": with the complement as the test set, every budget is scored on a
    different set of spots (5% budget -> predict 95%, 50% -> predict 50%), so
    budgets cannot be compared, and at 100% budget the test set is empty so there
    is no full-supervision reference to retain a fraction *of*. Reserving a fixed
    slice of each slide keeps the test set identical at every budget.

    Two holdout geometries, because they answer different questions:

      mode="random"   test spots are interleaved with selectable ones. Matches
                      deployment (you skip individual spots on a dense grid), but
                      every test spot has immediate labeled neighbours, so strong
                      spatial autocorrelation alone can score well. Always pair
                      with the nearest-neighbour control baseline.
      mode="spatial"  test spots form contiguous regions (a coordinate band per
                      slide). Harder and more honest about whether morphology
                      predicts expression, rather than neighbouring spots doing it.
    """

    def __init__(self, n_folds: int = 4, seed: int = 42, mode: str = "random"):
        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {n_folds}.")
        if mode not in ("random", "spatial"):
            raise ValueError(f"mode must be 'random' or 'spatial', got {mode!r}.")
        self.n_folds = n_folds
        self.seed = seed
        self.mode = mode

    def split(
        self,
        records: List[Dict],
        patient_key: str = "sample_id",
    ) -> List[Tuple[List[int], List[int]]]:
        """Returns n_folds (pool_indices, test_indices) pairs.

        Both sides span every slide. Each slide contributes 1/n_folds of its spots
        to the test side of each fold, so across folds every spot is tested once.
        """
        by_slide: Dict[str, List[int]] = {}
        for i, r in enumerate(records):
            by_slide.setdefault(str(r.get(patient_key, i)), []).append(i)

        rng = np.random.default_rng(self.seed)
        # chunks[slide] = list of n_folds index-arrays
        chunks: Dict[str, List[np.ndarray]] = {}
        for slide, idx in by_slide.items():
            idx_arr = np.asarray(idx)
            if self.mode == "spatial":
                # Order by a random 2-D direction, then cut into contiguous bands.
                # A random direction (rather than always x) stops the split from
                # aligning with tissue structure the same way on every slide.
                pos = np.asarray([records[i].get("position", (0.0, 0.0)) for i in idx_arr],
                                 dtype=float)
                theta = rng.uniform(0, np.pi)
                order = np.argsort(pos @ np.array([np.cos(theta), np.sin(theta)]))
            else:
                order = rng.permutation(len(idx_arr))
            chunks[slide] = [c for c in np.array_split(idx_arr[order], self.n_folds)]

        folds = []
        for k in range(self.n_folds):
            test, pool = [], []
            for slide, cs in chunks.items():
                for j, c in enumerate(cs):
                    (test if j == k else pool).extend(c.tolist())
            folds.append((sorted(pool), sorted(test)))
        return folds
