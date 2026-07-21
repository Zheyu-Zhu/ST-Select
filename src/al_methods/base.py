"""Base class and unified API for all active learning strategies."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
import torch


def model_device(model, fallback: str = "cpu") -> str:
    """The device a model's parameters actually live on.

    AL strategies must run the scoring model on the model's own device, not the
    strategy's configured `self.device` — the two can differ (e.g. the loop
    keeps the model on MPS while a strategy defaulted elsewhere), which would
    raise a cpu-vs-mps tensor mismatch.
    """
    try:
        return str(next(model.parameters()).device)
    except (StopIteration, AttributeError):
        return fallback


class ActiveLearningStrategy(ABC):
    """Base class for all AL acquisition strategies."""

    name: str = "base"
    family: str = "base"
    requires_features: bool = False
    requires_model: bool = False
    requires_positions: bool = False

    @abstractmethod
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
        """Select k indices from candidate_indices to label next."""
        ...

    def validate_inputs(
        self,
        candidate_indices: List[int],
        k: int,
        features: Optional[np.ndarray],
        positions: Optional[np.ndarray],
        model: Optional[torch.nn.Module],
    ) -> None:
        if k > len(candidate_indices):
            raise ValueError(
                f"Budget k={k} exceeds candidate pool size {len(candidate_indices)}"
            )
        if self.requires_features and features is None:
            raise ValueError(f"{self.name} requires features but none provided.")
        if self.requires_model and model is None:
            raise ValueError(f"{self.name} requires a model but none provided.")
        if self.requires_positions and positions is None:
            raise ValueError(f"{self.name} requires positions but none provided.")


def select_next_batch(
    strategy: ActiveLearningStrategy,
    candidate_indices: List[int],
    selected_indices: List[int],
    k: int,
    features: Optional[np.ndarray] = None,
    positions: Optional[np.ndarray] = None,
    model: Optional[torch.nn.Module] = None,
    dataloader: Optional[torch.utils.data.DataLoader] = None,
    extras: Optional[Dict] = None,
) -> List[int]:
    """Unified entry point for batch acquisition.

    Enforces the acquisition post-conditions every strategy must satisfy so a
    misbehaving strategy cannot silently corrupt the labeled-budget accounting:
      * every returned index is a member of ``candidate_indices``;
      * the returned batch is exactly ``k`` distinct indices.
    Some strategies can under-fill or duplicate under degenerate inputs (e.g.
    TypiClust with empty KMeans clusters, or greedy coverage when distances
    collapse). Rather than let the labeled set drift below budget, we dedupe,
    drop out-of-pool picks, and deterministically top up from the remaining
    candidates (seeded by the strategy's ``seed`` when available).
    """
    strategy.validate_inputs(candidate_indices, k, features, positions, model)
    raw = strategy.select(
        candidate_indices=candidate_indices,
        selected_indices=selected_indices,
        k=k,
        features=features,
        positions=positions,
        model=model,
        dataloader=dataloader,
        extras=extras,
    )

    candidate_set = set(candidate_indices)

    # Keep first occurrence, preserving order; drop anything outside the pool.
    deduped: List[int] = []
    seen = set()
    for idx in raw:
        if idx in candidate_set and idx not in seen:
            deduped.append(idx)
            seen.add(idx)

    # Top up to k from the remaining candidates if the strategy under-filled.
    if len(deduped) < k:
        remaining = [c for c in candidate_indices if c not in seen]
        seed = getattr(strategy, "seed", 42)
        rng = np.random.default_rng(seed)
        rng.shuffle(remaining)
        deduped.extend(remaining[: k - len(deduped)])

    result = deduped[:k]

    assert len(result) == k, (
        f"{strategy.name} could not fill budget k={k} "
        f"(got {len(result)} from {len(candidate_indices)} candidates)"
    )
    assert len(set(result)) == k, f"{strategy.name} returned duplicate indices"
    assert set(result) <= candidate_set, (
        f"{strategy.name} returned indices outside the candidate pool"
    )
    return result