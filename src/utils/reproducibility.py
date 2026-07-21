"""Reproducibility utilities."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _mps_ok() -> bool:
    return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()


def resolve_device(device: str = "auto") -> str:
    """Map a requested device string to the best device actually available.

    Used everywhere in place of the old `device if torch.cuda.is_available()
    else "cpu"` guard, which silently forced CPU on Apple Silicon (MPS). Honors
    an explicit request when possible; "auto" prefers cuda -> mps -> cpu.
    """
    if device in (None, "auto"):
        if torch.cuda.is_available():
            return "cuda"
        if _mps_ok():
            return "mps"
        return "cpu"
    if device == "cuda":
        return "cuda" if torch.cuda.is_available() else ("mps" if _mps_ok() else "cpu")
    if device == "mps":
        return "mps" if _mps_ok() else "cpu"
    return device  # "cpu" or an explicit override


def get_device(prefer: str = "auto") -> str:
    """Get the best available device (alias of resolve_device)."""
    return resolve_device(prefer)
