"""Experiment configuration management."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ExperimentConfig:
    """Configuration for a full AL experiment."""

    # Dataset
    dataset_name: str = "her2_breast"
    data_dir: str = "./hest_data"
    n_hvgs: int = 300
    patch_size: int = 224

    # Model
    model_name: str = "feature_predictor"
    pretrained: bool = True
    frozen_backbone: bool = False

    # Feature extraction
    feature_extractor: str = "densenet121"
    feature_dim: int = 1024  # densenet121 / DenseNet-style backbone output dim
    feature_cache_dir: str = "./feature_cache"

    # Training
    lr: float = 1e-4
    weight_decay: float = 1e-5
    epochs_per_round: int = 50
    full_epochs: int = 100
    batch_size: int = 64
    loss_fn: str = "mse"

    # Active learning
    al_methods: List[str] = field(default_factory=lambda: ["random", "badge", "coreset", "poisson_disk"])
    # Append the full-supervision upper bound automatically. Set False to skip it
    # when a ceiling has already been computed in a separate run (avoids retraining
    # a full-data model just to re-measure the shared upper bound).
    add_full_supervision: bool = True
    initial_budget: int = 100
    budget_per_round: int = 50
    num_rounds: int = 10
    # Budget-ratio sweep: fractions of the training pool at which each method's
    # AL trajectory is evaluated (an AL curve). 1.0 = full supervision, reported
    # as the shared upper bound. Replaces the former unused `mask_ratios`.
    budget_ratios: List[float] = field(
        default_factory=lambda: [0.05, 0.10, 0.20, 0.30, 0.50, 1.00]
    )

    # Evaluation
    n_folds: int = 4
    n_seeds: int = 3
    metric: str = "pcc_per_gene"
    # Cross-validation grouping: "patient" (strict, no patient in both splits —
    # the honest default) or "slide" (group by slide; a patient's other slides
    # may leak into train — matches much of the literature, useful as a
    # comparability / distribution-shift ablation).
    split_level: str = "patient"
    # Dispersion-ranked gene cutoffs for the top-N per-gene PCC sweep.
    pcc_top_n: List[int] = field(default_factory=lambda: [10, 50, 100, 200, 300])

    # End-to-end image-input mode: train a backbone (e.g. st_net) over raw
    # 224x224 patches instead of frozen cached features. Reads an index cache
    # (<feature_cache_dir>/<dataset>.npz with patch_pos + sample_ids) and pulls
    # patches from `patches_dir`. See scripts/prepare_hest_index.py.
    image_mode: bool = False
    patches_dir: str = "./hest_data/patches"
    # When set, full-supervision fold models are checkpointed here as
    # <dir>/<dataset>/full_supervision_fold{f}_seed{s}.pt (weights + provenance).
    save_models_dir: Optional[str] = None

    # System
    device: str = "cuda"
    num_workers: int = 0  # 0 avoids per-round worker-spawn overhead; raise for large image datasets
    seed: int = 42
    output_dir: str = "./results"

    def to_dict(self) -> Dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}
