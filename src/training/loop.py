"""Main active learning training loop."""

import copy
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ..al_methods.base import ActiveLearningStrategy, select_next_batch
from ..evaluation.metrics import compute_metrics
from .trainer import ALTrainer


class ActiveLearningLoop:
    """
    Complete AL experiment loop:
    1. Initialize labeled pool
    2. For each round: train model, acquire new samples, add to labeled pool
    3. Evaluate at each round
    """

    def __init__(
        self,
        trainer: ALTrainer,
        strategy: ActiveLearningStrategy,
        dataset,
        test_dataset,
        initial_budget: int = 100,
        budget_per_round: int = 50,
        num_rounds: int = 10,
        seed: int = 42,
        feature_cache: Optional[np.ndarray] = None,
        position_cache: Optional[np.ndarray] = None,
        budget_targets: Optional[List[int]] = None,
        top_n_cutoffs: Optional[List[int]] = None,
        dynamic_features: bool = False,
        test_group_ids: Optional[np.ndarray] = None,
    ):
        self.trainer = trainer
        self.strategy = strategy
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.initial_budget = initial_budget
        self.budget_per_round = budget_per_round
        self.num_rounds = num_rounds
        self.seed = seed
        self.feature_cache = feature_cache
        self.position_cache = position_cache
        self.top_n_cutoffs = top_n_cutoffs
        # When True, feature-based AL strategies score on the CURRENT model's
        # backbone output (recomputed each round from feature_cache) rather than
        # the static cached features. This is the "trainable backbone" regime:
        # the representation moves as the model trains, so AL sees a live signal.
        self.dynamic_features = dynamic_features
        self.test_group_ids = test_group_ids

        self.rng = np.random.default_rng(seed)
        self.all_indices = list(range(len(dataset)))
        n_total = len(self.all_indices)

        # Budget-target mode: evaluate the AL trajectory at a set of absolute
        # labeled-count checkpoints (derived from budget ratios upstream). When
        # not provided, fall back to the legacy round-based schedule expressed
        # as targets so a single code path serves both.
        if budget_targets is not None:
            targets = sorted({min(max(int(t), 1), n_total) for t in budget_targets})
        else:
            targets = []
            cum = min(self.initial_budget, n_total)
            for r in range(self.num_rounds):
                targets.append(min(cum, n_total))
                cum += self.budget_per_round
            targets = sorted(set(targets))
        self.budget_targets = targets

    def run(self) -> Dict[str, List]:
        """Execute the AL trajectory, evaluating at each budget target.

        The labeled pool is grown by acquisition until it reaches each target in
        ``self.budget_targets``; the model is (re)trained and evaluated at every
        target, yielding a budget/AL curve. ``frac`` records the labeled fraction
        of the training pool at each checkpoint (the budget ratios).
        """
        n_total = len(self.all_indices)
        results = {
            "round": [],
            "n_labeled": [],
            "frac": [],
            "pcc_per_gene": [],
            "pcc_per_spot": [],
            "mse": [],
            "mae": [],
        }
        if self.top_n_cutoffs:
            results["pcc_topN"] = []  # list of {cutoff: pcc} per checkpoint
        if self.test_group_ids is not None:
            results["pcc_per_gene_by_slide"] = []

        # Start from a random seed pool sized to the first target so model-based
        # scorers have something to train on before their first acquisition.
        selected = self._init_pool(self.budget_targets[0])

        # Weights of the model as trained at the *previous* checkpoint, used by
        # temporal-discrepancy strategies (TOD). None until a prior checkpoint
        # exists.
        prev_model_state = None

        for r, target in enumerate(self.budget_targets):
            # Grow the labeled pool up to `target` via acquisition.
            while len(selected) < target:
                candidates = list(set(self.all_indices) - set(selected))
                if not candidates:
                    break
                features = self.feature_cache
                # In the trainable-backbone regime, feature-ONLY strategies
                # (coreset, typiclust, poisson, ...) must score on the current
                # model's representation, so we project the cache through the
                # model's backbone. Strategies that also run the model
                # (BADGE/entropy/TOD, requires_model=True) apply the backbone
                # themselves from the raw cache, so they get live features for
                # free — projecting here would double-apply the backbone.
                if (self.dynamic_features and features is not None
                        and self.strategy.requires_features
                        and not self.strategy.requires_model):
                    features = self._project_features(features)
                elif features is None and self.strategy.requires_features:
                    features = self._extract_features()

                prev_model = None
                if prev_model_state is not None:
                    prev_model = copy.deepcopy(self.trainer.model)
                    prev_model.load_state_dict(prev_model_state)

                k = min(self.budget_per_round, target - len(selected), len(candidates))
                new_picks = select_next_batch(
                    strategy=self.strategy,
                    candidate_indices=candidates,
                    selected_indices=selected,
                    k=k,
                    features=features,
                    positions=self.position_cache,
                    model=self.trainer.model,
                    extras={"prev_model": prev_model},
                )
                assert not (set(new_picks) & set(selected)), (
                    "AL strategy returned already-labeled indices"
                )
                selected.extend(new_picks)

            # Train + evaluate at this budget checkpoint.
            self.trainer.model.apply(self._reset_weights_partial)
            self.trainer.train(self.dataset, selected)
            metrics = self._evaluate()

            results["round"].append(r)
            results["n_labeled"].append(len(selected))
            results["frac"].append(round(len(selected) / n_total, 4))
            results["pcc_per_gene"].append(metrics["pcc_per_gene"])
            results["pcc_per_spot"].append(metrics["pcc_per_spot"])
            results["mse"].append(metrics["mse"])
            results["mae"].append(metrics["mae"])
            if self.top_n_cutoffs:
                results["pcc_topN"].append(metrics.get("pcc_topN", {}))
            if self.test_group_ids is not None:
                results["pcc_per_gene_by_slide"].append(
                    metrics.get("pcc_per_gene_by_slide", float("nan"))
                )

            # Snapshot weights as the temporal-discrepancy reference for next time.
            prev_model_state = {
                k: v.detach().cpu().clone()
                for k, v in self.trainer.model.state_dict().items()
            }

            if len(selected) >= n_total:
                break

        return results

    def _init_pool(self, size: Optional[int] = None) -> List[int]:
        """Initialize the labeled pool randomly."""
        if size is None:
            size = self.initial_budget
        size = min(size, len(self.all_indices))
        return self.rng.choice(
            self.all_indices, size=size, replace=False
        ).tolist()

    def _evaluate(self) -> Dict[str, float]:
        """Evaluate model on test set."""
        test_loader = DataLoader(
            self.test_dataset, batch_size=64, shuffle=False,
            num_workers=getattr(self.trainer, "num_workers", 0),
        )

        all_preds = []
        all_targets = []

        self.trainer.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(self.trainer.device)
                preds = self.trainer.model(images)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(batch["expression"].numpy())

        predictions = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        metrics = compute_metrics(predictions, targets)
        if self.top_n_cutoffs:
            from ..evaluation.metrics import top_n_pcc_sweep
            metrics["pcc_topN"] = top_n_pcc_sweep(
                predictions, targets, self.top_n_cutoffs
            )
        if self.test_group_ids is not None:
            from ..evaluation.metrics import pcc_per_gene_grouped
            metrics["pcc_per_gene_by_slide"] = pcc_per_gene_grouped(
                predictions, targets, self.test_group_ids
            )
        return metrics

    def _project_features(self, feats: np.ndarray) -> np.ndarray:
        """Push cached input features through the current model's backbone.

        For the trainable-projection model this returns the round's *moving*
        representation (what AL should score on when the backbone is learnable).
        Falls back to the input features if the model has no usable backbone.
        """
        model = self.trainer.model
        extractor = getattr(model, "get_features", None)
        if extractor is None:
            return feats
        model.eval()
        out = []
        bs = 512
        with torch.no_grad():
            for i in range(0, len(feats), bs):
                x = torch.tensor(feats[i:i + bs], dtype=torch.float32).to(self.trainer.device)
                out.append(extractor(x).cpu().numpy())
        return np.concatenate(out, axis=0)

    def _extract_features(self) -> np.ndarray:
        """Extract features from the full dataset using the current model."""
        full_loader = DataLoader(
            self.dataset, batch_size=64, shuffle=False,
            num_workers=getattr(self.trainer, "num_workers", 0),
        )
        return self.trainer.get_features(full_loader)

    @staticmethod
    def _reset_weights_partial(m):
        """Reset only the head weights, keep backbone pretrained."""
        if isinstance(m, (torch.nn.Linear,)) and m.out_features < 1024:
            m.reset_parameters()
