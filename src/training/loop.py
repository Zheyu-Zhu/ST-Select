"""Main active learning training loop."""

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ..al_methods.base import ActiveLearningStrategy, select_next_batch
from ..evaluation.metrics import compute_metrics
from .trainer import ALTrainer


def group_aware_val_split(
    selected: List[int],
    val_frac: float,
    seed: int,
    group_ids: Optional[np.ndarray] = None,
) -> Tuple[List[int], List[int]]:
    """Split a labeled pool into (train, early-stopping val).

    Returns ``(selected, [])`` when validation is off or the pool is too small to
    spare rows — the caller then trains for its full epoch budget, which is the
    legacy fixed-epoch behaviour.

    With patient ids the holdout is whole patients: the overfitting we early-stop
    on is patient-level (batch effects), and a random row split shares those
    effects with the training rows, so its loss keeps falling while test PCC
    degrades. Patients are added smallest-first until the holdout is closest to
    ``val_frac`` of the pool, and never so many that fewer than two patients are
    left to train on.

    Shared by the AL loop and the full-supervision ceiling so every point on a
    budget curve is produced by the same training recipe.
    """
    if val_frac <= 0 or len(selected) < 20:
        return list(selected), []

    sel = np.asarray(selected)
    rng = np.random.default_rng(seed)
    target = val_frac * len(sel)

    if group_ids is not None:
        groups = np.asarray(group_ids)[sel]
        uniq, counts = np.unique(groups, return_counts=True)
        if len(uniq) >= 3:
            # Shuffle first, then stable-sort by count: smallest patients first,
            # with count ties broken by seed so different seeds hold out
            # different patients instead of all picking the same one.
            order = np.argsort(rng.random(len(uniq)), kind="stable")
            order = order[np.argsort(counts[order], kind="stable")]
            held, n_held = [], 0
            for gi in order:
                if len(held) >= len(uniq) - 2:
                    break
                # Stop on whichever side of the target is closer rather than
                # always crossing it: with few spots per patient one extra
                # patient can overshoot 15% to 25%, and that lost training data
                # hurts most at the low budgets we care about.
                if held and abs(n_held + counts[gi] - target) > abs(n_held - target):
                    break
                held.append(uniq[gi])
                n_held += counts[gi]
                if n_held >= target:
                    break
            if held and n_held < len(sel):
                mask = np.isin(groups, held)
                return sel[~mask].tolist(), sel[mask].tolist()

    # No usable groups: fall back to a random row split. Early stopping is then
    # weaker (val shares the training patients' batch effects) but still catches
    # plain overfitting, and keeps budgets comparable across datasets.
    n_val = max(2, int(round(target)))
    perm = rng.permutation(len(sel))
    return sel[perm[n_val:]].tolist(), sel[perm[:n_val]].tolist()


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
        eval_ctx: Optional[Dict] = None,
        val_frac: float = 0.0,
        record_epoch_curve: bool = False,
        train_group_ids: Optional[np.ndarray] = None,
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
        # Test-fold positions / valid_mask for the downstream-task metrics.
        self.eval_ctx = eval_ctx
        # >0 holds back this fraction of the LABELED pool as a validation set and
        # early-stops on it. Without this, every budget trains for the same fixed
        # epoch count, so a larger pool also means more gradient steps — which
        # confounds "more labels" with "more training" and can make the budget
        # curve non-monotonic (more labels appearing to *hurt* via overfitting to
        # the training patients). With it, each budget trains to its own optimum,
        # which is what a budget-vs-performance curve has to compare. The test
        # fold is never touched.
        self.val_frac = val_frac
        # Diagnostic only: evaluate on TEST after every epoch so the cost of the
        # stopping rule can be measured. Never select an epoch from this curve.
        self.record_epoch_curve = record_epoch_curve
        # Patient ids for the TRAIN pool. When present the early-stopping holdout
        # is group-aware (whole patients held out), which matters: the failure
        # mode we are trying to stop on is patient-level overfitting (batch
        # effects), and a random in-pool split shares those effects with the
        # training rows, so its loss keeps falling while test PCC degrades.
        # Holding out patients makes val loss a proxy for the real quantity.
        self.train_group_ids = (
            np.asarray(train_group_ids) if train_group_ids is not None else None
        )

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

    def _val_split(self, selected: List[int]) -> Tuple[List[int], List[int]]:
        return group_aware_val_split(
            selected, self.val_frac, self.seed, self.train_group_ids
        )

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
            "epochs_run": [],
            "epoch_curve": [],
        }
        if self.top_n_cutoffs:
            results["pcc_topN"] = []  # list of {cutoff: pcc} per checkpoint
        if self.test_group_ids is not None:
            results["pcc_per_gene_by_slide"] = []
            results["ari"] = []
            results["nmi"] = []
            if self.top_n_cutoffs:
                results["pcc_topN_by_slide"] = []
            if self.eval_ctx:
                results["spatial"] = []
                if self.eval_ctx.get("valid_mask") is not None:
                    results["celltype"] = []

        # Start from a random seed pool sized to the first target so model-based
        # scorers have something to train on before their first acquisition.
        selected = self._init_pool(self.budget_targets[0])

        # A single budget target is a silent no-op for acquisition: the seed pool
        # above already fills it, the while-loop below never runs, and every
        # strategy reports the same random draw. That looks exactly like "no
        # strategy beats random" while testing nothing, so refuse it instead.
        if len(self.budget_targets) == 1 and self.strategy.name != "random":
            raise ValueError(
                f"Strategy '{self.strategy.name}' was given a single budget "
                f"target ({self.budget_targets[0]} spots), which the random seed "
                f"pool fills entirely — the strategy would never select anything "
                f"and the result would be indistinguishable from random. Add a "
                f"smaller starting ratio (e.g. budget_ratios=[0.05, 0.3]) so the "
                f"strategy acquires the difference."
            )

        # Weights of the model as trained at the *previous* checkpoint, used by
        # temporal-discrepancy strategies (TOD). None until a prior checkpoint
        # exists.
        prev_model_state = None

        needs_rep = self.strategy.requires_features or self.strategy.requires_model
        for r, target in enumerate(self.budget_targets):
            # Prepare the representation for THIS round's acquisitions once — the
            # model is fixed until we (re)train below, so features don't change
            # between the while-loop's acquisition steps.
            features = self.feature_cache
            # head_only: in image mode we score AL on the backbone features
            # extracted here, and strategies must apply only model.head (not the
            # backbone again) to turn them into predictions — the backbone
            # consumes images, and re-applying it to 1024-d features would crash
            # (double-backbone). See BADGE/entropy image-mode handling.
            head_only = False
            if features is None and needs_rep:
                # Image mode: extract backbone features once (model is fixed).
                features = self._extract_features()
                head_only = True
            elif (self.dynamic_features and features is not None
                    and self.strategy.requires_features
                    and not self.strategy.requires_model):
                # Trainable-projection proxy: score on the current representation.
                features = self._project_features(features)

            prev_model = None
            if prev_model_state is not None:
                prev_model = copy.deepcopy(self.trainer.model)
                prev_model.load_state_dict(prev_model_state)

            # Grow the labeled pool up to `target` via acquisition.
            while len(selected) < target:
                candidates = list(set(self.all_indices) - set(selected))
                if not candidates:
                    break

                k = min(self.budget_per_round, target - len(selected), len(candidates))
                new_picks = select_next_batch(
                    strategy=self.strategy,
                    candidate_indices=candidates,
                    selected_indices=selected,
                    k=k,
                    features=features,
                    positions=self.position_cache,
                    model=self.trainer.model,
                    dataloader=None,
                    extras={"prev_model": prev_model, "head_only": head_only},
                )
                assert not (set(new_picks) & set(selected)), (
                    "AL strategy returned already-labeled indices"
                )
                selected.extend(new_picks)

            # Train + evaluate at this budget checkpoint.
            #
            # Reseed first, deterministically per checkpoint. Without this the
            # head re-initialisation below draws from whatever RNG state the round
            # happened to leave behind, and that state depends on the STRATEGY:
            # scorers needing a representation run _extract_features() earlier in
            # the round, others do not. The result was that arms split into two
            # groups with measurably different initial models — on the UNI e2e
            # HER2 run the strategies that extract features started +0.0074
            # per-slide above the ones that don't, at the very first checkpoint
            # where every arm is training on the *same* random seed pool and the
            # numbers should be identical. Comparing strategies then means
            # comparing initialisations too. Seeding off (seed, round) keeps every
            # arm's checkpoint-r init identical while still varying across rounds.
            torch.manual_seed(self.seed * 1000 + r)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed * 1000 + r)
            self.trainer.model.apply(self._reset_weights_partial)
            probe = None
            if self.record_epoch_curve:
                def probe(epoch, _model):
                    return {"epoch": epoch, **self._evaluate_light()}
            trn_sel, val_sel = self._val_split(selected)
            if val_sel:
                hist = self.trainer.train(self.dataset, trn_sel,
                                          val_dataset=self.dataset, val_indices=val_sel,
                                          epoch_probe=probe)
            else:
                hist = self.trainer.train(self.dataset, selected, epoch_probe=probe)
            metrics = self._evaluate()

            results["round"].append(r)
            results["n_labeled"].append(len(selected))
            results["frac"].append(round(len(selected) / n_total, 4))
            # Epochs actually run at this budget. With early stopping this is the
            # direct evidence for (or against) the fixed-epoch confound: if more
            # labels really did buy more overfitting, small budgets should stop
            # much earlier than large ones. Equals the epoch cap when val is off.
            results["epochs_run"].append(len(hist.get("train_loss", [])))
            results["epoch_curve"].append(hist.get("probe", []))
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
                results["ari"].append(metrics.get("ari", float("nan")))
                results["nmi"].append(metrics.get("nmi", float("nan")))
                if self.top_n_cutoffs:
                    results["pcc_topN_by_slide"].append(
                        metrics.get("pcc_topN_by_slide", {})
                    )
                if "spatial" in results:
                    results["spatial"].append(metrics.get("spatial", {}))
                if "celltype" in results:
                    results["celltype"].append(metrics.get("celltype", {}))

            # Snapshot weights as the temporal-discrepancy reference for next time.
            prev_model_state = {
                k: v.detach().cpu().clone()
                for k, v in self.trainer.model.state_dict().items()
            }

            if len(selected) >= n_total:
                break

        # Diversity of the final selected set — does a strategy that spreads its
        # picks wider (in representation / physical space) predict better? Uses
        # the frozen feature cache when available, else the last round's extracted
        # backbone features (image mode). Cheap, computed once per run.
        try:
            from ..evaluation.metrics import selection_diversity
            div_feats = self.feature_cache if self.feature_cache is not None else features
            results["selection_diversity"] = selection_diversity(
                div_feats, selected, self.position_cache
            )
        except Exception:
            results["selection_diversity"] = {}

        return results

    def _init_pool(self, size: Optional[int] = None) -> List[int]:
        """Initialize the labeled pool randomly."""
        if size is None:
            size = self.initial_budget
        size = min(size, len(self.all_indices))
        return self.rng.choice(
            self.all_indices, size=size, replace=False
        ).tolist()

    def _evaluate_light(self) -> Dict[str, float]:
        """PCC only, for the per-epoch probe.

        The full _evaluate() also clusters every slide (ARI/NMI) and computes
        Moran's I over 300 genes per slide. Running that after every epoch would
        cost more than the training it is measuring, and none of it is what the
        epoch curve is for.
        """
        preds, targets = self._predict_test()
        from ..evaluation.metrics import pcc_per_gene, pcc_per_spot
        out = {"pcc_per_gene": float(np.nanmean(pcc_per_gene(preds, targets))),
               "pcc_per_spot": float(np.nanmean(pcc_per_spot(preds, targets))),
               "mse": float(np.mean((preds - targets) ** 2))}
        if self.test_group_ids is not None:
            from ..evaluation.metrics import pcc_per_gene_grouped
            out["pcc_per_gene_by_slide"] = float(
                pcc_per_gene_grouped(preds, targets, self.test_group_ids))
        return out

    def _predict_test(self):
        test_loader = DataLoader(
            self.test_dataset, batch_size=64, shuffle=False,
            num_workers=getattr(self.trainer, "num_workers", 0),
        )
        all_preds, all_targets = [], []
        self.trainer.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(self.trainer.device)
                all_preds.append(self.trainer.model(images).cpu().numpy())
                all_targets.append(batch["expression"].numpy())
        return np.concatenate(all_preds, 0), np.concatenate(all_targets, 0)

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
            from ..evaluation.metrics import (
                pcc_per_gene_grouped, spatial_domain_recovery, top_n_pcc_grouped,
            )
            metrics["pcc_per_gene_by_slide"] = pcc_per_gene_grouped(
                predictions, targets, self.test_group_ids
            )
            # Spatial-domain recovery (ARI/NMI) — clustering on predicted vs true
            # expression per slide (review Table 6 / pitch Evaluation Task #2).
            dom = spatial_domain_recovery(predictions, targets, self.test_group_ids)
            metrics["ari"] = dom["ari"]
            metrics["nmi"] = dom["nmi"]
            if self.top_n_cutoffs:
                # Slide-level top-N PCC (the PI's preferred convention on top-N).
                metrics["pcc_topN_by_slide"] = top_n_pcc_grouped(
                    predictions, targets, self.test_group_ids, self.top_n_cutoffs
                )
            if self.eval_ctx:
                from ..run_experiment import _add_task_metrics
                _add_task_metrics(metrics, predictions, targets,
                                  self.test_group_ids, self.eval_ctx)
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
