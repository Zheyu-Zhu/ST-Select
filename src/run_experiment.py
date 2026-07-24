"""Main entry point for running AL experiments."""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torchvision.transforms as T

from .al_methods import get_strategy, list_strategies
from .datasets import STDataset, STFeatureDataset, HESTLoader, PatientKFold
from .datasets.patch_dataset import PatchImageDataset
from .datasets.splits import BudgetMasker
from .evaluation import compute_metrics, compare_methods
from .evaluation.visualization import plot_al_curves
from .models import STNet, HisToGene, Hist2ST, FeaturePredictor, TrainableFeaturePredictor, UNIRegressor
from .preprocessing import HVGSelector, PositionExtractor, PatchCropper
from .training import ALTrainer, ActiveLearningLoop
from .utils import ExperimentConfig, set_seed, get_device


# Models whose forward is `model(images) -> (B, n_genes)` and can be trained
# directly by ALTrainer with an MSE/L1/Huber regression loss.
#   - feature_predictor: MLP over cached features (tutorial §2.9.4 fast path).
#     This is the only model that is consistent with strategies that call
#     model() on cached feature vectors (BADGE, TOD), so it is the benchmark
#     default. The image-input models below need an image dataset instead.
#   - trainable_feature_predictor: learnable projection over frozen features —
#     a controlled proxy for a trainable backbone; its representation moves each
#     round, so feature-based AL sees a "live" signal (vs. frozen features).
REGRESSION_MODELS = {"feature_predictor", "trainable_feature_predictor", "st_net", "histogene", "hist2st", "uni_regressor"}

# Models that are implemented but need a non-standard input or training path
# that ALTrainer does not provide:
#   - thitogene: forward expects a list of 3 multi-scale crops, not one image.
#   - bleep/mclstexp/egn: retrieval/contrastive (feature bank + contrastive loss).
SPECIAL_INPUT_MODELS = {"thitogene"}
RETRIEVAL_MODELS = {"bleep", "mclstexp", "egn"}


def build_model(config: ExperimentConfig, n_genes: int) -> torch.nn.Module:
    """Build a prediction model from config.

    Only regression-style models (forward: images -> expression) are supported
    by the ALTrainer used in this benchmark. Retrieval/contrastive models
    (BLEEP, mclSTExp, EGN) require a separate training/inference path.
    """
    name = config.model_name
    if name == "feature_predictor":
        return FeaturePredictor(
            n_genes=n_genes,
            feature_dim=config.feature_dim,
        )
    elif name == "trainable_feature_predictor":
        return TrainableFeaturePredictor(
            n_genes=n_genes,
            feature_dim=config.feature_dim,
        )
    elif name == "st_net":
        return STNet(
            n_genes=n_genes,
            pretrained=config.pretrained,
            frozen_backbone=config.frozen_backbone,
        )
    elif name == "uni_regressor":
        return UNIRegressor(
            n_genes=n_genes,
            frozen_backbone=config.frozen_backbone,
        )
    elif name == "histogene":
        return HisToGene(n_genes=n_genes)
    elif name == "hist2st":
        # adj defaults to None in forward(), so it runs as CNN+Transformer here.
        return Hist2ST(n_genes=n_genes, pretrained=config.pretrained)
    elif name in SPECIAL_INPUT_MODELS:
        raise NotImplementedError(
            f"Model '{name}' expects multi-scale crop inputs and cannot be "
            f"trained by ALTrainer's single-image loop. Supported regression "
            f"models: {sorted(REGRESSION_MODELS)}."
        )
    elif name in RETRIEVAL_MODELS:
        raise NotImplementedError(
            f"Model '{name}' is a retrieval/contrastive model and requires a "
            f"dedicated training loop (feature bank + contrastive loss), which "
            f"ALTrainer does not implement. Supported regression models: "
            f"{sorted(REGRESSION_MODELS)}."
        )
    else:
        all_known = REGRESSION_MODELS | SPECIAL_INPUT_MODELS | RETRIEVAL_MODELS
        raise ValueError(f"Unknown model: {name}. Available: {sorted(all_known)}.")


def run_single_experiment(
    config: ExperimentConfig,
    strategy_name: str,
    train_dataset,
    test_dataset,
    feature_cache: np.ndarray = None,
    position_cache: np.ndarray = None,
    seed: int = 42,
) -> Dict[str, List]:
    """Run a single AL experiment with one strategy."""
    set_seed(seed)
    device = get_device(config.device)

    n_genes = train_dataset[0]["expression"].shape[0]
    model = build_model(config, n_genes)
    # Pass the per-run seed so seed-bearing strategies actually vary their
    # acquisition RNG across seeds (get_strategy forwards seed only to
    # constructors that accept it).
    strategy = get_strategy(strategy_name, seed=seed)

    trainer = ALTrainer(
        model=model,
        lr=config.lr,
        weight_decay=config.weight_decay,
        epochs=config.epochs_per_round,
        batch_size=config.batch_size,
        device=device,
        loss_fn=config.loss_fn,
    )

    loop = ActiveLearningLoop(
        trainer=trainer,
        strategy=strategy,
        dataset=train_dataset,
        test_dataset=test_dataset,
        initial_budget=config.initial_budget,
        budget_per_round=config.budget_per_round,
        num_rounds=config.num_rounds,
        seed=seed,
        feature_cache=feature_cache,
        position_cache=position_cache,
    )

    return loop.run()


def load_feature_cache(config: ExperimentConfig) -> Dict:
    """Load a prepared feature cache for the fast-path benchmark.

    Expects a single .npz at <feature_cache_dir>/<dataset_name>.npz with arrays:
        features     (N, D)   frozen patch features (CONCH/UNI/DINOv2/DenseNet)
        expressions  (N, G)   log1p-normalized top-G HVG expression
        positions    (N, 2)   level-0 (x, y) spot coordinates
        patient_ids  (N,)     patient/donor id per spot (for patient k-fold)
        sample_ids   (N,)     slide id per spot (optional)

    Returns a dict of these arrays. See tutorial §1.4 / §2.9.4.
    """
    cache_path = Path(config.feature_cache_dir) / f"{config.dataset_name}.npz"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Feature cache not found at {cache_path}. Prepare it first by "
            f"extracting frozen features (see preprocessing/ and "
            f"models.get_feature_extractor), or run a smoke test with synthetic "
            f"data. Expected arrays: features, expressions, positions, patient_ids."
        )
    data = np.load(cache_path, allow_pickle=True)
    return {
        "features": data["features"].astype("float32"),
        "expressions": data["expressions"].astype("float32"),
        "positions": data["positions"].astype("float32") if "positions" in data else None,
        "patient_ids": data["patient_ids"] if "patient_ids" in data else None,
        "sample_ids": data["sample_ids"] if "sample_ids" in data else None,
        # Dispersion-ranked gene names (top-N sweep relies on this column order);
        # older caches without it still work — top-N just uses positional order.
        "gene_names": data["gene_names"] if "gene_names" in data else None,
    }


def load_image_cache(config: ExperimentConfig) -> Dict:
    """Load an index cache for the end-to-end (image-input) fast path.

    Expects <feature_cache_dir>/<dataset_name>.npz with arrays produced by
    scripts/prepare_hest_index.py: expressions, positions, patient_ids,
    sample_ids, gene_names, patch_pos. Actual pixels are read lazily from
    config.patches_dir by PatchImageDataset. `features` is None (the backbone
    consumes raw patches, not cached features).
    """
    cache_path = Path(config.feature_cache_dir) / f"{config.dataset_name}.npz"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Image index cache not found at {cache_path}. Build it with "
            f"scripts/prepare_hest_index.py (needs hest_data/patches + st)."
        )
    data = np.load(cache_path, allow_pickle=True)
    if "patch_pos" not in data:
        raise ValueError(
            f"{cache_path} has no 'patch_pos' array — it looks like a frozen "
            f"feature cache, not an image index. Run without --image-mode, or "
            f"rebuild with scripts/prepare_hest_index.py."
        )
    return {
        "features": None,
        "expressions": data["expressions"].astype("float32"),
        "positions": data["positions"].astype("float32") if "positions" in data else None,
        "patient_ids": data["patient_ids"] if "patient_ids" in data else None,
        "sample_ids": data["sample_ids"] if "sample_ids" in data else None,
        "gene_names": data["gene_names"] if "gene_names" in data else None,
        "patch_pos": data["patch_pos"].astype("int64"),
    }


def _records_from_ids(patient_ids, sample_ids, n: int) -> List[Dict]:
    """Build minimal records (patient_id/sample_id per spot) for PatientKFold."""
    records = []
    for i in range(n):
        records.append({
            "patient_id": str(patient_ids[i]) if patient_ids is not None else str(i),
            "sample_id": str(sample_ids[i]) if sample_ids is not None else str(i),
        })
    return records


def run_full_benchmark(config: ExperimentConfig) -> Dict:
    """Run a complete benchmark: AL methods x patient-folds x seeds.

    Uses the feature-cache fast path (tutorial §2.9.4): a FeaturePredictor MLP is
    trained over frozen cached features, so the same representation feeds the
    predictor and the feature-based AL strategies. Reports per-(fold, seed)
    final-round PCC, includes a full-supervision upper bound, and runs Wilcoxon
    significance vs. the random baseline.
    """
    set_seed(config.seed)

    print(f"Running benchmark on {config.dataset_name}")
    print(f"AL methods: {config.al_methods}")
    print(f"Folds: {config.n_folds}, Seeds: {config.n_seeds}")

    output_dir = Path(config.output_dir) / config.dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cache = load_image_cache(config) if config.image_mode else load_feature_cache(config)
    n_total = len(cache["expressions"])
    records = _records_from_ids(cache["patient_ids"], cache["sample_ids"], n_total)
    # Grouping key for CV: patient-level (strict) or slide-level (leakier,
    # literature-comparable, tests whether AL helps once patient-shift is removed).
    group_key = "patient_id" if config.split_level == "patient" else "sample_id"
    folds = PatientKFold(n_folds=config.n_folds, seed=config.seed).split(
        records, patient_key=group_key
    )
    print(f"Split level: {config.split_level} (group key = {group_key})")

    # Methods to evaluate: the requested AL methods plus a full-supervision
    # upper bound (train on the entire training fold). See tutorial §5/§6.
    methods = list(config.al_methods)
    if getattr(config, "add_full_supervision", True) and "full_supervision" not in methods:
        methods.append("full_supervision")

    # final_pcc[method] = list of per-(fold, seed) final-round pcc_per_gene.
    final_pcc: Dict[str, List[float]] = {m: [] for m in methods}
    detailed: Dict[str, List[Dict]] = {m: [] for m in methods}

    for method_name in methods:
        print(f"\n--- {method_name} ---")
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            for seed_idx in range(config.n_seeds):
                seed = config.seed + seed_idx
                result = _run_one_run(
                    config, method_name, cache, train_idx, test_idx, seed,
                    fold_idx=fold_idx,
                )
                final_pcc[method_name].append(result["pcc_per_gene"][-1])
                detailed[method_name].append({
                    "fold": fold_idx, "seed": seed, **result,
                })
                print(
                    f"  fold {fold_idx} seed {seed}: "
                    f"final pcc_per_gene={result['pcc_per_gene'][-1]:.4f}"
                )

    # Significance vs. random baseline (if present and enough paired samples).
    comparisons = {}
    if "random" in final_pcc:
        comparisons = compare_methods(final_pcc, baseline="random", metric="pcc_per_gene")

    # Budget-ratio sweep: pcc_per_gene at each configured ratio, per (fold,seed).
    by_ratio = _aggregate_by_ratio(detailed, config.budget_ratios)
    # Top-N per-gene PCC sweep at the final (largest-budget) checkpoint.
    pcc_topn = _aggregate_top_n(detailed, config.pcc_top_n)
    # Family-level rollup (Random / Uncertainty / Coverage / Hybrid / Spatial / RL).
    by_family = _aggregate_by_family(final_pcc)

    summary = {
        "config": config.to_dict(),
        "final_pcc_per_gene": {m: final_pcc[m] for m in methods},
        "mean_pcc_per_gene": {m: float(np.mean(final_pcc[m])) for m in methods},
        "std_pcc_per_gene": {m: float(np.std(final_pcc[m])) for m in methods},
        "comparisons_vs_random": comparisons,
        "by_ratio": by_ratio,
        "pcc_topN": pcc_topn,
        "by_family": by_family,
        "detailed": detailed,
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {results_path}")
    return summary


def _run_one_run(
    config: ExperimentConfig,
    method_name: str,
    cache: Dict,
    train_idx: List[int],
    test_idx: List[int],
    seed: int,
    fold_idx: int = 0,
) -> Dict[str, List]:
    """Train+evaluate one (method, fold, seed) on the feature fast path."""
    set_seed(seed)
    device = get_device(config.device)

    features = cache["features"]
    expressions = cache["expressions"]
    positions = cache["positions"]
    n_genes = expressions.shape[1]

    train_positions = positions[train_idx] if positions is not None else None
    if config.image_mode:
        # End-to-end: serve raw patches so the backbone trains over pixels.
        sample_ids = cache["sample_ids"]
        patch_pos = cache["patch_pos"]
        train_ds = PatchImageDataset(
            sample_ids[train_idx], patch_pos[train_idx], expressions[train_idx],
            config.patches_dir, train_positions,
        )
        test_ds = PatchImageDataset(
            sample_ids[test_idx], patch_pos[test_idx], expressions[test_idx],
            config.patches_dir,
        )
    else:
        train_ds = STFeatureDataset(
            features[train_idx], expressions[train_idx], train_positions,
        )
        test_ds = STFeatureDataset(features[test_idx], expressions[test_idx])
    # Test-fold SLIDE ids: PCC is aggregated per-slide then averaged (the field
    # standard) so per-slide batch effects don't inflate a pooled correlation.
    # Fall back to patient ids if sample ids are unavailable.
    _sids = cache.get("sample_ids")
    if _sids is None:
        _sids = cache.get("patient_ids")
    test_group_ids = _sids[test_idx] if _sids is not None else None

    # The full-supervision upper bound trains once on the whole fold for the
    # (typically longer) full_epochs schedule; per-round AL training uses the
    # shorter epochs_per_round budget.
    train_epochs = (
        config.full_epochs
        if method_name == "full_supervision"
        else config.epochs_per_round
    )
    model = build_model(config, n_genes)
    trainer = ALTrainer(
        model=model, lr=config.lr, weight_decay=config.weight_decay,
        epochs=train_epochs, batch_size=config.batch_size,
        device=device, loss_fn=config.loss_fn, num_workers=config.num_workers,
        backbone_lr_mult=getattr(config, "backbone_lr_mult", 1.0),
    )

    # Feature cache passed to AL strategies is the *training-fold* feature matrix,
    # aligned to the training dataset's local indices. In image mode there is no
    # cached feature matrix; feature-based strategies must project through the
    # (training) backbone via the loop's _extract_features path instead.
    train_feature_cache = features[train_idx] if features is not None else None
    train_position_cache = positions[train_idx] if positions is not None else None

    n_train = len(train_ds)
    top_n_cutoffs = list(getattr(config, "pcc_top_n", []) or [])
    result_keys = ("pcc_per_gene", "pcc_per_spot", "pcc_per_gene_by_slide",
                   "mse", "mae", "frac", "n_labeled")
    if top_n_cutoffs:
        result_keys = result_keys + ("pcc_topN",)

    if method_name == "full_supervision":
        # Upper bound: train on the entire labeled training fold, single pass.
        # Reported as a single checkpoint at frac = 1.0.
        trainer.train(train_ds, list(range(n_train)))
        metrics = _evaluate_model(trainer, test_ds, device, top_n_cutoffs, test_group_ids)
        # Persist the trained ceiling model (the meaningful end-to-end weights).
        _maybe_save_model(config, trainer, method_name, fold_idx, seed, n_genes)
        keys = ("pcc_per_gene", "pcc_per_spot", "pcc_per_gene_by_slide", "mse", "mae")
        out = {k: [metrics[k]] for k in keys if k in metrics}
        out["frac"] = [1.0]
        out["n_labeled"] = [n_train]
        if top_n_cutoffs:
            out["pcc_topN"] = [metrics.get("pcc_topN", {})]
        return out

    budget_targets = _budget_targets(n_train, config.budget_ratios)
    strategy = get_strategy(method_name, seed=seed)
    # Trainable-backbone regime: AL scores on the current model's moving
    # representation (recomputed each round), not the static cache.
    dynamic = config.model_name == "trainable_feature_predictor"
    loop = ActiveLearningLoop(
        trainer=trainer, strategy=strategy, dataset=train_ds, test_dataset=test_ds,
        initial_budget=min(config.initial_budget, n_train),
        budget_per_round=config.budget_per_round, num_rounds=config.num_rounds,
        seed=seed, feature_cache=train_feature_cache, position_cache=train_position_cache,
        budget_targets=budget_targets, top_n_cutoffs=top_n_cutoffs or None,
        dynamic_features=dynamic, test_group_ids=test_group_ids,
    )
    res = loop.run()
    return {k: res[k] for k in result_keys if k in res}


def _maybe_save_model(config, trainer, method_name, fold_idx, seed, n_genes) -> None:
    """Persist a trained model's weights + provenance when --save-models-dir is set.

    Saves a self-describing checkpoint so a fold's end-to-end model can be
    reloaded standalone (state_dict + the config needed to rebuild the module).
    """
    save_dir = getattr(config, "save_models_dir", None)
    if not save_dir:
        return
    out_dir = Path(save_dir) / config.dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"{method_name}_fold{fold_idx}_seed{seed}.pt"
    torch.save(
        {
            "state_dict": trainer.model.state_dict(),
            "model_name": config.model_name,
            "n_genes": n_genes,
            "dataset_name": config.dataset_name,
            "method": method_name,
            "fold": fold_idx,
            "seed": seed,
            "image_mode": config.image_mode,
            "pretrained": config.pretrained,
            "frozen_backbone": config.frozen_backbone,
        },
        ckpt,
    )
    print(f"    saved model -> {ckpt}")


def _budget_targets(n_train: int, ratios: List[float]) -> List[int]:
    """Absolute labeled-count checkpoints from budget ratios, clamped to [1, n]."""
    targets = sorted({min(max(int(round(r * n_train)), 1), n_train) for r in ratios})
    return targets


def _aggregate_by_ratio(detailed: Dict[str, List[Dict]], ratios: List[float]) -> Dict:
    """{method: {ratio: [pcc_per_gene per (fold,seed)]}} by matching each run's
    recorded `frac` checkpoints to the nearest configured budget ratio."""
    out: Dict[str, Dict[str, List[float]]] = {}
    for method, runs in detailed.items():
        per_ratio: Dict[str, List[float]] = {f"{r:.2f}": [] for r in ratios}
        for run in runs:
            fracs = run.get("frac", [])
            pccs = run.get("pcc_per_gene", [])
            for r in ratios:
                if not fracs:
                    continue
                # nearest recorded checkpoint to this ratio
                j = min(range(len(fracs)), key=lambda i: abs(fracs[i] - r))
                per_ratio[f"{r:.2f}"].append(pccs[j])
        out[method] = per_ratio
    return out


def _aggregate_top_n(detailed: Dict[str, List[Dict]], cutoffs: List[int]) -> Dict:
    """{method: {N: mean pcc_per_gene over top-N genes at the final checkpoint}}."""
    if not cutoffs:
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for method, runs in detailed.items():
        acc: Dict[int, List[float]] = {int(n): [] for n in cutoffs}
        for run in runs:
            series = run.get("pcc_topN", [])
            if not series:
                continue
            last = series[-1]  # dict {N: pcc} at the largest budget
            for n in cutoffs:
                # keys may be str after JSON round-trip; handle both
                v = last.get(n, last.get(str(n)))
                if v is not None:
                    acc[int(n)].append(float(v))
        out[method] = {
            str(n): (float(np.mean(vals)) if vals else None) for n, vals in acc.items()
        }
    return out


def _aggregate_by_family(final_pcc: Dict[str, List[float]]) -> Dict:
    """Roll per-method final PCC up to reporting families
    (Random/Uncertainty/Coverage/Hybrid/Spatial/RL). full_supervision is kept
    as its own row (it is the shared upper bound, not an AL family)."""
    from .al_methods import report_family

    fam_scores: Dict[str, List[float]] = {}
    fam_best: Dict[str, float] = {}
    for method, scores in final_pcc.items():
        if not scores:
            continue
        if method == "full_supervision":
            fam = "FullSupervision"
        else:
            fam = report_family(method)
        fam_scores.setdefault(fam, []).extend(scores)
        fam_best[fam] = max(fam_best.get(fam, float("-inf")), float(np.mean(scores)))
    return {
        fam: {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "best_method_mean": fam_best[fam],
            "n_runs": len(vals),
        }
        for fam, vals in fam_scores.items()
    }


def _evaluate_model(trainer, test_ds, device, top_n_cutoffs=None, group_ids=None) -> Dict[str, float]:
    """Evaluate a trained model on a feature dataset."""
    from torch.utils.data import DataLoader
    loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    preds, targets = [], []
    trainer.model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(trainer.device)
            preds.append(trainer.model(x).cpu().numpy())
            targets.append(batch["expression"].numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    metrics = compute_metrics(preds, targets)
    if top_n_cutoffs:
        from .evaluation.metrics import top_n_pcc_sweep
        metrics["pcc_topN"] = top_n_pcc_sweep(preds, targets, top_n_cutoffs)
    if group_ids is not None:
        from .evaluation.metrics import pcc_per_gene_grouped
        # per-slide PCC then averaged (field standard; removes per-slide batch effect)
        metrics["pcc_per_gene_by_slide"] = pcc_per_gene_grouped(preds, targets, group_ids)
    return metrics


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="AL for ST Gene Expression Prediction")
    parser.add_argument("--dataset", default="her2_breast", help="Dataset name")
    parser.add_argument("--model", default="feature_predictor", help="Model name")
    parser.add_argument("--methods", nargs="+", default=["random", "badge", "coreset", "poisson_disk"])
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument(
        "--split", choices=["patient", "slide"], default="patient",
        help="CV grouping: patient-level (strict, default) or slide-level (leakier, literature-comparable)",
    )
    parser.add_argument("--initial-budget", type=int, default=100)
    parser.add_argument("--budget-per-round", type=int, default=50)
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument(
        "--budget-ratios", nargs="+", type=float, default=None,
        help="Budget-ratio sweep, e.g. 0.05 0.10 0.20 0.30 0.50 1.00",
    )
    parser.add_argument(
        "--pcc-top-n", nargs="+", type=int, default=None,
        help="Dispersion-ranked gene cutoffs for the top-N per-gene PCC sweep",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Epochs per AL round")
    parser.add_argument(
        "--full-epochs", type=int, default=100,
        help="Epochs for the full-supervision upper bound (trained once on the whole fold)",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr-mult", type=float, default=1.0,
                        help="LR multiplier for a pretrained backbone vs head (e.g. 0.1 for UNI fine-tune)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--image-mode", action="store_true",
        help="End-to-end: train the backbone over raw patches (use with --model st_net). "
             "Loads an index cache from scripts/prepare_hest_index.py.",
    )
    parser.add_argument(
        "--patches-dir", default="./hest_data/patches",
        help="Directory of HEST patch h5 files (image mode).",
    )
    parser.add_argument(
        "--save-models-dir", default=None,
        help="If set, checkpoint each full-supervision fold model here.",
    )
    parser.add_argument("--list-methods", action="store_true", help="List available AL methods")

    args = parser.parse_args()

    if args.list_methods:
        # Registry is populated on `import al_methods` (see al_methods/__init__).
        print("Available AL methods:")
        for name in sorted(list_strategies()):
            print(f"  - {name}")
        return

    cfg_kwargs = dict(
        dataset_name=args.dataset,
        model_name=args.model,
        al_methods=args.methods,
        n_folds=args.n_folds,
        n_seeds=args.n_seeds,
        split_level=args.split,
        initial_budget=args.initial_budget,
        budget_per_round=args.budget_per_round,
        num_rounds=args.num_rounds,
        epochs_per_round=args.epochs,
        full_epochs=args.full_epochs,
        lr=args.lr,
        backbone_lr_mult=args.backbone_lr_mult,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
        image_mode=args.image_mode,
        patches_dir=args.patches_dir,
        save_models_dir=args.save_models_dir,
    )
    if args.budget_ratios is not None:
        cfg_kwargs["budget_ratios"] = args.budget_ratios
    if args.pcc_top_n is not None:
        cfg_kwargs["pcc_top_n"] = args.pcc_top_n
    config = ExperimentConfig(**cfg_kwargs)

    run_full_benchmark(config)


if __name__ == "__main__":
    main()
