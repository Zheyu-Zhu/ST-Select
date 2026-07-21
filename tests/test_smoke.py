"""End-to-end smoke test on tiny synthetic data.

Runs the real benchmark machinery (feature cache -> patient k-fold -> AL loop ->
metrics -> Wilcoxon) on a handful of fake spots so runtime breaks surface fast.
No GPU, no real data, runs in seconds on CPU.

Run with:  python -m pytest tests/test_smoke.py -q
       or:  python tests/test_smoke.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

# Make the package importable when run directly (python tests/test_smoke.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.run_experiment import run_full_benchmark, build_model  # noqa: E402
from src.utils import ExperimentConfig  # noqa: E402
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.al_methods import get_strategy, list_strategies  # noqa: E402


FEATURE_DIM = 32
N_GENES = 8


def _make_synthetic_cache(cache_dir: Path, dataset: str, n_patients=4, spots_per_patient=40):
    """Write a tiny feature cache .npz in the format load_feature_cache expects."""
    rng = np.random.default_rng(0)
    n = n_patients * spots_per_patient
    # Compute in float64 then cast: float32 matmul hits an Accelerate/vecLib bug
    # on some macOS numpy builds (spurious inf/nan), which would corrupt the data.
    features = rng.standard_normal((n, FEATURE_DIM))
    # Expressions correlated with features so a model can actually learn something.
    w = rng.standard_normal((FEATURE_DIM, N_GENES))
    expressions = features @ w + 0.1 * rng.standard_normal((n, N_GENES))
    features = features.astype("float32")
    expressions = expressions.astype("float32")
    positions = rng.uniform(0, 1000, size=(n, 2)).astype("float32")
    patient_ids = np.array([f"P{i // spots_per_patient}" for i in range(n)])
    sample_ids = patient_ids.copy()

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_dir / f"{dataset}.npz",
        features=features,
        expressions=expressions,
        positions=positions,
        patient_ids=patient_ids,
        sample_ids=sample_ids,
    )


def _tiny_config(tmp: Path, dataset: str) -> ExperimentConfig:
    return ExperimentConfig(
        dataset_name=dataset,
        model_name="feature_predictor",
        feature_dim=FEATURE_DIM,
        feature_cache_dir=str(tmp / "feature_cache"),
        output_dir=str(tmp / "results"),
        al_methods=["random", "coreset", "poisson_disk"],
        n_folds=2,
        n_seeds=2,
        initial_budget=10,
        budget_per_round=5,
        num_rounds=3,
        epochs_per_round=2,
        batch_size=16,
        device="cpu",
    )


def test_metrics_basic():
    """Metrics are finite and PCC is ~1 for near-perfect predictions."""
    rng = np.random.default_rng(1)
    targets = rng.standard_normal((20, N_GENES))
    preds = targets + 1e-3 * rng.standard_normal((20, N_GENES))
    m = compute_metrics(preds, targets)
    assert np.isfinite(m["pcc_per_gene"]) and m["pcc_per_gene"] > 0.9
    assert m["mse"] >= 0.0


def test_build_model_feature_predictor():
    cfg = ExperimentConfig(model_name="feature_predictor", feature_dim=FEATURE_DIM)
    model = build_model(cfg, n_genes=N_GENES)
    import torch
    out = model(torch.randn(4, FEATURE_DIM))
    assert out.shape == (4, N_GENES)


def test_registry_populated():
    strategies = list_strategies()
    for required in ("random", "badge", "coreset", "poisson_disk", "tod"):
        assert required in strategies
    # Every strategy must be instantiable by name.
    for name in ("random", "coreset", "poisson_disk"):
        assert get_strategy(name) is not None


def test_full_benchmark_end_to_end():
    """The headline entry point runs folds x seeds x methods and saves metrics."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        dataset = "synthetic"
        _make_synthetic_cache(tmp / "feature_cache", dataset)
        cfg = _tiny_config(tmp, dataset)

        summary = run_full_benchmark(cfg)

        # full_supervision upper bound is auto-added.
        assert "full_supervision" in summary["mean_pcc_per_gene"]
        for method in cfg.al_methods:
            scores = summary["final_pcc_per_gene"][method]
            # n_folds * n_seeds runs each.
            assert len(scores) == cfg.n_folds * cfg.n_seeds
            assert all(np.isfinite(s) for s in scores)

        # Results file written.
        results_file = tmp / "results" / dataset / "results.json"
        assert results_file.exists()


def test_retrieval_bank_leakage_guard():
    """BLEEP must reject a retrieval bank that overlaps the test set."""
    from src.models import BLEEP
    model = BLEEP(n_genes=N_GENES)
    emb = np.random.randn(5, 512).astype("float32")
    expr = np.random.randn(5, N_GENES).astype("float32")
    raised = False
    try:
        model.build_retrieval_bank(emb, expr, train_indices=[0, 1, 2], test_indices=[2, 3])
    except ValueError:
        raised = True
    assert raised, "Expected leakage guard to raise on overlapping indices"


def test_acquisition_seed_varies():
    """A seed-bearing strategy must produce different picks under different seeds."""
    from src.al_methods import get_strategy
    cand = list(range(50, 200))
    a = get_strategy("random", seed=1).select(cand, list(range(50)), 20)
    b = get_strategy("random", seed=2).select(cand, list(range(50)), 20)
    assert a != b, "random acquisition did not vary with seed (frozen-seed bug)"
    # Same seed must still be reproducible.
    a2 = get_strategy("random", seed=1).select(cand, list(range(50)), 20)
    assert a == a2, "random acquisition not reproducible for a fixed seed"


def test_budget_postcondition_fills_k():
    """select_next_batch returns exactly k distinct in-pool indices even when a
    strategy under-fills (e.g. TypiClust with collapsed clusters)."""
    from src.al_methods import get_strategy
    from src.al_methods.base import select_next_batch
    rng = np.random.default_rng(0)
    # Only 4 distinct feature vectors -> KMeans yields empty clusters for large k.
    dup = np.repeat(rng.standard_normal((4, FEATURE_DIM)), 40, axis=0).astype("float32")
    cand = list(range(len(dup)))
    picks = select_next_batch(
        get_strategy("typiclust", seed=0), cand, [], 20, features=dup,
    )
    assert len(picks) == 20, f"expected 20 picks, got {len(picks)}"
    assert len(set(picks)) == 20, "picks contain duplicates"
    assert set(picks) <= set(cand), "picks escaped the candidate pool"


def test_tod_discrepancy_nonzero_across_rounds():
    """TOD must see a non-zero temporal discrepancy once a previous checkpoint
    exists (regression for the prev_model == current_model degeneracy)."""
    import copy
    import torch
    from src.al_methods import get_strategy
    from src.models import FeaturePredictor

    rng = np.random.default_rng(0)
    feats = rng.standard_normal((120, FEATURE_DIM)).astype("float32")
    cand = list(range(40, 120))

    curr = FeaturePredictor(n_genes=N_GENES, feature_dim=FEATURE_DIM)
    prev = copy.deepcopy(curr)
    # Perturb prev so it represents an earlier, different checkpoint.
    with torch.no_grad():
        for p in prev.parameters():
            p.add_(0.5 * torch.randn_like(p))

    tod = get_strategy("tod", device="cpu", seed=0)
    disc = tod._compute_discrepancy(curr, prev, cand, feats)
    assert float(np.max(disc)) > 0.0, "TOD discrepancy is identically zero"

    # Round-0 fallback (no prev_model) must still return k valid picks.
    picks0 = tod.select(cand, [], 10, features=feats, model=curr, extras={"prev_model": None})
    assert len(set(picks0)) == 10 and set(picks0) <= set(cand)


def test_full_supervision_uses_full_epochs():
    """The full-supervision branch must build its trainer with config.full_epochs."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        dataset = "synthetic_fe"
        _make_synthetic_cache(tmp / "feature_cache", dataset)
        cfg = _tiny_config(tmp, dataset)
        cfg.full_epochs = 7  # distinct from epochs_per_round (2)

        import src.run_experiment as re

        captured = {}
        orig_trainer = re.ALTrainer

        def spy(*a, **kw):
            captured.setdefault("epochs", []).append(kw.get("epochs"))
            return orig_trainer(*a, **kw)

        re.ALTrainer = spy
        try:
            re.run_full_benchmark(cfg)
        finally:
            re.ALTrainer = orig_trainer

        assert 7 in captured["epochs"], (
            f"full_supervision did not use full_epochs=7; saw {set(captured['epochs'])}"
        )


if __name__ == "__main__":
    test_metrics_basic()
    test_build_model_feature_predictor()
    test_registry_populated()
    test_retrieval_bank_leakage_guard()
    test_acquisition_seed_varies()
    test_budget_postcondition_fills_k()
    test_tod_discrepancy_nonzero_across_rounds()
    test_full_supervision_uses_full_epochs()
    test_full_benchmark_end_to_end()
    print("All smoke tests passed.")
