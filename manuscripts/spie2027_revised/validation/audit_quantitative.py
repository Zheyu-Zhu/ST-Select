"""Reproduce conference quantitative claims from saved CSVs and acquisition logs.

Reads the external ST-Net tree; writes only beside this script by default.
Does not import training code, load model weights, or read prediction arrays.
Python >=3.10, numpy, pandas and scipy are required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import t


FINAL = {0.05: 110, 0.1: 80, 0.3: 70, 0.5: 60}
DATASETS = ("HER2", "cscc")
ARMS = ("random", "coreset", "mc_dropout", "tod", "typiclust")
SOURCES: dict[str, dict] = {}


def register(path: Path) -> None:
    SOURCES[str(path.resolve())] = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def read_csv(path: Path) -> pd.DataFrame:
    register(path)
    return pd.read_csv(path)


def holm(pvalues: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p: cumulative max of (m-rank)*ordered p."""
    order = np.argsort(pvalues)
    adjusted = np.minimum(1.0, np.maximum.accumulate(
        pvalues[order] * np.arange(len(pvalues), 0, -1)))
    result = np.empty(len(pvalues))
    result[order] = adjusted
    return result


def corrected_summary(x: np.ndarray) -> dict:
    """Exploratory Nadeau-Bengio correction for one balanced 4-fold CV.

    Assumes nominal test/train ratio 1/3 at patient split level, not label
    budget. This is a heuristic sensitivity analysis for this grouped,
    single-partition, single-seed regression benchmark, not calibrated
    confirmatory inference. CI is pointwise, not multiplicity adjusted.
    """
    assert len(x) == 4 and np.isfinite(x).all()
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    se = float(np.sqrt((1 / 4 + 1 / 3) * sd**2))
    statistic = mean / se if se else (0.0 if mean == 0 else np.sign(mean) * np.inf)
    p = float(2 * t.sf(abs(statistic), df=3))
    half = float(t.ppf(0.975, df=3) * se)
    return dict(delta_mean=mean, delta_sd=sd, corrected_se=se,
                corrected_t=float(statistic), df=3, p_exploratory=p,
                pointwise_ci95_low=mean-half, pointwise_ci95_high=mean+half,
                wins=int((x > 0).sum()), n_folds=4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--st-root", type=Path,
        default=Path(r"C:\Users\zheyu\workspace\stnet_ref\ST-Net"))
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--manuscript", type=Path,
        default=Path(__file__).resolve().parent.parent / "main_original.tex",
        help="Optional preserved original used only for checking printed PCC precision")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows, learning = [], []
    for dataset in DATASETS:
        for sampler in ARMS:
            for budget, epoch in FINAL.items():
                for fold in range(4):
                    base = args.st_root / "weights_st" / dataset / f"{sampler}_{budget}" / f"fold{fold}"
                    path = base / "slidewise_metrics" / f"epoch_{epoch}.csv"
                    previous = base / "slidewise_metrics" / f"epoch_{epoch-10}.csv"
                    current, prior = read_csv(path), read_csv(previous)
                    assert current.slide.is_unique and prior.slide.is_unique
                    assert set(current.slide) == set(prior.slide)
                    assert (current.epoch == epoch).all()
                    now, before = float(current.pcc.mean()), float(prior.pcc.mean())
                    rows.append(dict(dataset=dataset, sampler=sampler, budget=budget,
                                     fold=fold, seed=42, epoch=epoch, n_slides=len(current),
                                     pcc=now, source=str(path.resolve())))
                    learning.append(dict(dataset=dataset, sampler=sampler, budget=budget,
                        fold=fold, earlier_epoch=epoch-10, final_epoch=epoch,
                        earlier_pcc=before, final_pcc=now, change=now-before,
                        percent_of_final=100*(now-before)/now if now else np.nan))
        for fold in range(4):
            path = (args.st_root / "weights_st" / dataset / "random_1.0" /
                    f"fold{fold}" / "slidewise_metrics" / "epoch_50.csv")
            d = read_csv(path)
            rows.append(dict(dataset=dataset, sampler="full_label", budget=1.0,
                fold=fold, seed=42, epoch=50, n_slides=len(d),
                pcc=float(d.pcc.mean()), source=str(path.resolve())))

    folds = pd.DataFrame(rows)
    assert len(folds) == 168 and not folds.duplicated(["dataset", "sampler", "budget", "fold"]).any()
    folds.to_csv(out / "pcc_fold_values.csv", index=False)
    summaries = (folds.groupby(["dataset", "sampler", "budget", "epoch"], sort=False)
        .pcc.agg(["mean", "std", "count"]).reset_index()
        .rename(columns={"std": "sample_sd", "count": "n_folds"}))
    assert (summaries.n_folds == 4).all()
    summaries["mean_sd_4dp"] = summaries.apply(lambda r: f"{r['mean']:.4f} +/- {r['sample_sd']:.4f}", axis=1)
    summaries.to_csv(out / "pcc_summary.csv", index=False)

    pair_rows, paired_fold_rows = [], []
    for dataset in DATASETS:
        for sampler in ARMS[1:]:
            for budget in FINAL:
                sub = folds.query("dataset == @dataset and budget == @budget")
                a = sub.query("sampler == @sampler").set_index("fold").pcc.sort_index()
                r = sub.query("sampler == 'random'").set_index("fold").pcc.sort_index()
                diff = a-r
                assert diff.index.tolist() == list(range(4))
                pair_rows.append(dict(dataset=dataset, sampler=sampler, budget=budget,
                                      **corrected_summary(diff.to_numpy())))
                paired_fold_rows.extend(dict(dataset=dataset, sampler=sampler, budget=budget,
                    fold=int(f), delta=float(v)) for f, v in diff.items())
    comparisons = pd.DataFrame(pair_rows)
    assert len(comparisons) == 32
    comparisons["p_holm_32_exploratory"] = holm(comparisons.p_exploratory.to_numpy())
    comparisons["reject_holm_005_exploratory"] = comparisons.p_holm_32_exploratory < 0.05
    comparisons.to_csv(out / "paired_pcc_exploratory.csv", index=False)
    paired_folds = pd.DataFrame(paired_fold_rows)
    paired_folds.to_csv(out / "paired_pcc_fold_values.csv", index=False)
    pooled = paired_folds.groupby(["dataset", "budget"]).agg(
        mean_delta=("delta", "mean"), descriptive_strategy_fold_wins=("delta", lambda x: int((x>0).sum())),
        descriptive_strategy_fold_count=("delta", "size")).reset_index()
    # These pooled counts and means are descriptions, never inputs to a test:
    # four strategies share each Random comparator and the same CV partition.
    pooled.to_csv(out / "pooled_descriptive_pcc.csv", index=False)

    matched_rows, matched_folds = [], []
    for dataset in DATASETS:
        delta, half_values, full_values = [], [], []
        for fold in range(4):
            path = (args.st_root / "weights_st" / dataset / "random_0.5" /
                    f"fold{fold}" / "slidewise_metrics" / "epoch_50.csv")
            half = float(read_csv(path).pcc.mean())
            full = float(folds.query("dataset == @dataset and sampler == 'full_label' and fold == @fold").pcc.iloc[0])
            delta.append(full-half)
            half_values.append(half)
            full_values.append(full)
            matched_folds.append(dict(dataset=dataset, fold=fold, half_pcc=half,
                                      full_pcc=full, full_minus_half=full-half))
        matched_rows.append(dict(dataset=dataset, comparison="full_minus_random_half_epoch50",
                                 half_mean=float(np.mean(half_values)), half_sample_sd=float(np.std(half_values, ddof=1)),
                                 full_mean=float(np.mean(full_values)), full_sample_sd=float(np.std(full_values, ddof=1)),
                                 **corrected_summary(np.asarray(delta))))
    pd.DataFrame(matched_rows).to_csv(out / "epoch50_matched_pcc.csv", index=False)
    pd.DataFrame(matched_folds).to_csv(out / "epoch50_matched_fold_values.csv", index=False)

    learn = pd.DataFrame(learning)
    learn.to_csv(out / "learning_last10_fold_values.csv", index=False)
    learn_summary = learn.groupby(["dataset", "sampler", "budget"], sort=False).agg(
        earlier_pcc=("earlier_pcc", "mean"), final_pcc=("final_pcc", "mean"),
        change=("change", "mean"), largest_fold_abs_change=("change", lambda x: abs(x).max())).reset_index()
    learn_summary["percent_of_final"] = 100*learn_summary.change/learn_summary.final_pcc
    learn_summary.to_csv(out / "learning_last10_summary.csv", index=False)

    spatial = []
    for dataset in DATASETS:
        sp = read_csv(args.st_root / f"{dataset}_spatial_metrics_full.csv")
        assert len(sp) == 84 and not sp.duplicated(["sampler", "budget", "fold"]).any()
        for (sampler, budget), group in sp.groupby(["sampler", "budget"]):
            assert len(group) == 4
            for metric in ("ari", "nmi", "moran_fidelity"):
                r = sp.query("sampler == 'random' and budget == @budget").set_index("fold")[metric]
                a = group.set_index("fold")[metric]
                spatial.append(dict(dataset=dataset, sampler=sampler, budget=budget, metric=metric,
                    mean=float(a.mean()), sample_sd=float(a.std(ddof=1)),
                    delta_mean=float((a-r).mean()), wins=int(((a-r)>0).sum())))
    pd.DataFrame(spatial).to_csv(out / "spatial_descriptive_summary.csv", index=False)

    timing_rows = []
    pattern = re.compile(r"\[(coreset|mc_dropout|tod|typiclust) Sampling\] epoch=(\d+), k=(\d+), pool=(\d+), ([\d.]+)s")
    for dataset in DATASETS:
        for sampler in ARMS[1:]:
            for budget in FINAL:
                for fold in range(4):
                    path = args.st_root / "logs" / f"{dataset}_{sampler}_{budget}_fold{fold}.log"
                    if not path.exists():
                        continue
                    register(path)
                    text = path.read_text(encoding="utf-8", errors="replace")
                    events = list((args.st_root / "weights_st" / dataset / f"{sampler}_{budget}" /
                                   f"fold{fold}" / "log").glob("events.out.tfevents.*"))
                    starts = [int(p.name.split(".")[3]) for p in events if p.name.split(".")[3].isdigit()]
                    latest_start = datetime.fromtimestamp(max(starts), tz=timezone.utc).isoformat() if starts else ""
                    for match in pattern.finditer(text):
                        timing_rows.append(dict(dataset=dataset, sampler=sampler, budget=budget, fold=fold,
                            epoch=int(match[2]), quota=int(match[3]), remaining_pool=int(match[4]),
                            seconds=float(match[5]), latest_tensorboard_start_utc=latest_start,
                            source=str(path.resolve())))
    timings = pd.DataFrame(timing_rows)
    assert not timings.duplicated(["dataset", "sampler", "budget", "fold", "epoch"]).any()
    timings.to_csv(out / "acquisition_timing_rounds.csv", index=False)
    timing_summary = timings.groupby(["dataset", "sampler", "budget", "fold"], sort=False).agg(
        n_logged_rounds=("seconds", "size"), median_seconds=("seconds", "median"),
        min_seconds=("seconds", "min"), max_seconds=("seconds", "max"), total_seconds=("seconds", "sum"),
        quota=("quota", "first"), latest_tensorboard_start_utc=("latest_tensorboard_start_utc", "first")).reset_index()
    timing_summary.to_csv(out / "acquisition_timing_summary.csv", index=False)

    precision_findings = {}
    if args.manuscript.exists():
        register(args.manuscript)
        original = args.manuscript.read_text(encoding="utf-8")
        audit = []
        for dataset, label in (("HER2", "tab:pcc_her2"), ("cscc", "tab:pcc_cscc")):
            start = original.index("\\label{"+label+"}")
            end = original.index("\\end{table}", start)
            for line in original[start:end].splitlines():
                sampler = next((s for s,prefix in (("random", "Random &"),
                    ("coreset", "CoreSet~"), ("mc_dropout", "MC-dropout~"),
                    ("tod", "TOD~"), ("typiclust", "TypiClust-inspired~"))
                    if line.startswith(prefix)), None)
                if sampler is None:
                    continue
                # Remove bold wrappers without changing the literal numbers.
                plain = re.sub(r"\\mathbf\{([^{}]*)\}", r"\1", line)
                cells = re.findall(r"(\d+\.\d+)\s*\\pm\s*(\d+\.\d+)", plain)
                assert len(cells) == 4, (dataset, sampler, cells)
                for budget, (printed_mean, printed_sd) in zip(FINAL, cells):
                    row = summaries.query("dataset == @dataset and sampler == @sampler and budget == @budget").iloc[0]
                    true_mean, true_sd = f"{row['mean']:.4f}", f"{row['sample_sd']:.4f}"
                    audit.append(dict(dataset=dataset, sampler=sampler, budget=budget,
                        printed_mean=printed_mean, audited_mean_4dp=true_mean,
                        mean_matches_4dp=printed_mean==true_mean,
                        printed_sd=printed_sd, audited_sample_sd_4dp=true_sd,
                        sd_matches_4dp=printed_sd==true_sd))
        printed_audit = pd.DataFrame(audit)
        assert len(printed_audit) == 40
        printed_audit.to_csv(out / "original_pcc_precision_audit.csv", index=False)
        precision_findings = {"original_manuscript_pcc_means_differing_at_4dp": int((~printed_audit.mean_matches_4dp).sum()),
            "original_manuscript_pcc_sds_differing_at_4dp": int((~printed_audit.sd_matches_4dp).sum())}

    findings = {
        "scope": "HER2/cSCC, seed42 only; saved slidewise PCC CSV means then four-fold summaries",
        "sd_ddof": 1,
        "fold_cells": len(folds),
        "comparisons_in_holm_family": len(comparisons),
        "holm_rejections_005_exploratory": int(comparisons.reject_holm_005_exploratory.sum()),
        "minimum_raw_p_exploratory": float(comparisons.p_exploratory.min()),
        "minimum_holm_p_exploratory": float(comparisons.p_holm_32_exploratory.min()),
        "her2_30_percent_pooled_delta_exact": float(pooled.query("dataset == 'HER2' and budget == 0.3").mean_delta.iloc[0]),
        **precision_findings,
        "last10_percent_of_final_extrema_by_budget": {
            str(b): {"min": float(g.percent_of_final.min()), "max": float(g.percent_of_final.max()),
                     "max_abs": float(g.percent_of_final.abs().max())}
            for b,g in learn_summary.groupby("budget")},
        "inference_limitations": [
            "The four CV folds share training patients and are not independent replicates.",
            "The corrected test uses a heuristic nominal test/train ratio 1/3 and df=3.",
            "One fixed partition and one optimization seed do not establish seed or split robustness.",
            "Grouped heterogeneous patients and nonlinear PCC were not calibrated in this test's derivation.",
            "No rejection does not establish equivalence or noninferiority.",
            "Pointwise intervals are approximate and not simultaneous; Holm applies only to the 32 PCC p-values.",
            "Epoch50 matching equalizes epochs, not optimizer steps or labeled-example presentations. The actual LR is iteration-based with shared denominator 1e8, not epoch-horizon-normalized; see separate LR scalar audit.",
            "Tail changes do not prove convergence or choose an early-stopping epoch.",
            "Timing logs lack load monitoring; cross-date timing differences are descriptive, not a controlled scaling experiment.",
        ],
        "method_sources": {
            "Nadeau_Bengio_1999": "https://papers.nips.cc/paper_files/paper/1999/file/7d12b66d3df6af8d429c1a357d8b9e1a-Paper.pdf",
            "Bouckaert_Frank_2004": "https://ml.cms.waikato.ac.nz/publications/2004/bouckaert-frank.pdf",
            "Holm_1979": "https://www.jstor.org/stable/4615733",
        },
        "analysis_environment_versions_not_training_versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__},
    }
    (out / "quantitative_findings.json").write_text(json.dumps(findings, indent=2)+"\n", encoding="utf-8")
    (out / "quantitative_source_hashes.json").write_text(json.dumps(SOURCES, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps(findings, indent=2))


if __name__ == "__main__":
    main()
