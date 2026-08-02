"""Measure cross-section redundancy in a feature cache.

Why this exists
---------------
The controlled budget re-run (scripts/run_budget.py) found that early stopping
does NOT straighten the budget curve: on pg_her2 the peak sits at 5-10% of the
labels and 100% is the worst point, on every fold, in every metric (per-slide PCC,
per-gene PCC, MSE, MAE all agree). So "5% of the labels matches 100%" is not an
artifact of the fixed-epoch confound - but taken at face value it says labeling
more actively hurts, which needs a mechanism before it goes in a paper.

VERDICT: the redundancy hypothesis this script was written to test is REFUTED.
Keep the script - the measurement is sound and the numbers are worth having - but
do not cite redundancy as the explanation for early saturation. See "Result"
below before using any number from here in a write-up.

The hypothesis was: each pg cohort contributes several SECTIONS per patient
(HER2 4.5 slides/patient, BC 3.0, Kidney 1.0). If those sections are
near-replicates of the same tissue, then most of the unlabeled pool is duplicate
biology, so 5% already covers the distinct tissue and the budget curve flattens
almost immediately; where each patient contributes a single section there is no
such duplication and more spots genuinely help.

    redundancy = (same-patient-different-slide - different-patient)
                 / (same-slide - different-patient)

in cosine similarity over the cached patch features (the same representation the
model consumes). 1.0 means a different section of the same patient looks just as
similar as the same section - extra spots from that patient carry no new tissue.
0.0 means a different section is no more similar than an unrelated patient's.

Result (4000-spot subsample, 5 repeats, range in brackets). "saturates at" is the
smallest budget from which that budget and every larger one are statistically
indistinguishable from 100% (paired Wilcoxon over 12 fold x seed runs, Bonferroni
over 6 comparisons), measured in the EARLY-STOPPED arm of run_budget.py:

    dataset      slides/patient   patients w/ 2+   redundancy          saturates at
    pg_her2            4.5              8         0.91 [0.90-0.93]        5%
    pg_bc              3.0             23         0.92 [0.91-0.93]       75%
    pg_kidney          1.0              1         0.23 [0.05-0.41] !!    50%

HER2 and BC have effectively the SAME redundancy (0.91 vs 0.92, both stable to
+/-0.01) and saturate 15x apart (5% vs 75%). Redundancy therefore does not predict
saturation, and the hypothesis is dead.

Why it looked convincing at first: the original saturation figures (her2 5%, bc 5%,
kidney 50%) came from the FIXED-EPOCH runs, where every budget trained for the same
50 epochs. That protocol overtrains large budgets (measured: 3.1x on her2, 4.4x on
bc, 2.2x on kidney relative to what early stopping selects) and so manufactures
early saturation. Once each budget trains to its own optimum, BC moves 5% -> 75%
and the apparent correlation with redundancy evaporates. The lesson is about the
training protocol, not about tissue redundancy.

Read the Kidney row as "no replicate sections", NOT as "redundancy = 0.23". Only
1 of its 22 patients has a second section, so a single patient carries the entire
cross-section term and the ratio swings 0.05-0.41 depending on the subsample. The
exact, subsample-free fact for Kidney is slides/patient = 1.0: there are almost no
replicate sections for spots to be redundant WITH. HER2/BC, with 8 and 23
contributing patients, are stable to +/-0.01 and can be quoted.

What DOES track the ordering across these three cohorts is the number of TRAINING
patients (her2 6 -> 5%, kidney 16 -> 50%, bc 17 -> 75%): a cohort with few patients
appears to exhaust its available diversity almost immediately. That is a
hypothesis, not a result - n = 3 cohorts, and patient count is confounded with
cohort size (10.2k / 16.8k / 22.3k train spots). Testing it needs a controlled
breadth-vs-depth run: hold the spot budget fixed and vary how many patients those
spots are drawn from. Until then the defensible claim is the narrow one: the
saturation point is cohort-specific (5% / 50% / 75%) and has to be measured per
cohort, so "X% matches 100%" must never be quoted as a universal constant.

    conda run -n st-select python scripts/measure_redundancy.py
    conda run -n st-select python scripts/measure_redundancy.py --datasets pg_bc_hvg300
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "feature_cache"

DEFAULT = ["pg_her2_hvg300", "pg_bc_hvg300", "pg_kidney_hvg300"]


def measure(name: str, max_spots: int, seed: int) -> dict:
    path = CACHE / f"{name}.npz"
    if not path.exists():
        return {"dataset": name, "error": "cache missing"}
    d = np.load(path, allow_pickle=True)
    if "sample_ids" not in d or "patient_ids" not in d:
        return {"dataset": name, "error": "needs sample_ids + patient_ids"}

    feats = d["features"].astype("float32")
    pid = np.asarray(d["patient_ids"]).astype(str)
    sid = np.asarray(d["sample_ids"]).astype(str)

    slides_per_patient = float(np.mean(
        [len(np.unique(sid[pid == p])) for p in np.unique(pid)]
    ))

    # Subsample: this is an O(n^2) similarity matrix and the estimate is stable
    # well below the full cohort.
    rng = np.random.default_rng(seed)
    if len(feats) > max_spots:
        keep = rng.choice(len(feats), size=max_spots, replace=False)
        feats, pid, sid = feats[keep], pid[keep], sid[keep]

    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    sim = feats @ feats.T

    same_slide = sid[:, None] == sid[None, :]
    same_pt = pid[:, None] == pid[None, :]
    off_diag = ~np.eye(len(feats), dtype=bool)

    within = sim[same_slide & off_diag]
    cross_section = sim[same_pt & ~same_slide & off_diag]
    across_pt = sim[~same_pt & off_diag]

    a = float(within.mean()) if within.size else float("nan")
    b = float(cross_section.mean()) if cross_section.size else float("nan")
    c = float(across_pt.mean()) if across_pt.size else float("nan")
    red = (b - c) / (a - c) if np.isfinite(b) and (a - c) != 0 else float("nan")

    # How many patients can contribute to the cross-section term at all. When this
    # is ~1 the ratio rests on a single patient and is not a usable estimate: on
    # Kidney (1 of 22 patients has a second section) it swings 0.05-0.41 across
    # subsamples, while HER2/BC sit at 0.91/0.92 +/- 0.01. Report it so the number
    # is never quoted as a point estimate when it cannot support one.
    multi = int(sum(len(np.unique(sid[pid == p])) > 1 for p in np.unique(pid)))

    return {"dataset": name, "n_spots": len(feats), "n_patients": len(np.unique(pid)),
            "n_slides": len(np.unique(sid)), "slides_per_patient": slides_per_patient,
            "n_patients_multi_slide": multi,
            "same_slide": a, "same_patient_diff_slide": b, "diff_patient": c,
            "redundancy": red}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DEFAULT)
    ap.add_argument("--max-spots", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=5,
                    help="Subsample repeats; the spread is reported, because the "
                         "ratio is unstable when few patients have 2+ sections")
    args = ap.parse_args()

    hdr = ("%-22s %7s %8s %9s %11s %13s %13s %18s"
           % ("dataset", "slides", "sl/pt", "multi-pt", "same-slide",
              "same-pt/diff", "diff-patient", "redundancy"))
    print(hdr)
    print("-" * len(hdr))
    for name in args.datasets:
        reps = [measure(name, args.max_spots, s) for s in range(args.seeds)]
        if "error" in reps[0]:
            print("%-22s %s" % (name, reps[0]["error"]))
            continue
        r = reps[0]
        reds = [x["redundancy"] for x in reps if np.isfinite(x["redundancy"])]
        lo, hi = (min(reds), max(reds)) if reds else (float("nan"),) * 2
        mean = sum(reds) / len(reds) if reds else float("nan")
        # One patient carrying the whole cross-section term cannot support a point
        # estimate; say so instead of printing a number that looks measured.
        note = f"{mean:.2f} [{lo:.2f}-{hi:.2f}]"
        if r["n_patients_multi_slide"] < 3 or (hi - lo) > 0.15:
            note += "  UNSTABLE"
        print("%-22s %7d %8.1f %9d %11.4f %13.4f %13.4f %18s"
              % (name, r["n_slides"], r["slides_per_patient"],
                 r["n_patients_multi_slide"], r["same_slide"],
                 r["same_patient_diff_slide"], r["diff_patient"], note))
    print()
    print("redundancy = (same-patient-different-slide - different-patient)")
    print("             / (same-slide - different-patient)")
    print("~0.9 => sections are near-replicates; most of the unlabeled pool is")
    print("        duplicate tissue, so the budget curve saturates almost at once.")
    print("multi-pt = patients with 2+ sections, i.e. how many can contribute to")
    print("           the cross-section term. At <3 the ratio is not a usable")
    print("           estimate - rely on sl/pt (exact) instead, which is 1.0 when")
    print("           there are no replicate sections to be redundant with.")


if __name__ == "__main__":
    sys.exit(main())
