"""cSCC results: per-slide PCC and Moran's I fidelity, in the manuscript's shape.

PCC comes straight from the stored final-epoch metrics; Moran fidelity comes from
add_spatial_metrics.py's output, which has to have been run for the cells being
reported. Cells that are not complete on all four folds are printed as TK rather
than averaged over the folds that finished: a three-fold mean is not comparable
with a four-fold one when the folds differ severalfold.

    python scripts/cscc_tables.py
    python scripts/cscc_tables.py --dataset HER2
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

ST = r"C:\Users\zheyu\workspace\stnet_ref\ST-Net"
EPOCH = {0.05: 110, 0.1: 80, 0.3: 70, 0.5: 60}
ARMS = ["random", "coreset", "mc_dropout", "tod", "typiclust"]
LABEL = {"random": "Random", "coreset": "CoreSet", "mc_dropout": "MC-dropout",
         "tod": "TOD", "typiclust": "TypiClust"}


def pcc_by_fold(ds):
    rows = []
    for f in glob.glob(os.path.join(ST, "weights_st", ds, "*", "fold*",
                                    "slidewise_metrics", "epoch_*.csv")):
        p = f.replace("\\", "/").split("/")
        s, r = p[-4].rsplit("_", 1)
        try:
            r = float(r)
        except ValueError:
            continue
        if r not in EPOCH:
            continue
        e = int(re.search(r"epoch_(\d+)", p[-1]).group(1))
        if e != EPOCH[r]:
            continue
        fold = int(re.search(r"fold(\d+)", p[-3]).group(1))
        d = pd.read_csv(f)
        rows.append(dict(sampler=s, budget=r, fold=fold,
                         pcc=d["pcc"].mean(), mse=d["mse"].mean(),
                         n_slides=len(d)))
    return pd.DataFrame(rows)


def cell(g, digits=4):
    """Mean +- s.d. over folds, or TK when the four folds are not all in."""
    if len(g) < 4:
        return "TK", np.nan
    return f"${g.mean():.{digits}f} \\pm {g.std(ddof=1):.{digits}f}$", g.mean()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cscc")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    d = pcc_by_fold(args.dataset)
    if d.empty:
        raise SystemExit(f"no completed cells for {args.dataset}")

    print(f"=== {args.dataset}: per-slide PCC, mean over 4 folds ===")
    print(f"{'strategy':12s}" + "".join(f"{int(b*100):>22d}%" for b in EPOCH))
    means = {}
    for a in ARMS:
        line = f"{LABEL[a]:12s}"
        for b in EPOCH:
            g = d[(d.sampler == a) & (d.budget == b)].pcc
            txt, m = cell(g)
            means[(a, b)] = m
            line += f"{txt.replace('$', '').replace(chr(92)+'pm', '+-'):>23s}"
        print(line)

    print(f"\n=== difference from Random, per budget ===")
    print(f"{'strategy':12s}" + "".join(f"{int(b*100):>16d}%" for b in EPOCH))
    for a in ARMS[1:]:
        line = f"{LABEL[a]:12s}"
        for b in EPOCH:
            gr = d[(d.sampler == "random") & (d.budget == b)].set_index("fold").pcc
            ga = d[(d.sampler == a) & (d.budget == b)].set_index("fold").pcc
            common = gr.index.intersection(ga.index)
            if len(common) < 4:
                line += f"{'TK':>17s}"
                continue
            diff = (ga[common] - gr[common])
            line += f"{diff.mean():>+10.4f} ({int((diff > 0).sum())}/4)"
        print(line)

    sp = os.path.join(ST, f"{args.dataset}_spatial_metrics.csv")
    print(f"\n=== Moran's I fidelity ===")
    if not os.path.exists(sp):
        print(f"  {os.path.basename(sp)} does not exist yet")
        return 0
    m = pd.read_csv(sp)
    have = m.groupby(["sampler", "budget"]).size()
    print(f"  {len(m)} rows on disk; complete cells: "
          f"{int((have == 4).sum())}/{len(ARMS)*len(EPOCH)}")
    print(f"{'strategy':12s}" + "".join(f"{int(b*100):>22d}%" for b in EPOCH))
    for a in ARMS:
        line = f"{LABEL[a]:12s}"
        for b in EPOCH:
            g = m[(m.sampler == a) & (np.isclose(m.budget, b))].moran_fidelity
            txt, _ = cell(g)
            line += f"{txt.replace('$', '').replace(chr(92)+'pm', '+-'):>23s}"
        print(line)

    print(f"\n=== ARI ===")
    print(f"{'strategy':12s}" + "".join(f"{int(b*100):>22d}%" for b in EPOCH))
    for a in ARMS:
        line = f"{LABEL[a]:12s}"
        for b in EPOCH:
            g = m[(m.sampler == a) & (np.isclose(m.budget, b))].ari
            txt, _ = cell(g)
            line += f"{txt.replace('$', '').replace(chr(92)+'pm', '+-'):>23s}"
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
