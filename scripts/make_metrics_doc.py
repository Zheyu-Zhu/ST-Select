"""A Word document describing the metrics the sweep records but the paper does
not report, for a second opinion on what is worth including.

Per-slide PCC and Moran's I fidelity are already in the manuscript and are only
named here for context. Everything else is something that exists on disk,
computed for both completed cohorts, and currently unused.

Each section says what the quantity measures, why it might earn a place, what
the numbers actually show, and what argues against including it. The last part
matters: several of these are interesting and still not worth a table.

    python scripts/make_metrics_doc.py
"""

import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
ST = Path(r"C:\Users\zheyu\workspace\stnet_ref\ST-Net")
FIG = ROOT / "figures" / "assets" / "extra"
OUT = ROOT / "ST-Select_extra_metrics.docx"

EPOCH = {0.05: 110, 0.1: 80, 0.3: 70, 0.5: 60}
ARMS = ["random", "coreset", "mc_dropout", "tod", "typiclust"]
LABEL = {"random": "Random", "coreset": "CoreSet", "mc_dropout": "MC-dropout",
         "tod": "TOD", "typiclust": "TypiClust"}
INK = RGBColor(0x1C, 0x1B, 0x18)
MUTED = RGBColor(0x5A, 0x5A, 0x5A)


# --------------------------------------------------------------- data helpers
def spatial(ds):
    p = ST / f"{ds}_spatial_metrics.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def slidewise_final(ds):
    rows = []
    for f in glob.glob(str(ST / "weights_st" / ds / "*" / "fold*" /
                           "slidewise_metrics" / "epoch_*.csv")):
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
        d = pd.read_csv(f)
        d["sampler"], d["budget"] = s, r
        d["fold"] = int(re.search(r"fold(\d+)", p[-3]).group(1))
        rows.append(d)
    return pd.concat(rows) if rows else pd.DataFrame()


def cost_table():
    rows = []
    for f in glob.glob(str(ST / "logs" / "*_fold*.log")):
        m = re.match(r"(\w+?)_(random|coreset|mc_dropout|tod|typiclust)"
                     r"_([0-9.]+)_fold(\d)", os.path.basename(f))
        if not m:
            continue
        secs = [float(x) for x in re.findall(
            r"pool=\d+, ([0-9.]+)s",
            open(f, encoding="utf-8", errors="replace").read())]
        if secs:
            rows.append(dict(dataset=m.group(1), sampler=m.group(2),
                             budget=float(m.group(3)), s=float(np.median(secs))))
    return pd.DataFrame(rows)


# --------------------------------------------------------------- doc helpers
def style(doc):
    n = doc.styles["Normal"]
    n.font.name = "Calibri"
    n.font.size = Pt(10.5)
    n.font.color.rgb = INK
    n.paragraph_format.space_after = Pt(7)
    n.paragraph_format.line_spacing = 1.15


def para(doc, text, size=10.5, italic=False, colour=INK, space=7):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.italic = italic
    r.font.color.rgb = colour
    p.paragraph_format.space_after = Pt(space)
    return p


def kv(doc, label, text):
    p = doc.add_paragraph()
    a = p.add_run(f"{label}  ")
    a.bold = True
    a.font.size = Pt(10.5)
    a.font.color.rgb = INK
    b = p.add_run(text)
    b.font.size = Pt(10.5)
    b.font.color.rgb = INK
    p.paragraph_format.space_after = Pt(5)


def figure(doc, name, caption, width=6.4):
    p = FIG / f"{name}.png"
    if not p.exists():
        para(doc, f"[missing figure: {name}.png]", italic=True, colour=MUTED)
        return
    doc.add_picture(str(p), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    c = para(doc, caption, size=8.5, italic=True, colour=MUTED, space=12)
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER


def table(doc, header, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    for i, h in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = ""
        r = cell.paragraphs[0].add_run(str(h))
        r.bold = True
        r.font.size = Pt(9)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(v))
            r.font.size = Pt(9)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return t


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    doc = Document()
    style(doc)
    for s in doc.sections:
        s.left_margin = s.right_margin = Inches(0.9)
        s.top_margin = s.bottom_margin = Inches(0.8)

    doc.add_heading("ActiveST-Bench: metrics we record but do not report", 0)
    para(doc, "Junchao — the manuscript currently reports two things: per-slide "
              "PCC and Moran's I fidelity. Everything below is already computed "
              "and sitting on disk for both completed cohorts (HER2, 80 runs; "
              "cSCC, 80 runs). Kidney is running now and is not in any of these "
              "numbers. I would like your view on which of these earn a place "
              "in an eight-page SPIE paper and which are supplementary at best.",
         space=10)
    para(doc, "Each section says what the quantity is, what it shows here, and "
              "the argument against including it. Figures are the same for both "
              "cohorts wherever both were available.", italic=True,
         colour=MUTED, space=14)

    # ------------------------------------------------------------------ 1
    doc.add_heading("1. Acquisition cost", level=1)
    kv(doc, "What it is:", "wall-clock seconds the selection step itself takes "
                           "per acquisition round, excluding training. Six "
                           "rounds per run.")
    kv(doc, "Why it might matter:", "a strategy that needs an hour of CPU per "
                                    "round to gain 0.005 correlation is not a "
                                    "practical recommendation, and cost is the "
                                    "axis this whole study is about.")
    c = cost_table()
    if not c.empty:
        rows = []
        for a in ARMS:
            r = [LABEL[a]]
            for ds in ("HER2", "cscc"):
                for b in (0.05, 0.5):
                    g = c[(c.sampler == a) & (c.dataset == ds) &
                          (np.isclose(c.budget, b))].s
                    r.append(f"{g.median():.0f} s" if len(g) else "—")
            rows.append(r)
        table(doc, ["Strategy", "HER2 5%", "HER2 50%", "cSCC 5%", "cSCC 50%"],
              rows)
    kv(doc, "What it shows:", "every strategy except TypiClust is flat or gets "
                              "cheaper as the budget grows. TypiClust rises by "
                              "a factor of 33 on HER2, because its k-means uses "
                              "one cluster per spot in the round quota. On the "
                              "one HER2 50% fold that finished, 2.2 of its 3.6 "
                              "hours were CPU clustering with the GPU idle.")
    figure(doc, "cost_per_round",
           "Median seconds per selection round. Log axes. Random is drawn at "
           "the floor because it is sub-second.")
    kv(doc, "Against including it:", "none that I can see. This is the cheapest "
                                     "addition in the list and the professor "
                                     "asked for it explicitly.")

    # ------------------------------------------------------------------ 2
    doc.add_heading("2. Over-smoothing: Moran's I of the predicted map itself",
                    level=1)
    kv(doc, "What it is:", "Moran's I measures how spatially clustered a map "
                           "is. Fidelity, which we already report, is the "
                           "correlation across genes between the true and "
                           "predicted I values — it asks whether the model gets "
                           "the ranking right. These are the two levels "
                           "themselves.")
    kv(doc, "Why it might matter:", "a model can rank genes correctly while "
                                    "producing maps that are far too smooth. "
                                    "Fidelity cannot show that; these two "
                                    "numbers can.")
    rows = []
    for ds, name in (("HER2", "HER2"), ("cscc", "cSCC")):
        m = spatial(ds)
        if m.empty:
            continue
        for b in EPOCH:
            g = m[np.isclose(m.budget, b)]
            if g.empty:
                continue
            rows.append([name, f"{int(b*100)}%",
                         f"{g.moran_true.mean():.3f}",
                         f"{g.moran_pred.mean():.3f}",
                         f"{g.moran_pred.mean() - g.moran_true.mean():+.3f}"])
    table(doc, ["Cohort", "Budget", "Measured I", "Predicted I", "Difference"],
          rows)
    kv(doc, "What it shows:", "predictions are consistently MORE clustered than "
                              "the truth, and the gap widens with budget. On "
                              "HER2 at 50% the predicted maps average I = 0.41 "
                              "against a measured 0.26. Both cohorts behave the "
                              "same way.")
    figure(doc, "oversmoothing",
           "Mean Moran's I over genes, measured against predicted. The shaded "
           "gap is the over-smoothing.")
    kv(doc, "Against including it:", "it is a property of the predictor, not of "
                                     "the acquisition strategy, so it does not "
                                     "speak to the paper's question. It may "
                                     "belong in a sentence rather than a table.")

    # ------------------------------------------------------------------ 3
    doc.add_heading("3. Convergence: the full epoch curve", level=1)
    kv(doc, "What it is:", "per-slide PCC recorded every ten epochs of every "
                           "run, rather than only at the end.")
    kv(doc, "Why it might matter:", "the paper's central claim is that active "
                                    "selection loses at small budgets. The "
                                    "obvious objection is that the small-budget "
                                    "arms simply had not converged. This is the "
                                    "direct evidence either way.")
    d = slidewise_final("HER2")
    kv(doc, "What it shows:", "at 5% every arm is still climbing steeply when "
                              "training stops; at 30% and 50% the curves are "
                              "flat well before the end. So the low-budget "
                              "result is honestly stated as 'worse at a fixed "
                              "training horizon', not 'worse at convergence'. "
                              "The paper already says this; the figure makes it "
                              "checkable.")
    figure(doc, "convergence_HER2",
           "HER2. Per-slide PCC against epoch, one panel per budget. An arm is "
           "drawn only where all four folds finished.")
    figure(doc, "convergence_cSCC", "cSCC, same construction.")
    kv(doc, "Against including it:", "it is four panels wide and it argues "
                                     "against our own result. I think that is a "
                                     "reason to include it, but it is a real "
                                     "cost in space and in framing.")

    # ------------------------------------------------------------------ 4
    doc.add_heading("4. Between-patient spread", level=1)
    kv(doc, "What it is:", "the same per-slide PCC, broken out by held-out "
                           "patient instead of averaged over folds.")
    kv(doc, "Why it might matter:", "it sets the scale against which every "
                                    "strategy effect should be read.")
    rows = []
    for ds, name in (("HER2", "HER2"), ("cscc", "cSCC")):
        dd = slidewise_final(ds)
        if dd.empty:
            continue
        g = dd[dd.sampler == "random"].groupby("slide").pcc.mean()
        rows.append([name, f"{g.min():+.3f}", f"{g.max():+.3f}",
                     f"{g.max()-g.min():.3f}", f"{g.std():.3f}"])
    table(doc, ["Cohort", "Worst slide", "Best slide", "Range", "s.d."], rows)
    kv(doc, "What it shows:", "on HER2 the patient means run from −0.022 to "
                              "+0.458 — a twentyfold spread — while the "
                              "strategy effects we are reporting are about "
                              "0.02. Which patients land in which fold matters "
                              "far more than which strategy is used.")
    figure(doc, "patient_spread_HER2",
           "HER2. Each dot is one slide under the Random baseline; the bar is "
           "the patient mean.")
    figure(doc, "patient_spread_cSCC", "cSCC, same construction.")
    kv(doc, "Against including it:", "it undercuts confidence in every number "
                                     "in the paper. It is also true, and a "
                                     "reviewer will find it anyway.")

    # ------------------------------------------------------------------ 5
    doc.add_heading("5. How much of the correlation is sequencing depth?",
                    level=1)
    kv(doc, "What it is:", "the targets are log(1+x) raw counts with no "
                           "library-size normalisation. Projecting each spot's "
                           "log library size out of both the prediction and the "
                           "target leaves the gene-specific part.")
    kv(doc, "Why it might matter:", "morphology predicts local cellularity, and "
                                    "cellularity drives how much RNA a spot "
                                    "captures. A model could score well by "
                                    "predicting depth and nothing else.")
    kv(doc, "What it shows:", "about a fifth of the reported correlation "
                              "survives — 21%, 22%, 24%, 24% across the four "
                              "budgets on HER2, measured over 693 slide-level "
                              "cells. Crucially the strategy ordering is "
                              "unchanged, so the comparison the paper makes is "
                              "unaffected; what changes is how the absolute "
                              "numbers should be read.")
    figure(doc, "depth_confound_HER2",
           "HER2. Per-slide PCC as reported, and after removing the "
           "library-size component from both sides. Labels are the fraction "
           "retained.")
    figure(doc, "depth_confound_cSCC", "cSCC, same construction.")
    kv(doc, "Against including it:", "it is the most likely of these to be "
                                     "misread as 'the results are invalid'. If "
                                     "it goes in it needs the sentence about "
                                     "relative comparisons being unaffected, in "
                                     "the same breath.")

    # ------------------------------------------------------------------ 6
    doc.add_heading("6. NMI beside ARI", level=1)
    kv(doc, "What it is:", "normalised mutual information between the same two "
                           "k-means partitions the adjusted Rand index scores.")
    kv(doc, "Why it might matter:", "the manuscript says NMI tracks ARI and "
                                    "omits it for space. A reviewer has already "
                                    "asked us to either show it or drop the "
                                    "claim.")
    both = pd.concat([spatial(d) for d in ("HER2", "cscc")], ignore_index=True)
    if not both.empty:
        r = float(np.corrcoef(both.ari, both.nmi)[0, 1])
        kv(doc, "What it shows:", f"they correlate at r = {r:.3f} across all "
                                  f"{len(both)} cells. NMI is systematically "
                                  f"higher than ARI, which the sentence in the "
                                  f"paper does not currently say.")
    figure(doc, "ari_vs_nmi",
           "Every strategy-budget-fold cell from both cohorts.", width=4.4)
    kv(doc, "Against including it:", "low information per column inch. One "
                                     "sentence with the correlation coefficient "
                                     "would satisfy the reviewer without a "
                                     "figure.")

    # ------------------------------------------------------------------ 7
    doc.add_heading("7. Recorded but probably not worth reporting", level=1)
    para(doc, "MSE and MAE are stored per slide for every run. They correlate "
              "closely with PCC and would lengthen the tables without changing "
              "any conclusion.")
    para(doc, "Mean absolute error of Moran's I per gene is stored alongside "
              "fidelity; it says the same thing as the over-smoothing gap in "
              "section 2, less directly.")
    para(doc, "All 7,545 per-spot prediction arrays are kept (about 6.7 GB), so "
              "any further per-gene or per-spot metric can be computed without "
              "retraining. If there is something you would want to see, it "
              "probably does not need a new run.")

    doc.add_heading("What I would pick", level=1)
    para(doc, "Acquisition cost (section 1) and the convergence curves "
              "(section 3) seem clearly worth the space to me: the first was "
              "asked for and is cheap, the second defends the paper's main "
              "claim at its weakest point. Between-patient spread (section 4) "
              "and the depth analysis (section 5) are the two that would most "
              "change how a reader interprets the numbers, and both are "
              "uncomfortable. NMI (section 6) I would reduce to one sentence.")
    para(doc, "Figures are in figures/assets/extra/ as PNG, SVG and PDF. "
              "Numbers come from HER2 and cSCC only; kidney is still running.",
         italic=True, colour=MUTED)

    doc.save(OUT)
    print(f"wrote {OUT}")
    print(f"  {len(doc.paragraphs)} paragraphs, "
          f"{len(doc.tables)} tables, "
          f"{len(glob.glob(str(FIG / '*.png')))} figures available")
    return 0


if __name__ == "__main__":
    sys.exit(main())
