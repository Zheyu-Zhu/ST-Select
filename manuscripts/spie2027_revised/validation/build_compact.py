"""Build the eight-page-target revision from prose and audited table inputs."""
from pathlib import Path
import csv
import re

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    with (ROOT / "validation" / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


current = (ROOT / "main.tex").read_text(encoding="utf-8")
expanded = ROOT / "main_expanded_reviewer_revision.tex"
if not expanded.exists():
    expanded.write_text(current, encoding="utf-8")
header = current[:current.index(r"\begin{abstract}")]
abstract = re.search(r"\\begin\{abstract\}.*?\\end\{abstract\}", current, re.S).group(0)
keywords = re.search(r"\\keywords\{[^}]+\}", current).group(0)
body = (ROOT / "validation" / "compact_body.tex").read_text(encoding="utf-8")
budgets = (0.05, 0.1, 0.3, 0.5)
methods = ("random", "coreset", "mc_dropout", "tod", "typiclust")
labels = dict(zip(methods, ("Random", "CoreSet", "MC-dropout", "TOD", "TypiClust-inspired")))
summary = {(r["dataset"], r["sampler"], float(r["budget"])): r for r in read("pcc_summary.csv")}
pooled = {(r["dataset"], float(r["budget"])): r for r in read("pooled_descriptive_pcc.csv")}
spatial = read("spatial_descriptive_summary.csv")

pcc = [r"\begin{table}[htbp]", r"\centering",
       r"\caption{PCC, mean $\pm$ sample SD over four patient-level folds. Bold marks the largest mean within cohort and budget; full-label references use 50 epochs and six-round Random release.}",
       r"\label{tab:pcc}", r"\small", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lcccc}", r"\hline",
       r"Strategy & 5\% & 10\% & 30\% & 50\% \\", r"\hline"]
for dataset, name in (("HER2", "HER2-positive breast cancer"), ("cscc", "cSCC")):
    pcc.append(r"\multicolumn{5}{l}{\emph{" + name + r"}} \\")
    for method in methods:
        cells = []
        for b in budgets:
            row = summary[(dataset, method, b)]
            mean = float(row["mean"])
            best = max(float(summary[(dataset, m, b)]["mean"]) for m in methods)
            value = (r"\mathbf{" + f"{mean:.4f}" + "}") if mean == best else f"{mean:.4f}"
            cells.append("$" + value + r" \pm " + f"{float(row['sample_sd']):.4f}" + "$")
        pcc.append(labels[method] + " & " + " & ".join(cells) + r" \\")
    ref = summary[(dataset, "full_label", 1.0)]
    pcc.append(r"Full label & \multicolumn{4}{c}{$" + f"{float(ref['mean']):.4f}" + r" \pm " + f"{float(ref['sample_sd']):.4f}" + r"$} \\")
    pcc.append(r"\hline")
pcc.extend([r"\end{tabular}", r"\end{table}"])

paired = [r"\begin{table}[htbp]", r"\centering",
          r"\caption{Mean active-minus-Random PCC averaged across four strategies; parentheses give positive strategy--fold differences out of 16. Shared comparators make these descriptive summaries, not independent replicates.}",
          r"\label{tab:paired}", r"\begin{tabular}{lcccc}", r"\hline",
          r"Cohort & 5\% & 10\% & 30\% & 50\% \\", r"\hline"]
for dataset, label in (("HER2", "HER2"), ("cscc", "cSCC")):
    cells = [f"${float(pooled[(dataset,b)]['mean_delta']):+.4f}$ ({pooled[(dataset,b)]['descriptive_strategy_fold_wins']}/16)" for b in budgets]
    paired.append(label + " & " + " & ".join(cells) + r" \\")
paired.extend([r"\hline", r"\end{tabular}", r"\end{table}"])

spatial_index = {(r["dataset"], r["sampler"], float(r["budget"]), r["metric"]): float(r["mean"]) for r in spatial}
spatial_rows = [r"\begin{table}[htbp]", r"\centering",
                r"\caption{Spatial measures, mean over four folds. Bold marks the largest mean per cohort, metric, and budget. ARI compares expression-derived clusters; Moran fidelity correlates measured and predicted gene-wise autocorrelation.}",
                r"\label{tab:spatial}", r"\small", r"\setlength{\tabcolsep}{3pt}",
                r"\begin{tabular}{lcccccccc}", r"\hline",
                r" & \multicolumn{4}{c}{HER2} & \multicolumn{4}{c}{cSCC} \\",
                r"Strategy & 5\% & 10\% & 30\% & 50\% & 5\% & 10\% & 30\% & 50\% \\", r"\hline"]
for metric, metric_label in (("ari", "Expression-cluster agreement (ARI)"), ("moran_fidelity", "Moran's $I$ fidelity")):
    spatial_rows.append(r"\multicolumn{9}{l}{\emph{" + metric_label + r"}} \\")
    for method in methods:
        cells = []
        for dataset in ("HER2", "cscc"):
            for b in budgets:
                value = spatial_index[(dataset, method, b, metric)]
                best = max(spatial_index[(dataset, m, b, metric)] for m in methods)
                cells.append((r"$\mathbf{" + f"{value:.4f}" + "}$") if value == best else f"${value:.4f}$")
        spatial_rows.append(labels[method] + " & " + " & ".join(cells) + r" \\")
    spatial_rows.append(r"\hline")
spatial_rows.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
body = body.replace("%%PCC_TABLE%%", "\n".join(pcc)).replace("%%POOLED_TABLE%%", "\n".join(paired)).replace("%%SPATIAL_TABLE%%", "\n".join(spatial_rows))
(ROOT / "main.tex").write_text(header + abstract + "\n\n" + keywords + "\n\n" + body, encoding="utf-8")
print("Built compact revision with authoritative PCC, paired-summary, and spatial table values.")
