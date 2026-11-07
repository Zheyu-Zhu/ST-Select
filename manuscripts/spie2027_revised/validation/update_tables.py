"""Regenerate revised PCC table cells from the audited, unrounded CSVs."""
from pathlib import Path
import csv
import re

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "main.tex"
BUDGETS = (0.05, 0.1, 0.3, 0.5)
METHODS = ("random", "coreset", "mc_dropout", "tod", "typiclust")
LABELS = {
    "random": "Random",
    "coreset": r"CoreSet~\cite{sener2018active}",
    "mc_dropout": r"MC-dropout~\cite{gal2016dropout}",
    "tod": r"TOD~\cite{huang2021semi}",
    "typiclust": r"TypiClust-inspired~\cite{hacohen2022active}",
}


def read(name):
    with (ROOT / "validation" / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


summary = {(r["dataset"], r["sampler"], float(r["budget"])): r for r in read("pcc_summary.csv")}
paired = {(r["dataset"], r["sampler"], float(r["budget"])): r for r in read("paired_pcc_exploratory.csv")}
text = PATH.read_text(encoding="utf-8")
for dataset, label in (("HER2", "pcc_her2"), ("cscc", "pcc_cscc")):
    pattern = re.compile(r"(\\begin\{table\}.*?\\label\{tab:" + label + r"\}.*?)(\\end\{table\})", re.S)
    # Avoid spanning earlier tables by locating the label, then its enclosing table.
    position = text.index(r"\label{tab:" + label + "}")
    start = text.rfind(r"\begin{table}", 0, position)
    end = text.index(r"\end{table}", position) + len(r"\end{table}")
    table = text[start:end]
    for method in METHODS:
        cells = []
        for b in BUDGETS:
            row = summary[(dataset, method, b)]
            mean = float(row["mean"])
            sd = float(row["sample_sd"])
            best = max(float(summary[(dataset, m, b)]["mean"]) for m in METHODS)
            number = (r"\mathbf{" + f"{mean:.4f}" + "}") if mean == best else f"{mean:.4f}"
            cells.append("$" + number + r" \pm " + f"{sd:.4f}" + "$")
        if dataset == "HER2":
            cells.append({"random": "$<1$\\,s", "coreset": "$11$--$12$\\,s", "mc_dropout": "$37$--$42$\\,s",
                          "tod": "$7$--$8$\\,s", "typiclust": "$71$--$83$\\,s"}[method])
        new = LABELS[method] + " & " + " & ".join(cells) + r" \\"
        prefix = re.escape(LABELS[method])
        table, n = re.subn(r"(?m)^" + prefix + r" & .*?\\\\$", lambda _: new, table)
        if n != 1:
            raise RuntimeError(f"Expected one {dataset}/{method} row, found {n}")
    text = text[:start] + table + text[end:]

pooled = {(r["dataset"], float(r["budget"])): r for r in read("pooled_descriptive_pcc.csv")}
rows = [r"\begin{table}[htbp]", r"\centering",
        r"\caption{Paired PCC difference from Random; parentheses give positive folds out of four. Pooled rows are descriptive because strategies share the Random comparator.}",
        r"\label{tab:paired}", r"\begin{tabular}{lcccc}", r"\hline",
        r"Strategy & 5\% & 10\% & 30\% & 50\% \\", r"\hline"]
for dataset, name in (("HER2", "HER2-positive breast cancer"), ("cscc", "cSCC")):
    rows.append(r"\multicolumn{5}{l}{\emph{" + name + r"}} \\")
    for method in METHODS[1:]:
        cells = []
        for b in BUDGETS:
            row = paired[(dataset, method, b)]
            cells.append(f"${float(row['delta_mean']):+.4f}$ ({int(row['wins'])}/4)")
        rows.append(LABELS[method] + " & " + " & ".join(cells) + r" \\")
    rows.append("Pooled mean $\\Delta$ & " + " & ".join(f"${float(pooled[(dataset,b)]['mean_delta']):+.4f}$" for b in BUDGETS) + r" \\")
    rows.append("Pooled positive & " + " & ".join(pooled[(dataset,b)]["descriptive_strategy_fold_wins"] + "/16" for b in BUDGETS) + r" \\")
    rows.append(r"\hline")
rows.extend([r"\end{tabular}", r"\end{table}"])
position = text.index(r"\label{tab:paired}")
start = text.rfind(r"\begin{table}", 0, position)
end = text.index(r"\end{table}", position) + len(r"\end{table}")
text = text[:start] + "\n".join(rows) + text[end:]
PATH.write_text(text, encoding="utf-8")
print("Updated 40 PCC mean/SD cells, timing ranges, and 32 paired PCC cells from audited data.")
