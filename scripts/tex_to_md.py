"""Render the manuscript as commentable markdown.

The .tex is the source of truth; this is a read-only view of it for review. Every
paragraph gets a stable tag (P1, P2, ...) so a comment can point at one without
quoting it, and tables are rebuilt as markdown so the numbers can be checked at
a glance rather than parsed out of LaTeX.

Open questions and TK placeholders are collected at the top, since those are the
things a reader most needs to know before starting.

    python scripts/tex_to_md.py
"""

import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
SRC = Path(r"C:\Users\zheyu\OneDrive\Desktop\SPIE2027\main.tex")
DST = Path(r"C:\Users\zheyu\OneDrive\Desktop\SPIE2027\main_for_comment.md")


def clean(t: str) -> str:
    """LaTeX to plain prose, keeping anything that carries meaning."""
    t = re.sub(r"\\cite\{([^}]*)\}", lambda m: "[" + ", ".join(
        k.strip() for k in m.group(1).split(",")) + "]", t)
    t = re.sub(r"~\\ref\{tab:([^}]*)\}", r" (Table: \1)", t)
    t = re.sub(r"\\ref\{tab:([^}]*)\}", r"Table: \1", t)
    t = re.sub(r"~?\\ref\{[^}]*\}", "", t)
    t = re.sub(r"\\label\{[^}]*\}", "", t)
    t = re.sub(r"\\textbf\{([^}]*)\}", r"**\1**", t)
    t = re.sub(r"\\emph\{([^}]*)\}|\\textit\{([^}]*)\}",
               lambda m: "*" + (m.group(1) or m.group(2)) + "*", t)
    t = re.sub(r"\\mathbf\{([^}]*)\}", r"**\1**", t)
    t = t.replace(r"\%", "%").replace(r"\&", "&").replace(r"\,", " ")
    t = t.replace(r"\ ", " ").replace("{,}", ",").replace("~", " ")
    t = t.replace("--", "–").replace(r"\dots", "...")
    t = re.sub(r"\$([^$]*)\$", lambda m: m.group(1)
               .replace(r"\Delta", "Δ").replace(r"\pm", "±").replace(r"\beta", "β")
               .replace(r"\lfloor", "").replace(r"\rfloor", "")
               .replace(r"\times", "×").replace(r"\leq", "≤")
               .replace(r"\mathbb{R}", "R").replace(r"\text{", "").replace("}", "")
               .replace("{", "").replace("^", "^").replace("_", "_"), t)
    t = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", "", t)
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


def table_to_md(block: str) -> str:
    cap = re.search(r"\\caption\{(.*?)\}\s*\n", block, re.S)
    lab = re.search(r"\\label\{([^}]*)\}", block)
    body = re.search(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}", block, re.S)
    if not body:
        return ""
    out = []
    if lab:
        out.append(f"**Table `{lab.group(1)}`**")
    if cap:
        out.append("*" + clean(cap.group(1)) + "*")
    out.append("")
    rows, first = [], True
    for line in body.group(1).split(r"\\"):
        line = line.replace(r"\hline", "").strip()
        if not line:
            continue
        mc = re.match(r"\\multicolumn\{\d+\}\{[^}]*\}\{(.*)\}$", line.strip())
        if mc:
            rows.append(["— " + clean(mc.group(1)) + " —"])
            continue
        cells = [clean(c) for c in line.split("&")]
        rows.append(cells)
    width = max(len(r) for r in rows)
    for r in rows:
        r += [""] * (width - len(r))
    for i, r in enumerate(rows):
        out.append("| " + " | ".join(r) + " |")
        if i == 0 and first:
            out.append("|" + "---|" * width)
            first = False
    return "\n".join(out)


def main() -> int:
    s = SRC.read_text(encoding="utf-8")
    s = s[s.index(r"\begin{abstract}"):s.index(r"\bibliography{")]
    s = re.sub(r"(?m)^\s*%.*$", "", s)

    tks = len(re.findall(r"\bTK\b", s))
    out = [
        "# Toward Cost-Efficient Spatial Transcriptomics through Active Spot Selection",
        "",
        "> Read-only view of `main.tex` for commenting. Edit the `.tex`, not this file —",
        "> regenerate with `python scripts/tex_to_md.py` after any change.",
        "",
        "**How to comment:** each paragraph is tagged `[P1]`, `[P2]`, … Reference the tag",
        "rather than quoting, e.g. *\"P14 — this overstates it\"*.",
        "",
        "## Status",
        "",
        f"- `TK` placeholders remaining: **{tks}** (Kidney cohort; the unfinished cSCC arms)",
        "- Data behind every number: `stnet_ref/ST-Net/her2_pcc_by_fold.csv`,",
        "  `HER2_spatial_metrics.csv`, `cscc_spatial_metrics.csv`",
        "- Figures: none placed yet. Panel (d) is drawn at",
        "  `figures/assets/panel_d_targets.png`",
        "",
        "## Open questions for the author",
        "",
        "1. The `geometry` package in the preamble overrides `spie.cls` margins",
        "   (`textwidth=6.75in`). SPIE format-checks submissions — remove it?",
        "2. Section 5 *New or breakthrough work* is an abstract-submission field, not",
        "   normally a manuscript section. Keep, or reclaim the space for a figure?",
        "3. The claim that AL beats random \"in classification and segmentation\" is",
        "   supported only by natural-image citations. Add a pathology-AL reference?",
        "4. The paper now claims **three** cohorts, but nothing has been run on",
        "   kidney -- every kidney entry is a `TK`. This has to resolve one way or",
        "   the other before submission: finish the runs, or say two.",
        "5. Cell-type composition is currently out of scope (two targets, not three).",
        "   Confirm.",
        "",
        "---",
        "",
    ]

    i = 0
    # `i` is mutated inside the loop below; declared here so the table branch
    # can advance it too.
    for chunk in re.split(r"(?=\\(?:section|subsection|begin\{abstract\}|begin\{table\}))", s):
        c = chunk.strip()
        if not c:
            continue
        if c.startswith(r"\begin{abstract}"):
            body = c[c.index("}") + 1:c.index(r"\end{abstract}")]
            i += 1
            out += ["## Abstract", "", f"`[P{i}]` " + clean(body), ""]
            kw = re.search(r"\\keywords\{(.*?)\}", c, re.S)
            if kw:
                out += ["**Keywords:** " + clean(kw.group(1)), ""]
            continue
        if c.startswith(r"\begin{table}"):
            # The chunk runs from \begin{table} to the next heading, so it also
            # holds whatever prose follows the table. Emitting only the table
            # silently dropped those paragraphs.
            end = c.index(r"\end{table}") + len(r"\end{table}")
            out += [table_to_md(c[:end]), ""]
            for para in re.split(r"\n\s*\n", c[end:]):
                p = clean(para)
                if len(p) > 2:
                    i += 1
                    out += [f"`[P{i}]` " + p, ""]
            continue
        h = re.match(r"\\(section|subsection)\{(.*?)\}", c)
        if h:
            lvl = "##" if h.group(1) == "section" else "###"
            out += [f"{lvl} {clean(h.group(2))}", ""]
            rest = c[h.end():]
            for para in re.split(r"\n\s*\n", rest):
                p = clean(para)
                if len(p) > 2:
                    i += 1
                    out += [f"`[P{i}]` " + p, ""]
    DST.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {DST}")
    print(f"  {i} tagged paragraphs, {len(re.findall(r'Table `', chr(10).join(out)))} tables")
    return 0


if __name__ == "__main__":
    sys.exit(main())
