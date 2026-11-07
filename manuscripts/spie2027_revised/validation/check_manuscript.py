"""Check source structure and consistency before rendering the revised manuscript."""
from pathlib import Path
from collections import Counter
import json
import re

ROOT = Path(__file__).resolve().parents[1]
text = (ROOT / "main.tex").read_text(encoding="utf-8")
bib = (ROOT / "main.bib").read_text(encoding="utf-8")
abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, re.S).group(1)
plain = re.sub(r"\\[a-zA-Z]+(?:\[[^\]]*\])?", "", abstract)
plain = re.sub(r"[{}$]", "", plain).replace("~", " ")
abstract_words = len(plain.split())
labels = re.findall(r"\\label\{([^}]+)\}", text)
references = re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", text)
citations = {key.strip() for group in re.findall(r"\\cite(?:\[[^]]*\])?\{([^}]+)\}", text) for key in group.split(",")}
bibkeys = re.findall(r"(?m)^@[a-zA-Z]+\{([^,]+),", bib)
sections = re.findall(r"\\(?:sub)*section\*?\{([^}]+)\}", text)
environments = Counter(re.findall(r"\\begin\{([^}]+)\}", text))
ends = Counter(re.findall(r"\\end\{([^}]+)\}", text))
keywords = re.search(r"\\keywords\{([^}]+)\}", text).group(1).split(",")
report = {
    "abstract_words_after_tex_command_removal": abstract_words,
    "keyword_count": len(keywords),
    "missing_citation_keys": sorted(citations - set(bibkeys)),
    "duplicate_bibkeys": [k for k,v in Counter(bibkeys).items() if v > 1],
    "undefined_references": sorted(set(references) - set(labels)),
    "duplicate_labels": [k for k,v in Counter(labels).items() if v > 1],
    "duplicate_section_titles": [k for k,v in Counter(sections).items() if v > 1],
    "environment_begin_end_match": environments == ends,
    "table_count": environments["table"],
    "figure_count": environments["figure"],
    "kidney_result_mention": bool(re.search(r"Kidney|47/48|47 of 48", text)),
    "acknowledgements_unnumbered": r"\section*{ACKNOWLEDGMENTS}" in text,
    "obsolete_epoch_lr_formula_present": r"\frac{t}{T_{\max}}" in text,
    "obsolete_0056_present": "0.0056" in text,
    "obsolete_below_25_percent_claim_present": "below 2.5" in text,
}
failures = []
if abstract_words > 250:
    failures.append("abstract exceeds 250 words")
if len(keywords) > 8:
    failures.append("too many keywords")
for key in ("missing_citation_keys", "duplicate_bibkeys", "undefined_references", "duplicate_labels", "duplicate_section_titles"):
    if report[key]:
        failures.append(key)
for key in ("kidney_result_mention", "obsolete_epoch_lr_formula_present", "obsolete_0056_present", "obsolete_below_25_percent_claim_present"):
    if report[key]:
        failures.append(key)
if not report["environment_begin_end_match"] or not report["acknowledgements_unnumbered"]:
    failures.append("structure")
report["failures"] = failures
(ROOT / "validation" / "manuscript_source_checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
if failures:
    raise SystemExit(1)
