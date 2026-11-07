"""Verify final PDF/source constraints and package the professor-review deliverables."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import zipfile

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
OUT = REPO / "output" / "spie2027_revision"
OUT.mkdir(parents=True, exist_ok=True)
source_checks = json.loads((ROOT / "validation" / "manuscript_source_checks.json").read_text(encoding="utf-8"))
pdf = pymupdf.open(ROOT / "main.pdf")
checks = {
    "manuscript_pages_including_references": len(pdf),
    "working_page_cap_including_references": 8,
    "professor_deadline": "2026-11-15",
    "public_spie_deadline": "2027-01-27",
    "official_page_cap_confirmed_in_portal": False,
    "letter_page_dimensions": all(abs(p.rect.width-612)<0.01 and abs(p.rect.height-792)<0.01 for p in pdf),
    "all_fonts_embedded": all(f[1] != "n/a" for p in pdf for f in p.get_fonts()),
    "no_overfull_boxes": "Overfull" not in (ROOT / "main.log").read_text(encoding="utf-8", errors="replace"),
    "no_undefined_references": "undefined references" not in (ROOT / "main.log").read_text(encoding="utf-8", errors="replace"),
    "no_bibtex_warnings": "Warning--" not in (ROOT / "main.blg").read_text(encoding="utf-8", errors="replace"),
    "source_checks_pass": not source_checks["failures"],
    "abstract_word_count_check": source_checks["abstract_words_after_tex_command_removal"],
    "keyword_count": source_checks["keyword_count"],
    "tables": source_checks["table_count"],
    "figures": source_checks["figure_count"],
    "main_pdf_sha256": hashlib.sha256((ROOT / "main.pdf").read_bytes()).hexdigest(),
    "submitted_or_sent": False,
}
assert len(pdf) <= 8, checks
assert all(checks[k] for k in ("letter_page_dimensions", "all_fonts_embedded", "no_overfull_boxes",
                              "no_undefined_references", "no_bibtex_warnings", "source_checks_pass")), checks
(ROOT / "validation" / "final_delivery_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
shutil.copy2(ROOT / "main.pdf", OUT / "ActiveST_Bench_revised_full.pdf")
shutil.copy2(ROOT / "response_to_reviewers.md", OUT / "response_to_reviewers.md")

root_files = ["main.tex", "main.bib", "spie.cls", "spiebib.bst", "main.bbl", "README.md",
              "response_to_reviewers.md", "submission_requirements_notes.md", "bibliography_audit.md"]
archive_path = OUT / "ActiveST_Bench_revised_source_and_validation.zip"
excluded = {"update_tables.py"}  # intermediate-layout helper, not the final build
with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
    for name in root_files:
        archive.write(ROOT / name, name)
    for folder in ("Figure", "validation"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.name not in excluded:
                archive.write(path, path.relative_to(ROOT).as_posix())
    archive.write(ROOT / "main_original.tex", "originals/main_original.tex")
with zipfile.ZipFile(archive_path) as archive:
    assert archive.testzip() is None
    assert archive.read("main.tex") == (ROOT / "main.tex").read_bytes()
    assert "Figure/Image1.pdf" in archive.namelist() and "Figure/Image2.pdf" in archive.namelist()
print(json.dumps(checks, indent=2))
print("Deliverables: " + str(OUT))
