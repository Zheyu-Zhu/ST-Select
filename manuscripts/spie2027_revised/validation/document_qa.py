"""Render manuscript/figure PDFs and record layout/text checks without changing PDFs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pymupdf as fitz


def inspect_pdf(path: Path, out: Path, dpi: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(path)
    pages = []
    for number, page in enumerate(doc, 1):
        page.get_pixmap(dpi=dpi, alpha=False).save(out / f"page_{number:02d}.png")
        blocks = page.get_text("blocks")
        pages.append({
            "page": number,
            "width_pt": page.rect.width,
            "height_pt": page.rect.height,
            "text_bounds": [min(b[0] for b in blocks), min(b[1] for b in blocks),
                            max(b[2] for b in blocks), max(b[3] for b in blocks)] if blocks else None,
            "words": len(page.get_text("words")),
            "fonts": sorted({str(f[3]) for f in page.get_fonts()}),
            "unembedded_fonts": [list(f[:5]) for f in page.get_fonts() if f[1] == "n/a"],
        })
    text = "\n\n".join(f"--- PAGE {i+1} ---\n{p.get_text()}" for i, p in enumerate(doc))
    (out / "extracted_text.txt").write_text(text, encoding="utf-8")
    summary = {"source": str(path.resolve()), "n_pages": len(doc), "pages": pages}
    (out / "pdf_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"source": summary["source"], "n_pages": len(doc),
                      "pages": [{"page": p["page"], "words": p["words"],
                                 "text_bounds": p["text_bounds"],
                                 "unembedded_fonts": p["unembedded_fonts"]} for p in pages]}, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()
    inspect_pdf(args.pdf, args.out, args.dpi)
