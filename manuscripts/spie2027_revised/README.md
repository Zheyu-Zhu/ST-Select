# SPIE 14525-21: revised full manuscript

Prepared 7 November 2026 for the author's **15 November 2026 professor-review deadline**. The user confirmed HER2/cSCC-only conference scope, with kidney reserved for the journal extension. The working limit is **at most eight pages total, including references**; the user's recollection of this limit has not been independently confirmed in the author portal.

## Review files

- `main.pdf`: revised full manuscript; see `validation/final_delivery_checks.json` for its verified page count and checks.
- `main.tex`, `main.bib`, `spie.cls`, `spiebib.bst`, `Figure/`: editable LaTeX project, suitable for an Overleaf upload.
- `response_to_reviewers.md`: point-by-point response and transparent list of unresolved concerns.
- `submission_requirements_notes.md`: official-source links, public deadlines, and unresolved portal requirements.
- `validation/`: numerical results, provenance, and reproducible analysis scripts.

The source archives in Downloads and accepted four-page submission were not modified. The local working directory preserves `main_original.tex` and an intermediate longer revision. The distribution ZIP includes the original at `originals/main_original.tex`. Only `main.tex` is the final editable manuscript; the compiled PDF accompanies the source ZIP separately.

## Material findings of validation

HER2 targets equal `log1p(raw counts)`. cSCC targets instead reconstruct as `log1p(10000 * count / total)` over 13,251 raw genes common to its slides. This matches all cSCC target values within 2.4e-7. The paper corrects the prior claim that both cohorts lacked library normalization.

The actual learning-rate decay is indexed by optimizer updates with denominator 1e8, not by epochs. Saved events confirm it. Nineteen HER2 SD entries and the pooled HER2 30% rounding were corrected. The high-budget “below 2.5%” learning-curve claim was replaced by audited ranges. Equal-epoch full-label comparisons are explicitly not optimizer-step matched.

The added statistical analysis is an exploratory corrected-CV sensitivity calculation. It does not repair the missing independent seeds or prove equivalence. No new training, physical-resolution normalization experiment, model architecture change, or kidney inclusion was performed.

## Rebuild and validation

Existing analysis scripts use the documented local data/output paths and need numpy, pandas, scipy, h5py, and Pillow as described in their own notes. They read data/results and write only validation outputs.

After editing `validation/compact_body.tex`, run `validation/build_compact.py` to regenerate `main.tex` with table values from audited CSVs. This overwrites the generated manuscript; edit `main.tex` directly if you do not intend to regenerate.

Compile `main.tex` with a normal LaTeX/BibTeX toolchain, or `tectonic --keep-logs main.tex`. The review environment used a temporary portable Tectonic executable; it did not install a system-wide TeX distribution. `validation/check_manuscript.py` checks source structure and abstract length. `validation/document_qa.py` renders the PDF using PyMuPDF and records page dimensions, font embedding, and text bounds.

## Submission status and remaining checks

This is a professor-review package. Nothing was uploaded, submitted, registered, or emailed. November 15 is the user's internal handoff deadline. The public SPIE Medical Imaging 2027 manuscript deadline is January 27, 2027, separately corroborated by SPIE's official award instructions linked in the requirements notes. Any paper-specific deadline, current proceedings page/file-size cap, registration prerequisite, author/funding approval, and required publication license must still be checked in the author's SPIE portal before submission.
