# Targeted bibliography audit: five BibTeX warnings

Checked 7 November 2026. Scope: `zhu2024asign`, `pang2021leveraging`, `zhu2025magnet`, `zhu2026duet`, and `zhu2025scr2` only. No bibliography entries were changed by this audit. Missing-volume warnings for preprints are formatting warnings, not evidence of nonexistent or incorrect references.

Three references now have verified conference-proceedings records. No journal replacement was verified for any of these five. The other two remain supported preprint citations; the search did not establish that a later publication is impossible or absent from every index.

## Suggested updates

### `zhu2024asign`: prefer the published CVPR 2025 record

The [official CVF proceedings page](https://openaccess.thecvf.com/content/CVPR2025/html/Zhu_ASIGN_An_Anatomy-aware_Spatial_Imputation_Graphic_Network_for_3D_Spatial_CVPR_2025_paper.html) supplies the final record and BibTeX: CVPR 2025, pages 30829–30838. The existing title and complete twelve-author list match. In particular, the publisher's title really uses **Graphic Network**; do not change it to “Graph Network” based on wording in the abstract.

Keep the local key if desired, but change the entry from `@article` to `@inproceedings`, replace `journal` with `booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}`, set `year = {2025}`, and add `pages = {30829--30838}` and the official URL. No journal volume is needed. The [2024 arXiv record](https://arxiv.org/abs/2412.03026) is valid; the change is an update to the published version, not repair of a broken source.

### `pang2021leveraging`: retain the bioRxiv preprint, add its identifier

The [bioRxiv record](https://www.biorxiv.org/content/10.1101/2021.11.28.470212v1) matches all three authors, the full existing title, and the 2021 date. Add `doi = {10.1101/2021.11.28.470212}` and `url = {https://doi.org/10.1101/2021.11.28.470212}`. The current `note = {HisToGene}` is a descriptive model name, not journal metadata.

No later publisher record was verified in the targeted search. Do not invent a volume, issue, or journal to silence the warning. If the current bibliography style still requests volume, retain and explain the warning or use a deliberate preprint-compatible entry type; verify the rendered reference after any style change.

### `zhu2025magnet`: prefer the PMLR record, with publisher year 2026

The [official PMLR record](https://proceedings.mlr.press/v301/zhu26b.html) identifies the paper in Proceedings of The 8th International Conference on Medical Imaging with Deep Learning, **PMLR 301:1943–1955 (2026)**. This is the proceedings of the **2025 meeting**, but the publisher's citation year is **2026**; retain that distinction. The title matches apart from the existing loss of the method's internal capitalization. The existing first ten authors match the publisher list in order.

Change to `@inproceedings`; use the publisher's `booktitle`, `series = {Proceedings of Machine Learning Research}`, `volume = {301}`, `pages = {1943--1955}`, `year = {2026}`, `publisher = {PMLR}`, and the official URL. Protect `{MagNet}` in the title. The final five authors after Mengmeng Yin are Yu Wang, Shilin Zhao, Yaohong Wang, Haichun Yang, and Yuankai Huo; the publisher's BibTeX provides the complete list. The existing `and others` truncation is not an incorrect author claim.

### `zhu2026duet`: retain the verified 2026 arXiv record

The [arXiv record](https://arxiv.org/abs/2605.14104) matches the title, year, and the first ten listed authors in order. It lists fourteen authors in total; the remaining four are Yu Wang, Shilin Zhao, Haichun Yang, and Yuankai Huo. The visible record has no journal-reference field. No later publisher record was verified in the targeted search.

Keep it explicitly identified as a preprint, preserve `{DUET}` capitalization, and add `doi = {10.48550/arXiv.2605.14104}` and `url = {https://arxiv.org/abs/2605.14104}` if desired. Do not add fictitious volume or pages solely to satisfy an `@article` warning.

### `zhu2025scr2`: prefer the published MIDL 2026 record

The [official PMLR record](https://proceedings.mlr.press/v315/zhu26a.html) identifies the paper in Proceedings of The 9th International Conference on Medical Imaging with Deep Learning, **PMLR 315:488–509 (2026)**. Its title and the existing first ten author names match; the title renders the method as SCR squared-ST. The [arXiv source](https://arxiv.org/abs/2512.13635) is valid, initially posted in December 2025 and revised January 2026.

Change to `@inproceedings`; use the publisher's `booktitle`, `series = {Proceedings of Machine Learning Research}`, `volume = {315}`, `pages = {488--509}`, `year = {2026}`, `publisher = {PMLR}`, and the official URL. Preserve the method's squared notation with a suitable protected TeX title. Remaining authors after Marilyn Lionts are Yuechen Yang, Yalin Zheng, Yu Wang, Shilin Zhao, Haichun Yang, and Yuankai Huo. The publisher's BibTeX can supply the complete list. Keep the current local key to avoid unnecessary manuscript citation edits.

## Scope and access limits

This was not an exhaustive citation audit. Metadata was checked against the listed publisher/preprint pages and publisher-provided citation exports. Some direct CVF/bioRxiv requests returned HTTP 403; their official search-indexed records supplied the cited metadata. PMLR paper pages and arXiv records were readable directly. No DOI was guessed for the proceedings entries. No scientific claim was assessed as supported solely because its bibliography record exists.
