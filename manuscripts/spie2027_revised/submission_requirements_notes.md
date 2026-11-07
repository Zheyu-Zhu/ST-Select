# SPIE Medical Imaging 2027: requirements and editorial notes

Checked 7 November 2026 (America/New_York). Research notes only: nothing was submitted, registered, uploaded, or sent to SPIE. The accepted paper identifier 14525-21 comes from the user's supplied correspondence; its conference mapping and personalized portal instructions have not been independently verified.

## Author's internal deadline and working page constraint

The author clarified during revision that the professor-review draft is due **15 November 2026** and recalled an **eight-page full-paper limit**, without confirming the portal text. The revised manuscript is prepared to fit **at most eight pages total, including references**, without changing standard margins or body font size. This internal handoff date is distinct from the public SPIE proceedings deadline. The portal's exact full-paper limit remains unverified; see the final validation report for the compiled document's actual page count.

## Public deadlines and submission stages

The [Medical Imaging abstract submission guidelines](https://spie.org/conferences-and-exhibitions/medical-imaging/presenters/abstract-submission-guidelines) list: manuscript/poster upload opens 14 December 2026; poster PDFs due 20 January 2027; proceedings manuscripts due 27 January 2027; advance oral-slide upload due 12 February 2027 (later slides go to onsite Speaker Check-in). The contact author or speaker must register before uploading. These are public event dates, not a verified paper-specific revision deadline.

The 27 January 2027 final-manuscript deadline is independently confirmed by SPIE's own [2027 Image Processing student award page](https://spie.smapply.io/prog/spie_medical_imaging_2027_image_processing_student_paper_award/). That award page concerns its own conference/eligibility; it is corroboration of the event deadline, not evidence that paper 14525-21 belongs to that conference or is award-eligible.

The accepted 2–4-page review supplement and the final archival proceedings manuscript are different stages. The 2–4-page limit excludes acknowledgements and references. Do not impose that limit on the full manuscript without a current proceedings requirement.

Access caveat: direct spie.org event pages returned an anti-bot iframe in this session. The browser search index exposed the 2027 abstract-guideline content under SPIE's spiecareercenter.org hostname; live page retrieval did not succeed. Public January 27 corroboration from the SPIE-hosted award site was readable. Recheck the author's portal for paper-specific deadlines, presentation type/time, any revised upload instructions, and the current full-paper length limit. No precise deadline time zone was verified for the proceedings deadline.

## Formatting supported by the official SPIE template

The [SPIE proceedings manuscript sample, Letter size](https://spie.org/Documents/Publications/ProcSPIETemplate_Letter.pdf) specifies a self-contained single-paragraph abstract of at most 250 words; up to eight keywords; one column; no running headers, footers, or page numbers; numbered references in citation order; standard embedded fonts. Letter margins are 1.0 inch top, 1.25 bottom, and approximately 0.88 left/right. Use the SPIE class/template rather than independently compressing margins.

Its style table gives a 16-point bold centered sentence-case title, 12-point centered author/affiliation text, 10-point body text, 11-point bold centered section headings, and 9-point captions. Acknowledgements and References are unnumbered. The current title uses title case; the source also has two consecutive Discussion sections. Correct these during manuscript cleanup.

Unresolved: a current Medical Imaging 2027 proceedings page minimum/maximum and PDF upload size limit were not verified from accessible primary sources. Do not substitute historical generic SPIE page limits or another conference's limits. The [event manuscript guidelines](https://spie.org/conferences-and-exhibitions/medical-imaging/presenters/manuscript-submission-guidelines) and paper-specific portal are the intended places to confirm these.

## Independent editorial recommendations for the reviewer issues

### Negative result and definite, bounded conclusion

Use a direct conclusion: in the evaluated two-cohort, fixed-predictor, fixed-horizon benchmark, the four active rules do not deliver a consistent low-budget PCC improvement over uniform Random; all four cohort-mean differences are negative at 10% on both cohorts. This is a useful observed limitation of these implementations and setting. It is not proof that all active learning fails or that Random is equivalent/noninferior to the alternatives.

Do not use lack of significance as proof of equivalence, promise sequencing-cost savings, define a universal budget threshold, or infer a successful assay from retrospective masking. Explain why Random remains the necessary reference and why pointwise and spatial criteria must both be reported. Separate the observed low-budget result from explanations that have not been tested. Kidney stays outside this revision unless explicitly added by the lead author.

The supplied full manuscript's conclusion also generalizes positive 30% differences too broadly: its cSCC table has three of four active means below Random at 30%. Limit the positive 30% statement to HER2, or give cohort-specific statements. The manuscript's all-four-fold PCC/ARI reversal wording likewise needs correction: the all-fold consistency is in ARI; PCC has positive-fold exceptions.

### Resolution criticism

Separate H&E pixel resolution, H&E patch field of view, and the ST measurement grid's spatial resolution. A 224-pixel patch does not harmonize physical context: the reported H&E scales imply approximately 148–212 micrometers per side on HER2 and 122–124 micrometers on cSCC (224 multiplied by micrometers/pixel). The stored patches are already 224 by 224, so the current Resize does not perform resolution harmonization.

State that the study covers two legacy array cohorts, that each expression label aggregates a capture spot, and that performance/strategy rankings have not been tested on higher-resolution platforms or after harmonizing physical field of view. Do not assert that resolution caused the negative result, that spatial redundancy was established by patch size, or that results transfer to single-cell/subcellular platforms.

### Single run and statistics criticism

Describe the actual design as one training/acquisition seed per method-budget-fold with four patient-level folds, not a single model overall and not replicated seed robustness. Fold SD is variation across held-out folds; it is not seed uncertainty. Report paired fold differences and counts transparently, together with the limitation that four overlapping folds, shared Random comparators, and a single seed cannot separate acquisition and training randomness.

Adding pooled strategy-fold p-values does not fix this design. Any patient-resampling sensitivity must identify the patient as the resampling unit, preserve paired method predictions, and disclose the small patient count and remaining conditional-on-training limitation. Reanalysis cannot replace independently trained seeds. If repeat training is not completed, the response should acknowledge the unresolved reproducibility limitation plainly and keep the conclusion descriptive and bounded.

### Other source issues already identified

- TypiClust still reports 71–1,584 seconds per round, including the load-contaminated timings identified in the handoff. Use the audited clean comparison and its conditions.
- At HER2 5% Moran fidelity, Random is bolded even though the displayed TypiClust mean is larger.
- Source has consecutive `Discussion` and `Discussion and Future Work` section commands.
- Full-label references are already included in the full manuscript. They are not a new journal contribution; an audited schedule-matched analysis would be an additional analysis.

These are recommendations for the lead agent's edits. No manuscript or response text was modified by this review.
