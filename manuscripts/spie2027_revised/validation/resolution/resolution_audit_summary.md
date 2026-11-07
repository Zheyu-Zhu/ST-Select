# Resolution audit

This is a post hoc description of existing seed-42 predictions, not a scale-normalization experiment.

- HER2: 36 slides / 8 patients; all 13620 referenced PNG headers are 224x224. Native mpp 0.662118–0.946333; field width 148.314–211.979 um.
  Patient grouping accounts for 99.2% of observed slide-resolution variance. The nearest-center Euclidean distance is below 224 native pixels for 2108 spots (15.48%); axis-aligned square intersection occurs for 2124 spots (15.59%).
- cscc: 12 slides / 4 patients; all 8671 referenced PNG headers are 224x224. Native mpp 0.545403–0.553919; field width 122.170–124.078 um.
  Patient grouping accounts for 10.9% of observed slide-resolution variance. The nearest-center Euclidean distance is below 224 native pixels for 0 spots (0.00%); axis-aligned square intersection occurs for 8670 spots (99.99%).

Available local WSIs permitted 12 spot-crop checks across 4 HER2 slides: every checked training PNG exactly matches a 224x224 level-0 WSI crop centered at integer-truncated HEST coordinates. Only intersecting TIFF tiles were decompressed. cSCC original WSIs are not present locally, so its native scale is supported by metadata and package correspondence rather than an independent WSI pixel match.

The downloadable HEST .h5 patches have separate patch_size_src/target/downsample attributes; they are resampled and are not the network PNGs. Multiplying their target width by native mpp would describe the wrong product. The trainer's 224-to-224 resize is a no-op for the verified PNGs; no physical-field normalization experiment was performed.

## Target preprocessing finding

HER2 targets match log1p(raw HEST X). cSCC targets reconstruct as log1p(10000 * c_ig / sum_{h in J} c_ih), where J is the intersection of HEST var names across all 12 cSCC slides (13,251 genes). The formula is checked over every one of 8,671 spots and all 300 model targets. This empirically establishes library-size scaling over the shared gene universe; the original producing script has not been recovered. cSCC does not exactly match normalization by HEST obs.total_counts because that field includes genes outside J. Do not describe both cohorts as unnormalized log-raw targets. Every saved Random-50% ground-truth vector matches the processed targets. Row correspondence is verified by all 300 entries and shared zero support in input order, avoiding ambiguous nearest-neighbor matching for duplicate target vectors. See target_preprocessing_checks.csv, cscc_spot_normalization_checks.csv and the released common-gene list. All-zero target rows cannot identify a fitted factor and are separately counted.

## Descriptive resolution strata

Patients are split at the within-cohort median of patient-average native mpp, assigning every slide from a patient to one group. PCC and active-minus-Random differences are averaged first across each patient's slides, then equally across patients in the stratum. This differs from the paper's fold aggregation and is explicitly exploratory. Only seed 42 is available for these cohorts. No significance tests or independent-slide assumptions are used. Groups share overlapping CV training sets; resolution is associated with patient, acquisition and tissue characteristics, so these comparisons cannot establish a causal resolution effect or remove scale confounding.

| Cohort | Budget | Strategy | Lower-mpp patient mean delta | Higher-mpp patient mean delta |
|---|---:|---|---:|---:|
| HER2 | 5% | coreset | -0.0359 | -0.0394 |
| HER2 | 5% | mc_dropout | -0.0306 | -0.0233 |
| HER2 | 5% | tod | -0.0029 | -0.0052 |
| HER2 | 5% | typiclust | -0.0079 | +0.0080 |
| HER2 | 10% | coreset | -0.0158 | -0.0208 |
| HER2 | 10% | mc_dropout | -0.0146 | -0.0120 |
| HER2 | 10% | tod | -0.0048 | -0.0091 |
| HER2 | 10% | typiclust | -0.0102 | -0.0054 |
| HER2 | 30% | coreset | -0.0002 | +0.0110 |
| HER2 | 30% | mc_dropout | -0.0021 | +0.0028 |
| HER2 | 30% | tod | +0.0023 | +0.0069 |
| HER2 | 30% | typiclust | +0.0012 | +0.0076 |
| HER2 | 50% | coreset | -0.0006 | +0.0071 |
| HER2 | 50% | mc_dropout | -0.0015 | +0.0114 |
| HER2 | 50% | tod | +0.0005 | +0.0057 |
| HER2 | 50% | typiclust | -0.0016 | +0.0082 |
| cscc | 5% | coreset | -0.0424 | +0.0000 |
| cscc | 5% | mc_dropout | -0.0342 | +0.0050 |
| cscc | 5% | tod | -0.0113 | +0.0044 |
| cscc | 5% | typiclust | +0.0043 | +0.0026 |
| cscc | 10% | coreset | -0.0240 | +0.0069 |
| cscc | 10% | mc_dropout | -0.0245 | -0.0081 |
| cscc | 10% | tod | -0.0165 | +0.0035 |
| cscc | 10% | typiclust | -0.0017 | -0.0050 |
| cscc | 30% | coreset | -0.0049 | -0.0021 |
| cscc | 30% | mc_dropout | -0.0017 | -0.0087 |
| cscc | 30% | tod | +0.0076 | -0.0022 |
| cscc | 30% | typiclust | +0.0010 | -0.0019 |
| cscc | 50% | coreset | -0.0052 | +0.0057 |
| cscc | 50% | mc_dropout | -0.0005 | +0.0059 |
| cscc | 50% | tod | +0.0055 | +0.0031 |
| cscc | 50% | typiclust | +0.0030 | +0.0049 |

## Rebuttal boundary

The observed PNG scale variation is now reported in physical units and evaluated descriptively using existing predictions. It is not experimentally separated from patient/cohort effects. A future controlled study should compare a shared physical field of view with explicitly documented resampling and a fixed pixel-input control; it must rerun both selection and predictor training. No such experiment has been performed here. Morphological redundancy and patch overlap are distinct, and neither by itself establishes the mechanism of acquisition rankings.

## Code facts

Current samplers._features takes the 1024-dimensional DenseNet trunk output after ReLU and global average pooling and before model.classifier. The separate classifier is Linear(1024,1024), ReLU, Dropout(0.3), Linear(1024,300). Thus '1024-dimensional features immediately before the regression head' is accurate; describing acquisition features as the added head's output is not.

The beta=1 scripts use warmup=5, interval=3, cutoff=20 with the same trainer, so labels accumulate at displayed epochs 1,6,9,12,15,18. beta=1 is a nominal full-label reference with six integer-floor quotas and can omit up to five spots. It is neither all labels from epoch 1 nor optimization-matched to lower-budget runs. These are read-only code findings, not changes to training.

Reproduce: run audit_resolution.py using the st-select Python environment. Dependencies: numpy, pandas, scipy, h5py, Pillow, tifffile, zarr. Input roots are configurable. Outputs are written only beside this script.
