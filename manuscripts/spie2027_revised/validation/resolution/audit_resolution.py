"""Read-only input-scale audit and descriptive patient-grouped PCC analysis.

Run with the project's st-select Python environment. Only writes beside this file.
No model imports, inference, training, WSI-wide pixel reads, or external mutations.
HEST .h5 patch images are a different, resampled product from training PNGs.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import cKDTree

FINAL = {0.05: 110, 0.1: 80, 0.3: 70, 0.5: 60}
STRATEGIES = ["random", "coreset", "mc_dropout", "tod", "typiclust"]


def read_case_maps(training: Path):
    """Read constants as literals; do not import the training package."""
    tree = ast.parse((training / "cross_fold.py").read_text(encoding="utf-8"))
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                    "HER2_6LAYER", "HER2_3LAYER", "CSCC_CASES"
                }:
                    constants[target.id] = ast.literal_eval(node.value)
    return {
        "HER2": {**constants["HER2_6LAYER"], **constants["HER2_3LAYER"]},
        "cscc": constants["CSCC_CASES"],
    }


def decode(values):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def selected_x_columns(h5, columns):
    """Small column-subset read, supporting dense and CSR AnnData X."""
    x = h5["X"]
    if isinstance(x, h5py.Dataset):
        order = np.argsort(columns)
        return x[:, np.asarray(columns)[order]][:, np.argsort(order)]
    from scipy.sparse import csr_matrix
    shape = tuple(x.attrs["shape"])
    matrix = csr_matrix((x["data"][:], x["indices"][:], x["indptr"][:]), shape=shape)
    return matrix[:, columns].toarray()


def wsi_spot_check(repo, patch_root, slide, records, xy):
    path = repo / "hest_data" / "wsis" / f"{slide}.tif"
    if not path.exists():
        return []
    import tifffile
    import zarr

    tiff = tifffile.TiffFile(path)
    # Read page 0 directly: some source pyramids have malformed series metadata.
    store = tiff.pages[0].aszarr()
    image = zarr.open(store, mode="r")
    rows = []
    try:
        for i in np.unique(np.linspace(0, len(records) - 1, 3).astype(int)):
            record = records[i]
            x, y = np.asarray(xy[i], dtype=int)
            if min(x, y) < 112 or x + 112 > image.shape[1] or y + 112 > image.shape[0]:
                continue
            # Zarr only decompresses intersecting tiles; never reads the WSI array.
            crop = image[y - 112:y + 112, x - 112:x + 112]
            with Image.open(patch_root / slide / f"{record['patch_id']}.png") as im:
                png = np.asarray(im.convert("RGB"))
            equal = bool(np.array_equal(png, crop))
            rows.append(dict(slide=slide, record_index=int(i), x=int(x), y=int(y),
                             pixel_exact_native_level0_crop=equal,
                             mean_absolute_pixel_difference=float(np.abs(
                                 png.astype(float) - crop.astype(float)).mean())))
    finally:
        store.close()
        tiff.close()
    return rows


def correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and np.std(x) > 0 and np.std(y) > 0 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path,
                        default=Path(__file__).resolve().parents[4])
    parser.add_argument("--training", type=Path,
                        default=Path(r"C:\Users\zheyu\workspace\stnet_ref\ST-Net"))
    parser.add_argument("--target-gene-dir", type=Path,
                        default=Path(r"C:\Users\zheyu\OneDrive\Desktop\random\random_clean\target_genes"))
    args = parser.parse_args()
    out = Path(__file__).resolve().parent
    repo, training = args.repo, args.training
    package = training.parent
    cases = read_case_maps(training)
    metadata_path = repo / "hest_data" / "HEST_v1_3_0.csv"
    metadata = pd.read_csv(metadata_path).set_index("id", verify_integrity=True)
    cscc_gene_sets = []
    for slides in cases["cscc"].values():
        for slide in slides:
            with h5py.File(repo / "hest_data" / "st" / f"{slide}.h5ad", "r") as h5:
                cscc_gene_sets.append(set(decode(h5["var/_index"][:])))
    cscc_common_genes = set.intersection(*cscc_gene_sets)
    (out / "cscc_common_gene_panel_13251.txt").write_text(
        "\n".join(sorted(cscc_common_genes)) + "\n", encoding="utf-8")
    rows, crop_checks, target_checks, scale_rows = [], [], [], []
    for dataset, patients in cases.items():
        target_names = (args.target_gene_dir / f"{dataset}_target_genes_300.txt").read_text(
            encoding="utf-8").splitlines()
        target_names = [g.strip() for g in target_names if g.strip()]
        assert len(target_names) == 300 and target_names == sorted(target_names)
        for patient, slides in patients.items():
            for slide in slides:
                records = np.load(package / "data_infor" / dataset / patient / f"{slide}.npy",
                                  allow_pickle=True)
                target = np.stack([r["expression_norm"][:300] for r in records])
                meta = metadata.loc[slide]
                with h5py.File(repo / "hest_data" / "st" / f"{slide}.h5ad", "r") as h5:
                    genes = decode(h5["var/_index"][:])
                    lookup = {gene: i for i, gene in reversed(list(enumerate(genes)))}
                    counts = selected_x_columns(h5, [lookup[g] for g in target_names])
                    raw_log = np.log1p(counts).astype(np.float32)
                    assert raw_log.shape == target.shape
                    assert np.array_equal(counts > 0, target > 0), slide
                    # Direct row verification permits duplicate target vectors.
                    # Empirically test both raw-log and row-scaled raw-log targets.
                    positive = counts > 0
                    ratios = np.full(counts.shape, np.nan, dtype=float)
                    ratios[positive] = np.expm1(target)[positive] / counts[positive]
                    informative = positive.any(axis=1)
                    factors = np.ones(len(records), dtype=float)
                    factors[informative] = np.nanmedian(ratios[informative], axis=1)
                    assert np.isfinite(factors).all(), slide
                    scaled_log = np.log1p(counts * factors[:, None]).astype(np.float32)
                    scaled_error = float(np.max(np.abs(scaled_log - target)))
                    assert scaled_error < 2e-6, (slide, scaled_error)
                    source_rows = np.arange(len(records))
                    library = h5["obs/total_counts"][:]
                    total_scale = np.divide(10000.0, library, out=np.zeros_like(library, dtype=float), where=library > 0)
                    cp10k_error = float(np.max(np.abs(np.log1p(
                        counts * total_scale[:, None]) - target)))
                    common_error = None
                    if dataset == "cscc":
                        common_columns = np.array([g in cscc_common_genes for g in genes])
                        common_library = np.empty(len(records), dtype=float)
                        # Stream expression only, at most 128 spots at a time.
                        for start in range(0, len(records), 128):
                            stop = min(start + 128, len(records))
                            common_library[start:stop] = h5["X"][start:stop, :][:, common_columns].sum(axis=1)
                        common_scale = np.divide(10000.0, common_library, out=np.zeros_like(common_library), where=common_library > 0)
                        reconstructed = np.log1p(counts * common_scale[:, None])
                        common_error = float(np.max(np.abs(reconstructed - target)))
                        assert common_error < 2e-6, (slide, common_error)
                        for index in range(len(records)):
                            scale_rows.append(dict(slide=slide, record_index=index,
                                                   hest_total_counts=float(library[index]),
                                                   common_13251_gene_counts=float(common_library[index]),
                                                   exact_formula_scale=float(common_scale[index]),
                                                   empirically_fitted_scale=float(factors[index]),
                                                   max_target_formula_error=float(np.max(np.abs(reconstructed[index] - target[index])))))
                    xy = h5["obsm/spatial"][:][source_rows]
                    raw_integer_error = float(np.max(np.abs(counts - np.rint(counts))))
                    assert raw_integer_error == 0.0
                raw_log_error = float(np.max(np.abs(raw_log[source_rows] - target)))
                gt = np.load(training / "weights_st" / dataset / "random_0.5" /
                             f"fold{heldout_fold(dataset, patient)}" / "gt" / f"{slide}_gt.npy")
                assert gt.shape == target.shape
                gt_error = float(np.max(np.abs(gt - target)))
                assert gt_error < 1e-6
                with h5py.File(repo / "hest_data" / "patches" / f"{slide}.h5", "r") as h5:
                    attrs = dict(h5["img"].attrs)
                    native_mpp = float(attrs["pixel_size"])
                    hest_src_width = int(attrs["patch_size_src"])
                # One CSV entry is rounded to nine decimal places.
                assert abs(native_mpp - float(meta.pixel_size_um_estimated)) < 1e-8, slide
                header_sizes = set()
                for record in records:
                    with Image.open(package / "patches" / slide / f"{record['patch_id']}.png") as im:
                        header_sizes.add(im.size)
                assert header_sizes == {(224, 224)}, (slide, header_sizes)
                tree = cKDTree(xy)
                nearest = tree.query(xy, k=2)[0][:, 1]
                cheb = tree.query(xy, k=2, p=np.inf)[0][:, 1]
                row = dict(dataset=dataset, patient=patient, hest_patient=str(meta.patient),
                           fold=heldout_fold(dataset, patient), slide=slide, n_spots=len(records),
                           native_mpp=native_mpp,
                           metadata_native_mpp=float(meta.pixel_size_um_estimated),
                           training_png_width_px=224,
                           training_patch_fov_um=224 * native_mpp,
                           hest_download_patch_source_width_px=hest_src_width,
                           hest_download_patch_fov_um=hest_src_width * native_mpp,
                           min_neighbor_distance_px=float(nearest.min()),
                           median_neighbor_distance_px=float(np.median(nearest)),
                           n_neighbor_euclidean_lt224=int((nearest < 224).sum()),
                           n_square_patch_overlap=int((cheb < 224).sum()),
                           metadata_spot_pitch_um=float(meta.inter_spot_dist),
                           metadata_spot_diameter_um=float(meta.spot_diameter))
                rows.append(row)
                crop_checks.extend(wsi_spot_check(repo, package / "patches", slide, records, xy))
                target_checks.append(dict(dataset=dataset, slide=slide, n_spots=len(records),
                                          n_targets=300, max_log1p_raw_error=raw_log_error,
                                          max_fitted_row_scaled_log1p_error=scaled_error,
                                          n_zero_target_rows=int((~informative).sum()),
                                          row_scale_factor_min=float(factors[informative].min()),
                                          row_scale_factor_median=float(np.median(factors[informative])),
                                          row_scale_factor_max=float(factors[informative].max()),
                                          max_log1p_cp10k_hest_total_error=cp10k_error,
                                          max_log1p_cp10k_common_13251_error=common_error,
                                          max_ground_truth_error=gt_error,
                                          raw_count_max_integer_error=raw_integer_error,
                                          row_order_matches_hest=bool(np.array_equal(
                                              source_rows, np.arange(len(records))))))
    slide_info = pd.DataFrame(rows)
    slide_info.to_csv(out / "slide_resolution_audit.csv", index=False)
    pd.DataFrame(crop_checks).to_csv(out / "native_wsi_crop_checks.csv", index=False)
    assert all(r["pixel_exact_native_level0_crop"] for r in crop_checks)
    pd.DataFrame(target_checks).to_csv(out / "target_preprocessing_checks.csv", index=False)
    pd.DataFrame(scale_rows).to_csv(out / "cscc_spot_normalization_checks.csv", index=False)
    patient_info = slide_info.groupby(["dataset", "patient"], as_index=False).agg(
        mean_mpp=("native_mpp", "mean"), min_mpp=("native_mpp", "min"),
        max_mpp=("native_mpp", "max"), n_slides=("slide", "size"), n_spots=("n_spots", "sum"))
    patient_info["resolution_stratum"] = ""
    for dataset in cases:
        mask = patient_info.dataset == dataset
        threshold = patient_info.loc[mask, "mean_mpp"].median()
        patient_info.loc[mask, "resolution_stratum"] = np.where(
            patient_info.loc[mask, "mean_mpp"] <= threshold, "lower_mpp", "higher_mpp")
    patient_info.to_csv(out / "patient_resolution_groups.csv", index=False)
    slide_info = slide_info.merge(patient_info[["dataset", "patient", "resolution_stratum"]],
                                  on=["dataset", "patient"], validate="many_to_one")
    metric_rows = []
    for dataset in cases:
        for budget, epoch in FINAL.items():
            for sampler in STRATEGIES:
                for fold in range(4):
                    path = training / "weights_st" / dataset / f"{sampler}_{budget}" / f"fold{fold}" / "slidewise_metrics" / f"epoch_{epoch}.csv"
                    d = pd.read_csv(path)
                    assert not d.slide.duplicated().any() and (d.epoch == epoch).all()
                    for record in d.to_dict("records"):
                        metric_rows.append(dict(dataset=dataset, budget=budget, sampler=sampler,
                                                fold=fold, slide=record["slide"], pcc=record["pcc"]))
    metrics = pd.DataFrame(metric_rows).merge(slide_info, on=["dataset", "fold", "slide"],
                                             validate="many_to_one")
    random = metrics[metrics.sampler == "random"][["dataset", "budget", "fold", "slide", "pcc"]].rename(columns={"pcc": "random_pcc"})
    metrics = metrics.merge(random, on=["dataset", "budget", "fold", "slide"], validate="many_to_one")
    metrics["delta_pcc_vs_random"] = metrics.pcc - metrics.random_pcc
    metrics.to_csv(out / "slidewise_pcc_with_resolution.csv", index=False)
    patient_metrics = metrics.groupby(["dataset", "budget", "sampler", "patient", "resolution_stratum"], as_index=False).agg(
        pcc=("pcc", "mean"), random_pcc=("random_pcc", "mean"),
        delta_pcc_vs_random=("delta_pcc_vs_random", "mean"), n_slides=("slide", "nunique"),
        mean_mpp=("native_mpp", "mean"))
    patient_metrics.to_csv(out / "patientwise_pcc_with_resolution.csv", index=False)
    strata = patient_metrics.groupby(["dataset", "budget", "sampler", "resolution_stratum"], as_index=False).agg(
        n_patients=("patient", "nunique"), n_slides=("n_slides", "sum"),
        mean_patient_pcc=("pcc", "mean"), mean_patient_random_pcc=("random_pcc", "mean"),
        mean_patient_delta=("delta_pcc_vs_random", "mean"),
        min_patient_delta=("delta_pcc_vs_random", "min"),
        max_patient_delta=("delta_pcc_vs_random", "max"))
    strata.to_csv(out / "resolution_stratified_pcc.csv", index=False)
    correlations = []
    for (dataset, budget, sampler), d in patient_metrics.groupby(["dataset", "budget", "sampler"]):
        correlations.append(dict(dataset=dataset, budget=budget, sampler=sampler,
                                 n_patients=len(d), descriptive_r_mpp_vs_pcc=correlation(d.mean_mpp, d.pcc),
                                 descriptive_r_mpp_vs_delta=correlation(d.mean_mpp, d.delta_pcc_vs_random)))
    pd.DataFrame(correlations).to_csv(out / "descriptive_patient_correlations.csv", index=False)
    summary = {"metadata_source": str(metadata_path),
               "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
               "network_input": "Separate native-resolution 224x224 training PNGs, not HEST resampled .h5 patch images",
               "wsi_checks": crop_checks, "cohorts": {}, "target_preprocessing": {}}
    checks = pd.DataFrame(target_checks)
    for dataset, d in checks.groupby("dataset"):
        summary["target_preprocessing"][dataset] = dict(
            max_log1p_raw_error=float(d.max_log1p_raw_error.max()),
            max_fitted_row_scaled_log1p_error=float(d.max_fitted_row_scaled_log1p_error.max()),
            scale_factor_min=float(d.row_scale_factor_min.min()),
            scale_factor_max=float(d.row_scale_factor_max.max()),
            max_ground_truth_error=float(d.max_ground_truth_error.max()),
            max_log1p_cp10k_hest_total_error=float(d.max_log1p_cp10k_hest_total_error.max()))
        if dataset == "cscc":
            summary["target_preprocessing"][dataset]["n_common_genes"] = len(cscc_common_genes)
            summary["target_preprocessing"][dataset]["max_log1p_cp10k_common_13251_error"] = float(d.max_log1p_cp10k_common_13251_error.max())
    for dataset, d in slide_info.groupby("dataset"):
        total_ss = float(np.sum((d.native_mpp - d.native_mpp.mean()) ** 2))
        within_ss = float(np.sum((d.native_mpp - d.groupby("patient").native_mpp.transform("mean")) ** 2))
        summary["cohorts"][dataset] = dict(
            n_patients=int(d.patient.nunique()), n_slides=len(d), n_spots=int(d.n_spots.sum()),
            native_mpp_min=float(d.native_mpp.min()), native_mpp_max=float(d.native_mpp.max()),
            patch_fov_um_min=float(d.training_patch_fov_um.min()), patch_fov_um_max=float(d.training_patch_fov_um.max()),
            fraction_resolution_variance_between_patients=1 - within_ss / total_ss,
            n_euclidean_neighbor_lt224=int(d.n_neighbor_euclidean_lt224.sum()),
            fraction_euclidean_neighbor_lt224=float(d.n_neighbor_euclidean_lt224.sum() / d.n_spots.sum()),
            n_square_patch_overlap=int(d.n_square_patch_overlap.sum()),
            fraction_square_patch_overlap=float(d.n_square_patch_overlap.sum() / d.n_spots.sum()))
    (out / "resolution_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary(out, summary, strata)
    print(json.dumps(summary["cohorts"], indent=2))
    print(f"Validated {len(crop_checks)} native WSI crops; outputs: {out}")


def heldout_fold(dataset, patient):
    if dataset == "HER2":
        return (ord(patient[-1]) - ord("A")) % 4
    return int(patient.removeprefix("cscc_patient")) - 1


def write_summary(out, summary, strata):
    text = ["# Resolution audit", "", "This is a post hoc description of existing seed-42 predictions, not a scale-normalization experiment.", ""]
    for dataset, d in summary["cohorts"].items():
        text += [f"- {dataset}: {d['n_slides']} slides / {d['n_patients']} patients; all {d['n_spots']} referenced PNG headers are 224x224. Native mpp {d['native_mpp_min']:.6f}–{d['native_mpp_max']:.6f}; field width {d['patch_fov_um_min']:.3f}–{d['patch_fov_um_max']:.3f} um.",
                 f"  Patient grouping accounts for {100*d['fraction_resolution_variance_between_patients']:.1f}% of observed slide-resolution variance. The nearest-center Euclidean distance is below 224 native pixels for {d['n_euclidean_neighbor_lt224']} spots ({100*d['fraction_euclidean_neighbor_lt224']:.2f}%); axis-aligned square intersection occurs for {d['n_square_patch_overlap']} spots ({100*d['fraction_square_patch_overlap']:.2f}%)."]
    text += ["", f"Available local WSIs permitted {len(summary['wsi_checks'])} spot-crop checks across {len(set(r['slide'] for r in summary['wsi_checks']))} HER2 slides: every checked training PNG exactly matches a 224x224 level-0 WSI crop centered at integer-truncated HEST coordinates. Only intersecting TIFF tiles were decompressed. cSCC original WSIs are not present locally, so its native scale is supported by metadata and package correspondence rather than an independent WSI pixel match.",
             "", "The downloadable HEST .h5 patches have separate patch_size_src/target/downsample attributes; they are resampled and are not the network PNGs. Multiplying their target width by native mpp would describe the wrong product. The trainer's 224-to-224 resize is a no-op for the verified PNGs; no physical-field normalization experiment was performed.",
             "", "## Target preprocessing finding", "", "HER2 targets match log1p(raw HEST X). cSCC targets reconstruct as log1p(10000 * c_ig / sum_{h in J} c_ih), where J is the intersection of HEST var names across all 12 cSCC slides (13,251 genes). The formula is checked over every one of 8,671 spots and all 300 model targets. This empirically establishes library-size scaling over the shared gene universe; the original producing script has not been recovered. cSCC does not exactly match normalization by HEST obs.total_counts because that field includes genes outside J. Do not describe both cohorts as unnormalized log-raw targets. Every saved Random-50% ground-truth vector matches the processed targets. Row correspondence is verified by all 300 entries and shared zero support in input order, avoiding ambiguous nearest-neighbor matching for duplicate target vectors. See target_preprocessing_checks.csv, cscc_spot_normalization_checks.csv and the released common-gene list. All-zero target rows cannot identify a fitted factor and are separately counted.",
             "", "## Descriptive resolution strata", "", "Patients are split at the within-cohort median of patient-average native mpp, assigning every slide from a patient to one group. PCC and active-minus-Random differences are averaged first across each patient's slides, then equally across patients in the stratum. This differs from the paper's fold aggregation and is explicitly exploratory. Only seed 42 is available for these cohorts. No significance tests or independent-slide assumptions are used. Groups share overlapping CV training sets; resolution is associated with patient, acquisition and tissue characteristics, so these comparisons cannot establish a causal resolution effect or remove scale confounding.",
             "", "| Cohort | Budget | Strategy | Lower-mpp patient mean delta | Higher-mpp patient mean delta |", "|---|---:|---|---:|---:|"]
    for (dataset, budget, sampler), d in strata[strata.sampler != "random"].groupby(["dataset", "budget", "sampler"]):
        vals = d.set_index("resolution_stratum").mean_patient_delta
        text.append(f"| {dataset} | {budget:.0%} | {sampler} | {vals['lower_mpp']:+.4f} | {vals['higher_mpp']:+.4f} |")
    text += ["", "## Rebuttal boundary", "", "The observed PNG scale variation is now reported in physical units and evaluated descriptively using existing predictions. It is not experimentally separated from patient/cohort effects. A future controlled study should compare a shared physical field of view with explicitly documented resampling and a fixed pixel-input control; it must rerun both selection and predictor training. No such experiment has been performed here. Morphological redundancy and patch overlap are distinct, and neither by itself establishes the mechanism of acquisition rankings.",
             "", "## Code facts", "", "Current samplers._features takes the 1024-dimensional DenseNet trunk output after ReLU and global average pooling and before model.classifier. The separate classifier is Linear(1024,1024), ReLU, Dropout(0.3), Linear(1024,300). Thus '1024-dimensional features immediately before the regression head' is accurate; describing acquisition features as the added head's output is not.",
             "", "The beta=1 scripts use warmup=5, interval=3, cutoff=20 with the same trainer, so labels accumulate at displayed epochs 1,6,9,12,15,18. beta=1 is a nominal full-label reference with six integer-floor quotas and can omit up to five spots. It is neither all labels from epoch 1 nor optimization-matched to lower-budget runs. These are read-only code findings, not changes to training.",
             "", "Reproduce: run audit_resolution.py using the st-select Python environment. Dependencies: numpy, pandas, scipy, h5py, Pillow, tifffile, zarr. Input roots are configurable. Outputs are written only beside this script."]
    (out / "resolution_audit_summary.md").write_text("\n".join(text) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
