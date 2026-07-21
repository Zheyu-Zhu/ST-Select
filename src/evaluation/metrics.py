"""Evaluation metrics for ST expression prediction."""

from typing import Dict

import numpy as np


def pcc_per_gene(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Per-gene Pearson Correlation Coefficient.
    Correlation across spots for each gene, then averaged over genes.

    predictions: (N_spots, N_genes)
    targets: (N_spots, N_genes)
    Returns: array of per-gene PCC values (N_genes,)
    """
    n_genes = predictions.shape[1]
    pccs = np.zeros(n_genes)

    for g in range(n_genes):
        pred_g = predictions[:, g]
        tgt_g = targets[:, g]

        # Handle constant predictions/targets
        if pred_g.std() < 1e-8 or tgt_g.std() < 1e-8:
            pccs[g] = 0.0
            continue

        pccs[g] = np.corrcoef(pred_g, tgt_g)[0, 1]

    return pccs


def pcc_per_spot(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Per-spot Pearson Correlation Coefficient.
    Correlation across genes for each spot, then averaged over spots.

    predictions: (N_spots, N_genes)
    targets: (N_spots, N_genes)
    Returns: array of per-spot PCC values (N_spots,)
    """
    n_spots = predictions.shape[0]
    pccs = np.zeros(n_spots)

    for s in range(n_spots):
        pred_s = predictions[s]
        tgt_s = targets[s]

        if pred_s.std() < 1e-8 or tgt_s.std() < 1e-8:
            pccs[s] = 0.0
            continue

        pccs[s] = np.corrcoef(pred_s, tgt_s)[0, 1]

    return pccs


def mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((predictions - targets) ** 2))


def mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(predictions - targets)))


def pcc_per_gene_grouped(
    predictions: np.ndarray, targets: np.ndarray, group_ids: np.ndarray
) -> float:
    """Per-gene PCC computed *within each group*, then averaged over groups.

    The default per-gene PCC pools ALL test spots together, so between-group
    baseline differences (e.g. per-slide batch effects — each slide's overall
    expression level differs) leak into the correlation and inflate it. The
    field-standard fix is to compute per-gene PCC *within each slide* and average
    across slides; pass group_ids = sample/slide ids for that. (Passing patient
    ids gives the per-patient variant.) Returns mean-over-groups of the
    mean-over-genes per-gene PCC.
    """
    group_ids = np.asarray(group_ids)
    per_group = []
    for gid in np.unique(group_ids):
        m = group_ids == gid
        if m.sum() < 3:  # need a few spots for a meaningful correlation
            continue
        g = pcc_per_gene(predictions[m], targets[m])
        per_group.append(float(np.nanmean(g)))
    return float(np.nanmean(per_group)) if per_group else float("nan")


# Backwards-compatible alias (was patient-only).
def pcc_per_gene_by_patient(predictions, targets, patient_ids) -> float:
    return pcc_per_gene_grouped(predictions, targets, patient_ids)


def top_n_pcc_sweep(
    predictions: np.ndarray, targets: np.ndarray, cutoffs=(10, 50, 100, 200, 300)
) -> Dict[int, float]:
    """Mean per-gene PCC over the first-N dispersion-ranked genes, for each N.

    Assumes the gene (column) axis is already ordered by descending HVG
    dispersion rank (as produced by HVGSelector / the feature cache), so the
    first N columns are the top-N HVGs. Returns {N: mean_per_gene_pcc}.
    """
    n_genes = predictions.shape[1]
    out: Dict[int, float] = {}
    for n in cutoffs:
        n = int(n)
        if n < 1:
            continue
        m = min(n, n_genes)
        pccs = pcc_per_gene(predictions[:, :m], targets[:, :m])
        out[n] = float(np.nanmean(pccs))
    return out


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute all standard metrics."""
    gene_pccs = pcc_per_gene(predictions, targets)
    spot_pccs = pcc_per_spot(predictions, targets)

    return {
        "pcc_per_gene": float(np.nanmean(gene_pccs)),
        "pcc_per_gene_std": float(np.nanstd(gene_pccs)),
        "pcc_per_gene_median": float(np.nanmedian(gene_pccs)),
        "pcc_per_spot": float(np.nanmean(spot_pccs)),
        "pcc_per_spot_std": float(np.nanstd(spot_pccs)),
        "mse": mse(predictions, targets),
        "mae": mae(predictions, targets),
        "n_genes": predictions.shape[1],
        "n_spots": predictions.shape[0],
    }
