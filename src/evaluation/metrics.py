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


def top_n_pcc_grouped(
    predictions: np.ndarray, targets: np.ndarray, group_ids: np.ndarray,
    cutoffs=(10, 50, 100, 200, 300),
) -> Dict[int, float]:
    """Slide-level top-N per-gene PCC: for each N, compute the top-N per-gene PCC
    *within each slide* then average over slides.

    This is the PI's preferred convention applied to the top-N sweep — it removes
    per-slide batch effects (each slide's overall expression level differs) that
    inflate a pooled top-N correlation. Combines top_n_pcc_sweep (top-N ranked
    genes) with pcc_per_gene_grouped (per-slide averaging). Returns {N: mean}.
    """
    group_ids = np.asarray(group_ids)
    n_genes = predictions.shape[1]
    out: Dict[int, float] = {}
    for n in cutoffs:
        n = int(n)
        if n < 1:
            continue
        m = min(n, n_genes)
        per_group = []
        for gid in np.unique(group_ids):
            mask = group_ids == gid
            if mask.sum() < 3:  # need a few spots for a meaningful correlation
                continue
            pccs = pcc_per_gene(predictions[mask][:, :m], targets[mask][:, :m])
            per_group.append(float(np.nanmean(pccs)))
        out[n] = float(np.nanmean(per_group)) if per_group else float("nan")
    return out


def spatial_domain_recovery(
    predictions: np.ndarray, targets: np.ndarray, group_ids: np.ndarray,
    n_clusters: int = 10, seed: int = 42,
) -> Dict[str, float]:
    """Do the predicted expression profiles recover the same spatial domains as
    the ground truth? Cluster predicted vs. true expression *within each slide*
    (KMeans) and score agreement with ARI and NMI, averaged over slides.

    This is the "spatial domain identification / diversity-based clustering"
    evaluation from the review (Table 6: ARI, NMI) and the pitch's Evaluation
    Task #2 — it uses the model's *output features* (predicted expression) rather
    than a pointwise error, so a model that preserves tissue structure scores
    well even where absolute PCC is modest. Clustering is per slide because
    spatial domains are defined within a slide. Returns {'ari':..., 'nmi':...}.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    group_ids = np.asarray(group_ids)
    aris, nmis = [], []
    for gid in np.unique(group_ids):
        mask = group_ids == gid
        n = int(mask.sum())
        k = min(n_clusters, n)
        if k < 2 or n < 2 * k:  # too few spots for a stable k-cluster comparison
            continue
        pred_g = predictions[mask]
        tgt_g = targets[mask]
        if pred_g.std() < 1e-8 or tgt_g.std() < 1e-8:
            continue
        try:
            lbl_true = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(tgt_g)
            lbl_pred = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(pred_g)
        except (ValueError, FloatingPointError):
            continue
        aris.append(adjusted_rand_score(lbl_true, lbl_pred))
        nmis.append(normalized_mutual_info_score(lbl_true, lbl_pred))
    return {
        "ari": float(np.mean(aris)) if aris else float("nan"),
        "nmi": float(np.mean(nmis)) if nmis else float("nan"),
    }


def _morans_i(values: np.ndarray, positions: np.ndarray, k: int = 6) -> np.ndarray:
    """Moran's I per column, over a k-nearest-neighbour spatial graph.

    values: (N, G) — one spatial map per column. positions: (N, 2) spot coords.
    Returns (G,) of Moran's I in roughly [-1, 1]: +1 = strongly clustered in space,
    0 = spatially random, negative = checkerboard.

        I_g = (N / S0) * (z_g' W z_g) / (z_g' z_g),  z = column-centred values

    Computed vectorised over all G columns at once (W is sparse, so `W @ Z` is
    cheap), which matters because we call this per slide per method.
    """
    from sklearn.neighbors import kneighbors_graph

    n = len(values)
    if n < k + 2:
        return np.full(values.shape[1], np.nan)
    W = kneighbors_graph(positions, n_neighbors=min(k, n - 1), mode="connectivity")
    W = ((W + W.T) > 0).astype(float)          # symmetrise: i,j neighbours if either
    s0 = W.sum()
    if s0 == 0:
        return np.full(values.shape[1], np.nan)

    Z = values - values.mean(axis=0, keepdims=True)
    den = (Z ** 2).sum(axis=0)
    num = np.asarray((W @ Z) * Z).sum(axis=0)
    out = np.full(values.shape[1], np.nan)
    ok = den > 1e-12                            # constant columns have undefined I
    out[ok] = (n / s0) * num[ok] / den[ok]
    return out


def spatial_autocorrelation_preservation(
    predictions: np.ndarray, targets: np.ndarray, positions: np.ndarray,
    group_ids: np.ndarray, k: int = 6, max_genes: int = 300,
) -> Dict[str, float]:
    """Does the predicted expression map preserve the tissue's spatial architecture?

    This is the pitch's Evaluation Task #5 / the review's "spatial autocorrelation
    preservation" metric. Per slide we compute Moran's I for every gene on the true
    map and on the predicted map, then report:

      moran_true / moran_pred : mean spatial autocorrelation of each (how clustered
                                the real vs predicted maps are). pred << true means
                                the model is producing spatially flat predictions.
      moran_fidelity          : correlation across genes between I_true and I_pred —
                                does the model get *which* genes are spatially
                                structured right? This is the discriminative one; a
                                model can match the average and still rank genes
                                wrongly.
      moran_abs_err           : mean |I_pred − I_true|.

    Averaged over slides (spatial structure is only defined within a slide).
    """
    group_ids = np.asarray(group_ids)
    g = min(max_genes, predictions.shape[1])
    t_means, p_means, fids, errs = [], [], [], []
    for gid in np.unique(group_ids):
        m = group_ids == gid
        if m.sum() < 10:                        # too few spots for a spatial graph
            continue
        pos = positions[m]
        it = _morans_i(targets[m][:, :g], pos, k)
        ip = _morans_i(predictions[m][:, :g], pos, k)
        ok = np.isfinite(it) & np.isfinite(ip)
        if ok.sum() < 3:
            continue
        t_means.append(float(np.mean(it[ok])))
        p_means.append(float(np.mean(ip[ok])))
        errs.append(float(np.mean(np.abs(ip[ok] - it[ok]))))
        if it[ok].std() > 1e-8 and ip[ok].std() > 1e-8:
            fids.append(float(np.corrcoef(it[ok], ip[ok])[0, 1]))
    f = lambda v: float(np.mean(v)) if v else float("nan")
    return {"moran_true": f(t_means), "moran_pred": f(p_means),
            "moran_fidelity": f(fids), "moran_abs_err": f(errs)}


def celltype_composition_similarity(
    predictions: np.ndarray, targets: np.ndarray, group_ids: np.ndarray = None,
    valid_mask: np.ndarray = None,
) -> Dict[str, float]:
    """Similarity between predicted and true cell-type composition per spot.

    The pitch's Evaluation Task #4. `targets` are proportions over cell types
    (rows sum to 1); spots with no cells have undefined composition and are
    excluded via `valid_mask`.

      cosine     : mean per-spot cosine similarity (scale-free, the usual choice)
      jsd        : mean Jensen-Shannon divergence, 0 = identical, 1 = disjoint
                   (predictions are clipped ≥0 and renormalised to a distribution)
      pcc_per_type: per-cell-type PCC across spots, averaged over types — the
                   composition analogue of per-gene PCC, and the strict one.
    """
    if valid_mask is not None:
        keep = np.asarray(valid_mask, dtype=bool)
        predictions, targets = predictions[keep], targets[keep]
        if group_ids is not None:
            group_ids = np.asarray(group_ids)[keep]
    if len(predictions) == 0:
        return {"cosine": float("nan"), "jsd": float("nan"), "pcc_per_type": float("nan")}

    pn = np.linalg.norm(predictions, axis=1)
    tn = np.linalg.norm(targets, axis=1)
    ok = (pn > 1e-12) & (tn > 1e-12)
    cos = float(np.mean((predictions[ok] * targets[ok]).sum(1) / (pn[ok] * tn[ok]))) \
        if ok.any() else float("nan")

    # JSD needs both rows to be probability distributions.
    p = np.clip(predictions, 0, None)
    p = p / np.maximum(p.sum(1, keepdims=True), 1e-12)
    q = np.clip(targets, 0, None)
    q = q / np.maximum(q.sum(1, keepdims=True), 1e-12)
    m = 0.5 * (p + q)
    def _kl(a, b):
        mask = a > 0
        return np.where(mask, a * np.log2(np.clip(a, 1e-12, None) / np.clip(b, 1e-12, None)), 0.0).sum(1)
    jsd = float(np.mean(0.5 * _kl(p, m) + 0.5 * _kl(q, m)))

    if group_ids is not None:
        ppt = pcc_per_gene_grouped(predictions, targets, group_ids)
    else:
        ppt = float(np.nanmean(pcc_per_gene(predictions, targets)))
    return {"cosine": cos, "jsd": jsd, "pcc_per_type": ppt}


def selection_diversity(
    features: np.ndarray, selected_indices, positions: np.ndarray = None,
    max_pairs: int = 4000, seed: int = 42,
) -> Dict[str, float]:
    """Diversity of an AL strategy's *selected* spots, in representation and
    (optionally) physical space. Lets us ask whether a strategy that samples a
    more spread-out / less redundant set actually predicts better.

    - feature_spread: mean pairwise Euclidean distance among selected feature
      vectors (higher = more diverse coverage of the representation space).
    - spatial_spread: same over 2D spot coordinates, if positions are given.
    Pairwise distances are subsampled to max_pairs for large selections.
    Returns {'feature_spread':..., 'spatial_spread':...(optional)}.
    """
    from sklearn.metrics import pairwise_distances

    sel = np.asarray(list(selected_indices), dtype=int)
    out: Dict[str, float] = {}
    if len(sel) < 2:
        return {"feature_spread": float("nan")}
    rng = np.random.default_rng(seed)
    if len(sel) > 1500:  # cap the O(n^2) distance matrix
        sel = rng.choice(sel, size=1500, replace=False)

    def _mean_pairwise(mat):
        d = pairwise_distances(mat)
        iu = np.triu_indices(len(mat), k=1)
        return float(np.mean(d[iu])) if len(iu[0]) else float("nan")

    if features is not None:
        out["feature_spread"] = _mean_pairwise(features[sel])
    if positions is not None:
        out["spatial_spread"] = _mean_pairwise(positions[sel])
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
