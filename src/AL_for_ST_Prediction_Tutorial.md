# A Practical Tutorial on Active Learning for Spatial Transcriptomics (ST) Gene-Expression Prediction

> This document is written for engineers and researchers who want to (a) preprocess ST data, (b) implement and benchmark active-learning (AL) strategies, and (c) train modern image-to-expression models on top. It is organized into four parts, **Preprocessing (HEST-1k based)**, **Active-Learning methods (with a strong medical-imaging focus)**, **Published ST prediction models**, and **Datasets**. Every method is paired with its paper and official GitHub repository so you can drop it in directly. Environment setup (CUDA, conda, paths) is left to the reader; the code snippets below are framework-agnostic Python and should run with any reasonable PyTorch + scanpy stack.

---

## Table of Contents

1. [Preprocessing Pipeline (HEST-1k)](#1-preprocessing-pipeline-hest-1k)
2. [Active Learning Methods](#2-active-learning-methods)
3. [ST Gene-Expression Prediction Models](#3-st-gene-expression-prediction-models)
4. [Datasets](#4-datasets)
5. [End-to-End Checklist](#5-end-to-end-checklist)
6. [Common Pitfalls](#6-common-pitfalls)
7. [Reference Resources](#7-reference-resources)

---

## 1. Preprocessing Pipeline (HEST-1k)

[HEST-1k](https://github.com/mahmoodlab/HEST) is the de-facto unified ST benchmark. It re-packages slides from three major platforms (Spatial Transcriptomics, Visium, Xenium) into a uniform `(WSI .tif, h5ad)` pair, where the per-spot pixel coordinates are already aligned to level-0 of the WSI. Building your pipeline on top of HEST is strongly recommended, it removes most platform-specific parsing pain and gives you a single dataset interface across ~1k slides.

- Repo, https://github.com/mahmoodlab/HEST
- Paper, Jaume et al., "HEST-1k, A Dataset for Spatial Transcriptomics and Histology Image Analysis", **NeurIPS 2024 Datasets & Benchmarks Track**. https://arxiv.org/abs/2406.16192
- Hugging Face, https://huggingface.co/datasets/MahmoodLab/hest

The preprocessing pipeline has three stages, **image cropping**, **HVG selection**, and **get position**. Below each subsection contains conceptual rationale, code snippets, and edge cases.

### 1.1 Image Cropping (extracting H&E patches)

**Step 1, TIF -> single-frame PNG (or use HEST's helper directly)**

HEST WSIs are typically OME-TIFF with multiple resolution levels. The spot pixel coordinates in `h5ad.obsm['spatial']` correspond to **level 0** (full resolution), so you must extract at level 0 to avoid offset bugs.

```python
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None   # WSIs can be huge; disable PIL's safety threshold

def tif_to_png(tif_path, out_path, level=0):
    """Convert a (possibly multi-frame) WSI .tif to a single-frame RGB PNG."""
    img = PIL.Image.open(tif_path)
    if hasattr(img, 'n_frames') and img.n_frames > 1:
        img.seek(level)             # level=0 is the resolution h5ad coords are aligned to
    img.convert('RGB').save(out_path)
```

If you prefer not to materialize PNGs, OpenSlide / tifffile both read level-0 directly,

```python
import openslide
slide = openslide.OpenSlide(tif_path)
W, H = slide.level_dimensions[0]
region = slide.read_region((x - 112, y - 112), 0, (224, 224)).convert('RGB')
```

**Step 2, per-spot 224x224 patch cropping**

Spatial Transcriptomics / Visium spots have a diameter of ~100 microns, which corresponds to ~224 px at level 0 for typical 0.5-microns/pixel slides. The community standard is 224x224, matching ImageNet-pretrained CNNs (ST-Net, BLEEP, EGN all use 224). You can pick larger contexts (e.g., 256 or 512) if the spot is in a fibrotic / sparse tissue and the model benefits from broader context.

```python
from PIL import Image
img = Image.open(slide_png).convert('RGB')

# spot coords come from h5ad.obsm['spatial'], see section 1.3
for x, y in spot_coords:
    patch = img.crop((x - 112, y - 112, x + 112, y + 112))    # 224x224
```

**Pre-dump vs. on-the-fly cropping.** For small datasets (<10k spots), pre-dumping all patches as PNGs is fine. For large ones (BC has ~50k spots), on-the-fly cropping inside the DataLoader is much more disk-efficient. Cache the PIL handle (or open with `tifffile.memmap` for tiled TIFFs) per slide and crop lazily.

**HEST also ships a built-in patcher,** which writes 224x224 patches plus a manifest CSV, recommended for first-time users,

```python
from hest import HESTData
hest = HESTData(slide_dir)
hest.dump_patches(patch_save_dir='./patches', target_patch_size=224)
```

### 1.2 HVG Selection (Highly Variable Genes)

**Cardinal rule, concatenate all slides of a dataset into a single AnnData, then run `normalize_total + log1p + highly_variable_genes` jointly.** If you select HVGs per slide independently, the gene sets diverge across slides and your downstream expression vectors are not comparable.

```python
import scanpy as sc
import numpy as np

# 1) concat every slide in a dataset
adatas = [sc.read_h5ad(f'{ST_DIR}/{sid}.h5ad') for sid in slide_ids]
combined = sc.concat(adatas, label='_slide_id', keys=slide_ids)

# 2) library-size normalization (each spot scaled to 10,000 total counts)
sc.pp.normalize_total(combined, target_sum=1e4)

# 3) log1p
sc.pp.log1p(combined)

# 4) pick top-N HVGs (Seurat flavor is the most robust default)
sc.pp.highly_variable_genes(combined, n_top_genes=1000, flavor='seurat')

# 5) order HVGs by normalized dispersion so the list is reproducible
disp = combined.var.loc[combined.var['highly_variable'], 'dispersions_norm']
hvg_names = disp.sort_values(ascending=False).index.tolist()

# 6) write per-slide expression matrices
X = combined.X.toarray() if hasattr(combined.X, 'toarray') else combined.X
gene_idx = [combined.var_names.get_loc(g) for g in hvg_names]
for sid in slide_ids:
    mask = (combined.obs['_slide_id'] == sid).values
    expr = X[mask][:, gene_idx].astype('float32')             # (N_spots, n_HVG)
    np.save(f'hvg/{sid}.npy', {'gene_names': hvg_names, 'expression': expr})
```

**Practical choices and tradeoffs,**

- `n_top_genes`, usually 1000 at preprocessing time, and most ST prediction papers report results on a curated **top-250 / 300** subset (BLEEP, ST-Net, mclSTExp all use ~250-300). Save the full 1000 so you can re-rank later without re-running scanpy.
- `flavor`,
  - `seurat` (default), classical mean-dispersion bins, robust.
  - `seurat_v3` and `cell_ranger`, alternatives, sometimes pick more lowly-expressed genes, you'll see slightly different lists.
- **Cross-dataset gene alignment.** Different datasets use different gene identifiers (HGNC symbol vs. Ensembl ID, mouse vs. human orthologs). Always map names to a canonical set (e.g., HGNC symbol) before intersecting HVG lists across datasets.
- **Marker-gene supplements.** If your downstream analysis cares about specific pathways (e.g., immune markers), augment the HVG list with a curated marker panel to make sure they survive selection.

### 1.3 Get Position (spot pixel coordinates)

HEST aligns every spot to level-0 pixel coordinates of the WSI. They are stored under `adata.obsm['spatial']` as an (N, 2) numpy array.

```python
import scanpy as sc, pandas as pd

adata = sc.read_h5ad(f'{ST_DIR}/{slide_id}.h5ad')
coords = adata.obsm['spatial']                 # (N, 2)
# Column order is (x, y) = (pxl_col, pxl_row), NOT (row, col)!
df = pd.DataFrame({'x': coords[:, 0], 'y': coords[:, 1]},
                  index=[f'spot_{i}' for i in range(len(coords))])
df.to_csv(f'spot_coords/{slide_id}.csv')
```

**The (x, y) vs. (row, col) trap.** HEST follows the OpenSlide convention, `obsm['spatial'][:, 0]` is the column (x) and `obsm['spatial'][:, 1]` is the row (y). Roughly nine out of ten patch-misalignment bugs we've seen come from accidentally interpreting these as (row, col).

**Pixel <-> micrometer conversion.** For Visium data, `adata.uns['spatial'][lib_id]['scalefactors']` contains,

- `spot_diameter_fullres`, spot diameter in level-0 pixels.
- `tissue_hires_scalef`, scale factor from level-0 to the hi-res image stored under `adata.uns['spatial'][lib_id]['images']['hires']`.

Use these if you ever need to draw on the downsampled overview image or convert distances into microns for biology.

**Neighbor graph (often useful for spatial AL).** Build a 2D kNN graph among spots on the same slide,

```python
from sklearn.neighbors import NearestNeighbors
xy = coords                                    # (N, 2)
knn = NearestNeighbors(n_neighbors=7).fit(xy)
dist, idx = knn.kneighbors(xy)                 # dist[:, 0] is self
```

This graph powers spatial smoothing, spatial-stratified sampling, and graph-based AL (CoreGCN etc.) below.

### 1.4 Recommended On-Disk Format

A `data_infor/{dataset}/{slide_id}.npy` storing a list of per-spot dicts works well and is what most published baselines expect,

```python
{
  'sample_id':  'SPA149',           # slide id
  'spot_id':    'spot_0',           # spot index inside the slide
  'x': 1234, 'y': 5678,             # level-0 pixel coordinates
  'expression': np.array(300,),     # log1p-normalized top-300 HVG expression
  # optional fields:
  # 'patch_path': './patches/SPA149/spot_0.png',
  # 'patient_id': 'Patient_A',
  # 'fold': 0,                       # 4-fold CV id
}
```

This format lets the DataLoader iterate spots, lazily crop patches by (slide_id, x, y), and read the precomputed expression vector. The same format works for every backbone listed in section 3.

---

## 2. Active Learning Methods

This section reviews representative AL methods from 2017-2025, with an emphasis on methods that have been used (or are directly applicable) to medical imaging. Methods are grouped into six categories. For each method we give the paper, the official GitHub repository, the core mathematical idea, and how to adapt it to per-spot ST regression. A summary comparison table is at the end (section 2.7).

### 2.1 Uncertainty-Based

These methods rank candidate samples by some notion of model uncertainty, then label the most uncertain ones. They are easy to implement, model-agnostic, and the standard "Phase 0" baseline.

#### Classical baselines, Entropy / Margin / Least Confidence
- Paper, Settles, "Active Learning Literature Survey", 2009. https://burrsettles.com/pub/settles.activelearning.pdf
- Idea, score by the entropy of the predictive distribution (classification) or by per-spot prediction variance / output magnitude (regression). One line of code. Always include these as your sanity-check baselines.

#### MC Dropout (ICML 2016)
- Paper, Gal & Ghahramani, "Dropout as a Bayesian Approximation, Representing Model Uncertainty in Deep Learning". https://arxiv.org/abs/1506.02142
- Code, https://github.com/yaringal/DropoutUncertaintyExps
- Idea, leave dropout on at inference time, do `T` stochastic forward passes, and use the per-output variance as an estimate of epistemic uncertainty. For regression, score = sum of per-gene predictive variance.

```python
def mc_dropout_variance(model, x, T=20):
    model.train()                                    # keep dropout active
    preds = torch.stack([model(x) for _ in range(T)], dim=0)
    return preds.var(dim=0).sum(dim=-1)              # (B,) per-sample uncertainty
```

#### BatchBALD (NeurIPS 2019)
- Paper, Kirsch et al., "BatchBALD, Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning". https://arxiv.org/abs/1906.08158
- Code, https://github.com/BlackHC/BatchBALD
- Idea, BALD scores each sample by the mutual information between its prediction and the model weights. BatchBALD generalizes this to *batch* acquisition, jointly maximizing the mutual information over a set of `k` samples so they are not redundant. For regression you can use a Gaussian-likelihood BALD (Houlsby et al.).

#### Learning Loss for Active Learning (CVPR 2019)
- Paper, Yoo & Kweon, "Learning Loss for Active Learning". https://arxiv.org/abs/1905.03677
- Code, https://github.com/Mephisto405/Learning-Loss-for-Active-Learning
- Idea, attach a small auxiliary head that predicts the model's per-sample loss. At training time, the head is supervised by the detached task loss via a pairwise ranking objective. At acquisition time, the head scores unlabeled samples without needing labels.

Implementation sketch,

```python
class NetWithLossHead(nn.Module):
    def __init__(self, output_dim, hidden=1024):
        super().__init__()
        self.backbone = backbone                                  # e.g., DenseNet-121
        self.main_head = nn.Linear(hidden, output_dim)
        self.loss_head = nn.Sequential(nn.Linear(hidden, 128),
                                       nn.ReLU(),
                                       nn.Linear(128, 1))
    def forward(self, x):
        f = self.backbone(x)
        return self.main_head(f), self.loss_head(f).squeeze(-1), f

# Training loss (margin-ranking on loss-pair differences)
pred, pred_loss, _ = model(x)
L_task = F.mse_loss(pred, y, reduction='none').mean(dim=-1)       # (B,)
L_pred = margin_ranking_loss(pred_loss, L_task.detach(), margin=1.0)
total  = L_task.mean() + lam * L_pred
```

Important detail, freeze gradients flowing from the loss head back into the backbone after a fixed epoch (e.g., epoch 120/200) so the loss head does not destabilize the backbone late in training.

#### TOD, Temporal Output Discrepancy (ICCV 2021)
- Paper, Huang et al., "Semi-Supervised Active Learning with Temporal Output Discrepancy". https://arxiv.org/abs/2107.14153
- Code, https://github.com/siyuhuang/TOD
- Idea, the discrepancy of a sample's prediction between two recent training checkpoints is itself a usable uncertainty signal, no auxiliary network needed. Cheap and surprisingly competitive.

### 2.2 Diversity / Coverage-Based

These methods select samples that *cover* the feature space, ignoring per-sample uncertainty. They work especially well at low budgets and on datasets with many redundant easy samples.

#### Core-Set (ICLR 2018)
- Paper, Sener & Savarese, "Active Learning for Convolutional Neural Networks, A Core-Set Approach". https://arxiv.org/abs/1708.00489
- Code, https://github.com/ozansener/active_learning_coreset
- Idea, frame AL as the k-Center problem. Greedy k-Center, iteratively pick the unlabeled sample farthest (in feature space) from the closest already-selected sample. The resulting "core set" approximates the worst-case generalization bound.

```python
def greedy_k_center(features, selected_idx, k):
    # features: (N, D); selected_idx: list of already-selected global indices
    dist = pairwise_distances(features, features[selected_idx]).min(axis=1)
    picks = []
    for _ in range(k):
        i = int(np.argmax(dist))
        picks.append(i)
        new_d = pairwise_distances(features, features[[i]]).ravel()
        dist = np.minimum(dist, new_d)
    return picks
```

#### BADGE (ICLR 2020)
- Paper, Ash et al., "Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds". https://arxiv.org/abs/1906.03671
- Code, https://github.com/JordanAsh/badge
- Idea, represent each sample by its "gradient embedding" `g = backbone_feat ⊗ (pred - target)` (or `concat(feat, pred)` for regression), then run k-means++ initialization (D^2 sampling) on these embeddings. The magnitude of `g` captures uncertainty (large gradients ⇒ the model is wrong), and the directional spread captures diversity, so BADGE balances both for free.

```python
def gradient_embedding(model, x):
    f = model.backbone(x)
    pred = model.head(f)
    # concat is a regression-friendly approximation of the original outer-product form
    return torch.cat([f, pred], dim=-1)                # (B, D+G)
```

#### CDAL, Contextual Diversity (ECCV 2020)
- Paper, Agarwal et al., "Contextual Diversity for Active Learning". https://arxiv.org/abs/2008.05723
- Code, https://github.com/sharat29ag/CDAL
- Idea, measure diversity by the KL divergence between softmax outputs of pairs of samples, then greedy-pick samples that maximize the total pairwise divergence. Works well for segmentation and context-rich images.

#### MaxHerding / Generalized Coverage (NeurIPS 2024)
- Paper, Bae et al., "Generalized Coverage for More Robust Low-Budget Active Learning". https://arxiv.org/abs/2407.12212
- Code, https://github.com/baekrok/MaxHerding
- Idea, lift coverage from binary balls to a continuous Gaussian-kernel score, `cov(S) = Σ_j max_{i∈S} K(x_j, x_i)`. Greedy maximization gives provably better worst-case coverage than k-Center under noisy distances, and it is consistently the most robust low-budget baseline as of 2024.

```python
def maxherding(features, k, sigma=None):
    if sigma is None:
        sigma = np.median(pairwise_distances(features[:1000]))
    K = rbf_kernel(features, gamma=1/(2*sigma**2))     # (N, N)
    max_sim = np.zeros(K.shape[0])
    picks = []
    for _ in range(k):
        gain = (np.maximum(K, max_sim[None, :]) - max_sim[None, :]).sum(axis=1)
        i = int(np.argmax(gain))
        picks.append(i)
        max_sim = np.maximum(max_sim, K[i])
    return picks
```

#### ProbCover (NeurIPS 2022)
- Paper, Yehuda et al., "Active Learning Through a Covering Lens". https://arxiv.org/abs/2205.11320
- Code, https://github.com/avihu111/TypiClust (same repo as TypiClust)
- Idea, define a delta-ball graph on the feature space (edge if `||f_i - f_j|| < delta`). Greedy max-cover, repeatedly pick the node covering the most still-uncovered neighbors. The strict greedy is `O(kN log N)`; for very large pools you can switch to the one-shot top-k coverage approximation.

#### TypiClust (ICML 2022)
- Paper, Hacohen et al., "Active Learning on a Budget, Opposite Strategies Suit High and Low Budgets". https://arxiv.org/abs/2202.02794
- Code, https://github.com/avihu111/TypiClust
- Idea, K-Means the feature space into `k` clusters, then from each cluster pick the most "typical" sample (highest density / lowest distance to its cluster center). The paper's headline finding, at low budgets, picking typical samples beats picking uncertain ones, because uncertainty-based methods over-pick weird outliers when labels are scarce.

```python
def typiclust(features, k, pca_dim=128, seed=42):
    reduced = PCA(pca_dim, random_state=seed).fit_transform(features)
    labels  = KMeans(k, random_state=seed, n_init=10).fit_predict(reduced)
    picks = []
    for c in range(k):
        members = np.where(labels == c)[0]
        center  = reduced[members].mean(axis=0)
        d = np.linalg.norm(reduced[members] - center, axis=1)
        picks.append(int(members[np.argmin(d)]))
    return picks
```

### 2.3 Hybrid / Adversarial / VAE-Based

These methods learn a *separate* network to score informativeness, decoupling acquisition from the task model.

#### VAAL, Variational Adversarial Active Learning (ICCV 2019)
- Paper, Sinha et al. https://arxiv.org/abs/1904.00370
- Code, https://github.com/sinhasam/vaal
- Idea, train a VAE on all data plus a discriminator that classifies "labeled vs. unlabeled". Unlabeled samples that the discriminator confidently labels as unlabeled are the most under-represented and get acquired.

#### TA-VAAL, Task-Aware VAAL (CVPR 2021)
- Paper, Kim et al., "Task-Aware Variational Adversarial Active Learning". https://arxiv.org/abs/2002.04709
- Code, https://github.com/cubeyoung/TA-VAAL
- Idea, extend VAAL by also feeding a Learning-Loss ranking signal into the discriminator, so the score reflects both representativeness and task difficulty. Usually +1-3% over VAAL on classification benchmarks.

#### WAAL, Wasserstein Adversarial AL (AISTATS 2020)
- Paper, Shui et al., "Deep Active Learning, Unified and Principled Method for Query and Training". https://arxiv.org/abs/1911.09162
- Code, https://github.com/cjshui/WAAL
- Idea, use a Wasserstein-distance regularizer to align labeled and unlabeled distributions during training, while the discriminator outputs query scores. Gives AL a principled query+train joint objective.

#### CCAL, Contrastive Coding Active Learning (NeurIPS 2021)
- Paper, Du et al., "Contrastive Coding for Active Learning under Class Distribution Mismatch". https://arxiv.org/abs/2105.05768
- Code, https://github.com/RUC-DWBI-ML/CCAL
- Idea, train two contrastive encoders, a semantic one and an OOD-aware one, then use their disagreement to acquire samples robust to class-distribution mismatch. Important in medical imaging where the unlabeled pool can contain OOD slides.

#### ALFA-Mix, Active Learning by Feature Mixing (CVPR 2022)
- Paper, Parvaneh et al. https://arxiv.org/abs/2203.07034
- Code, https://github.com/AminParvaneh/alpha_mix_active_learning
- Idea, interpolate features of labeled and unlabeled samples (`alpha * f_u + (1 - alpha) * f_l`) and find the interpolation `alpha` that flips the model's prediction. Samples requiring the smallest alpha to flip are the most informative.

#### SIMILAR / DISTIL Framework (NeurIPS 2021)
- Paper, Kothawade et al., "SIMILAR, Submodular Information Measures Based Active Learning In Realistic Scenarios". https://arxiv.org/abs/2107.00717
- Code, https://github.com/decile-team/distil
- Idea, a unified submodular formulation that subsumes diversity, representation, and uncertainty as different submodular information measures (Facility Location, Graph Cut, etc.). Practical, the DISTIL library implements two-dozen AL methods through this lens.

### 2.4 Medical-Imaging / Pathology / WSI Specific

These methods explicitly target the constraints of medical imaging, expensive expert labels, high class imbalance, OOD noise, and large 2D images.

#### MedAL (ICMLA 2018)
- Paper, Smailagic et al., "MedAL, Accurate and Robust Deep Active Learning for Medical Image Analysis". https://arxiv.org/abs/1809.09287
- Idea, combined uncertainty + feature-distance to selected set. Among the earliest deep-AL works tailored to medical images.

#### Suggestive Annotation (MICCAI 2017)
- Paper, Yang et al., "Suggestive Annotation, A Deep Active Learning Framework for Biomedical Image Segmentation". https://arxiv.org/abs/1706.04737
- Code, https://github.com/yulequan/SA-AL
- Idea, FCN ensemble + bootstrapping, predicted-disagreement scores rank unlabeled patches, top-scoring ones are presented to the annotator. The canonical MICCAI AL paper.

#### CEAL, Cost-Effective Active Learning (CVPR 2017)
- Paper, Wang et al., "Cost-Effective Active Learning for Deep Image Classification". https://arxiv.org/abs/1701.03551
- Idea, an early SSL+AL hybrid, low-confidence samples are sent to the annotator while high-confidence ones get pseudo-labels and join training automatically.

#### ConfiDNet (NeurIPS 2019)
- Paper, Corbière et al., "Addressing Failure Prediction by Learning Model Confidence". https://arxiv.org/abs/1910.04851
- Code, https://github.com/valeoai/ConfidNet
- Idea, learn a calibrated "true class probability" predictor as a confidence head; the predicted confidence is then used both for failure detection and as an AL acquisition signal.

#### CALD, Consistency-based Active Learning for Detection (CVPR 2021)
- Paper, Yu et al., "Consistency-based Active Learning for Object Detection". https://arxiv.org/abs/2103.10374
- Code, https://github.com/we1pingyu/CALD
- Idea, use the inconsistency of the model's predictions under augmentations to estimate uncertainty. Generalizes to dense regression and segmentation; the trick transfers cleanly to per-spot ST predictions.

#### CoreGCN / Sequential GCN AL (CVPR 2021)
- Paper, Caramalau et al., "Sequential Graph Convolutional Network for Active Learning". https://arxiv.org/abs/2006.10219
- Code, https://github.com/razvancaramalau/Sequential-GCN-for-Active-Learning
- Idea, treat labeled+unlabeled samples as nodes of a graph (edges by feature similarity), then a GCN predicts a query value per node. Because ST data has *natural* spatial graph structure, CoreGCN is a particularly good fit.

#### PathAL / Histopathology AL works
A growing line of AL papers focused on whole-slide images,
- "An active learning approach for reducing annotation cost in skin lesion analysis", MICCAI 2018.
- "Suggestive Annotation of Brain Tumour Images with Gradient-Guided Sampling", MICCAI 2020.
- "PathAL, An Active Learning Framework for Histopathology Image Analysis", IEEE TMI 2022.
- "Active learning for accelerating pathologist annotation", Nature BME 2023.

These share three motifs that transfer well to ST,
1. patch-level acquisition under a global slide budget,
2. spatial coverage constraints (do not over-sample one region),
3. annotator-in-the-loop UX (rare but increasingly considered in AL papers).

#### USIM-DAL (MICCAI 2021)
- Paper, "Self-Paced Multi-Task Active Learning for Diagnosing Brain Disorders".
- Idea, self-paced curriculum + multi-task AL, picking samples that maximize joint utility across multiple downstream tasks.

### 2.5 Spatial / Geometric Prior (ST-Native)

ST data carries a strong 2D structural prior, neighboring spots share biology. Methods exploiting this prior tend to be the simplest and most surprising.

#### Poisson-Disk Sampling (PH2ST, Medical Image Analysis 2026)
- Paper, Shen et al., "Spatially Resolved Gene Expression Prediction from Histology via Multi-View Graph Contrastive Learning with HSIC-bottleneck Regularization", MedIA 2026.
- Idea, run Poisson-disk sampling on the spot physical coordinates so any two selected spots are at least `r` apart. Maximally uniform coverage, **needs no features and no model**. The radius `r` is implied by the budget and the slide area.

```python
def poisson_disc_2d(positions, k, r=None, seed=42):
    """positions: (N, 2) spot coords (px or um); k: budget; returns local indices."""
    rng = np.random.default_rng(seed)
    if r is None:
        x_range = positions[:, 0].ptp(); y_range = positions[:, 1].ptp()
        area = max(x_range * y_range, 1e-6)
        r = np.sqrt(area / k) * 0.8
    # standard Bridson Poisson-disc algorithm with a uniform grid
    # cell size = r / sqrt(2)
    ...
```

#### Spatial-Stratified Sampling
- Idea, partition the tissue area into a grid (or Voronoi cells of an initial uniform sample) and draw equal counts from each cell. Implementable in 20 lines; an oft-overlooked baseline that already beats random in most ST settings.

#### Tissue-Aware Sampling
- Mask out background spots (low tissue area / low gene count) before any sampler runs. HEST's `adata.obs['in_tissue']` already does this for Visium; for ST you can use Otsu-thresholding on the H&E intensity.

### 2.6 Reinforcement-Learning AL (frontier)

#### LAL-RL, Reinforcement-Learned AL (NeurIPS 2018)
- Paper, Konyushkova et al., "Learning Active Learning from Data". https://arxiv.org/abs/1703.03365
- Code, https://github.com/ksenia-konyushkova/LAL
- Idea, treat AL as an MDP, the policy outputs "which sample to label next", trained on simulated AL episodes by REINFORCE / Q-learning.

#### Imitation Learning AL (ICLR 2020)
- Paper, Liu et al., "Learning How to Active Learn, A Deep Reinforcement Learning Approach".
- Idea, imitate the oracle "best-acquisition" policy on a meta-training set, transfer to new datasets.

#### Meta-AL (NeurIPS 2020 and later)
- A small but growing line of papers framing AL as meta-learning. Promising but expensive to train.

### 2.7 AL Method Summary Table

| Method | Family | Year/Venue | Complexity | Needs feature extractor | Best at |
|---|---|---|---|---|---|
| Random | baseline | -- | O(1) | no | every paper, always include |
| Entropy / Margin | uncertainty | classical | O(N) | uses existing model | classification |
| MC Dropout | uncertainty | ICML 2016 | O(T*N) | no | small models, low budget |
| BatchBALD | uncertainty + diversity | NeurIPS 2019 | O(kN) | no | small-output classification |
| Learning Loss | uncertainty | CVPR 2019 | O(N) | no | task-agnostic, regression-friendly |
| TOD | uncertainty | ICCV 2021 | O(N) | no | very cheap, SSL-friendly |
| Core-Set | coverage | ICLR 2018 | O(kN) | yes | low budget, image classification |
| BADGE | hybrid | ICLR 2020 | O(kN) | yes | the strongest single baseline |
| CDAL | diversity | ECCV 2020 | O(N^2) | yes | segmentation |
| MaxHerding | coverage | NeurIPS 2024 | O(kN) | yes | low budget, noisy distances |
| ProbCover | coverage | NeurIPS 2022 | O(N^2) / O(N) approx | yes | medium budget |
| TypiClust | clustering | ICML 2022 | O(N*k) | yes | very low budget |
| VAAL | adversarial | ICCV 2019 | extra VAE | yes | imbalanced data |
| TA-VAAL | adversarial + task | CVPR 2021 | extra VAE | yes | when VAAL is being beaten |
| CCAL | contrastive | NeurIPS 2021 | extra encoder | yes | class-distribution mismatch / OOD |
| ALFA-Mix | mixup | CVPR 2022 | O(N) | yes | strong mid-budget performer |
| Sequential GCN | GCN | CVPR 2021 | GCN forward | yes | data with graph structure (ST!) |
| Suggestive Annotation | ensemble | MICCAI 2017 | T models | no | medical segmentation |
| ConfiDNet | confidence | NeurIPS 2019 | extra head | no | medical with calibration needs |
| Poisson Disk | spatial | MedIA 2026 | O(N) | no | ST-native, cheapest non-random baseline |
| Spatial Stratified | spatial | -- | O(N) | no | very simple, surprisingly strong on ST |
| LAL-RL | RL | NeurIPS 2018 | meta-train policy | depends | when meta-training data is available |

### 2.8 Unified Sampler API (for ST workloads)

Regardless of method, wrap each AL strategy behind one common interface,

```python
def select_next_batch(
    candidate_indices: List[int],          # global indices of the unlabeled pool
    selected_indices:  List[int],          # already-labeled global indices
    k:                 int,                # budget for this round
    features:  Optional[np.ndarray] = None,    # (N_total, D)  for Core-Set / BADGE / ...
    positions: Optional[np.ndarray] = None,    # (N_total, 2)  for Poisson / Stratified
    model:     Optional[torch.nn.Module] = None,  # current predictor (uncertainty / LL)
    extras:    Optional[Dict] = None,      # method-specific extras (loss_head, dropout, ...)
) -> List[int]:
    """Return k indices, a subset of candidate_indices."""
```

A typical AL training loop then looks like,

```python
selected = init_random_pool(all_indices, k=initial_budget)
for r in range(num_rounds):
    model = train_predictor(data[selected], epochs=epochs_per_round)
    candidates = list(set(all_indices) - set(selected))
    new_picks = select_next_batch(
        candidates, selected, k=budget_per_round,
        features=feature_cache, positions=spot_coords, model=model,
    )
    selected += new_picks
final_model = train_predictor(data[selected], epochs=full_epochs)
metrics = evaluate(final_model, test_data)
```

Plug in any new AL method by implementing one `select_next_batch` variant; everything else stays fixed.

### 2.9 Notes on adapting AL methods to ST regression

A few practical points that often surprise people who come from classification AL,

1. **Outputs are ~300-dim regression vectors, not class logits.** Uncertainty needs to be defined per-spot, common choices, sum of per-gene predictive variance (MC Dropout), Frobenius norm of `pred - mean(pred)`, or use the Learning-Loss head.
2. **Slide-level vs. spot-level acquisition.** If you must request whole slides, AL collapses to slide selection (Diversity / Coverage in slide-feature space). For per-spot acquisition the candidate pool is the union of unlabeled spots across slides.
3. **Spatial coverage matters.** A pure uncertainty sampler will cluster on tumor borders. Add a per-slide cap or a spatial diversity penalty.
4. **Feature cache.** Pre-extract patch features once with a frozen FM (CONCH / UNI / Virchow / DINOv2) and reuse across AL methods. Saves wall-clock dramatically.
5. **Initial labeled pool.** Most AL methods are sensitive to the initial pool. Use Random or TypiClust to seed; report mean +/- std over 3-5 seeds.

---

## 3. ST Gene-Expression Prediction Models

This section lists peer-reviewed (no preprint) image-to-expression models from 2020-2025, grouped by paradigm. Most have an official PyTorch implementation under a public GitHub repo.

### 3.1 Regression-Based

#### ST-Net (Nature Biomedical Engineering 2020)
- Paper, He et al., "Integrating spatial gene expression and breast tumour morphology via deep learning". https://www.nature.com/articles/s41551-020-0578-x
- Code, https://github.com/bryanhe/ST-Net
- Architecture, ImageNet-pretrained DenseNet-121 backbone, global average pool, MLP head that maps 1024-dim features to the HVG vector. Trained with MSE.
- Why still relevant, simple, fast, and reproducible. Every subsequent paper uses it as a baseline.

#### EGN, Exemplar Guided Network (WACV 2023)
- Paper, Yang et al., "Exemplar Guided Deep Neural Network for Spatial Transcriptomics Analysis of Gene Expression Prediction". https://arxiv.org/abs/2210.16721
- Code, https://github.com/Yan98/EGN
- Architecture, a query patch attends to its K nearest-neighbor "exemplar" patches retrieved from a CONCH/ViT feature bank. The exemplars' expressions are fused with the query feature via a small transformer. Requires building a feature bank in advance.

#### HisToGene (2022)
- Paper, Pang et al., "Leveraging information in spatial transcriptomics to predict super-resolution gene expression from histology images in tumors".
- Code, https://github.com/maxpmx/HisToGene
- Architecture, a ViT processes the whole slide once and outputs a token grid; per-spot expression is regressed from the token at the spot's position.

#### Hist2ST (Briefings in Bioinformatics 2022)
- Paper, Zeng et al., "Spatial transcriptomics prediction from histology jointly through Transformer and graph neural networks". https://academic.oup.com/bib/article/23/5/bbac297/6645485
- Code, https://github.com/biomed-AI/Hist2ST
- Architecture, three modules in series, CNN for patch features, ViT for cross-spot global context, GNN over a 2D spot graph for spatial smoothing.

#### iStar (Nature Biotechnology 2024)
- Paper, Zhang et al., "Inferring super-resolution tissue architecture by integrating spatial transcriptomics with histology".
- Code, https://github.com/daviddaiweizhang/istar
- Architecture, sub-spot-resolution prediction, upsamples coarse spot expression to a finer pixel-level grid using H&E guidance. The first to push ST prediction below the spot scale.

#### THItoGene (Briefings in Bioinformatics 2024)
- Paper, "THItoGene, a deep learning method for predicting spatial transcriptomics from histological images".
- Code, https://github.com/yrjia1015/THItoGene
- Architecture, multi-scale patches (concentric crops at 112 / 224 / 448) fed into a hierarchical transformer.

#### TCGN, Transformer + Convolutional GNN (IEEE TMI 2024)
- Paper, "Transformer with convolutions and graph neural network for spatial gene expression prediction".
- Architecture, in the same family as Hist2ST but with a stronger GNN message-passing scheme and conv-augmented attention.

### 3.2 Retrieval / Contrastive-Based

These methods embed patches and expression into a shared space, then predict by nearest-neighbor retrieval on the labeled side.

#### BLEEP (NeurIPS 2024)
- Paper, Xie et al., "Spatially Resolved Gene Expression Prediction from Histology Images via Bi-modal Contrastive Learning". https://arxiv.org/abs/2306.01859
- Code, https://github.com/bowang-lab/BLEEP
- Architecture, DenseNet-121 encodes the image, an MLP encodes the log-normalized expression. CLIP-style symmetric contrastive loss with temperature 0.07. At inference, for a query patch retrieve the top-`k=50` training expressions and average them.
- Important, the retrieval pool must contain **only training spots**, never validation spots; if you mix you get unrealistic numbers (effectively leaking validation labels).

#### mclSTExp (Briefings in Bioinformatics 2024)
- Paper, Min et al., "Multimodal Contrastive Learning for Spatial Gene Expression Prediction Using Histology Images". https://academic.oup.com/bib/article/25/6/bbae551/7842869
- Code, https://github.com/SCBIT-YYLab/mclSTExp
- Architecture, BLEEP extended with multi-view contrastive augmentations.

### 3.3 Foundation-Model Fine-Tuning

#### OmiCLIP / Loki (Nature Methods 2025)
- Paper, Chen et al., "A visual–omics foundation model to bridge histopathology with spatial transcriptomics".
- Code, https://github.com/GuangyuWangLab2021/Loki
- Architecture, a CLIP-style two-tower model pretrained on ~2.5M (patch, transcriptome) pairs. Downstream, either fine-tune the image tower for direct regression or zero-shot retrieve expression from the joint embedding.

#### UMPIRE (Nature Computational Science 2025)
- Paper, Han et al., "Towards a generalizable pathology foundation model via spatial transcriptomics-guided contrastive learning".
- Code, https://github.com/seqonics/umpire (check official release for the canonical link)
- Architecture, an ST-guided pathology foundation model; downstream typically frozen feature + linear head; achieves strong generalization to unseen organs.

#### Pathology backbones commonly used as feature extractors (not ST-native, but very useful)
- **CONCH**, https://github.com/mahmoodlab/CONCH (Nature Medicine 2024). Vision-language pathology FM, ViT-B/16, paired with PubMed-derived captions.
- **UNI**, https://github.com/mahmoodlab/UNI (Nature Medicine 2024). Self-supervised pathology FM trained on 100M+ patches, ViT-L.
- **Virchow / Virchow2**, Nature Medicine 2024. Trained on 1.5M slides by Paige.
- **PLIP**, Nature Medicine 2023. CLIP fine-tuned on Twitter pathology posts.

Any retrieval / regression model above can use these as a frozen backbone, often gaining 2-5 PCC points without extra training.

### 3.4 ST Prediction Models Quick Reference

| Model | Year | Venue | Paradigm | Backbone | External resources |
|---|---|---|---|---|---|
| ST-Net | 2020 | Nat. Biomed. Eng. | CNN regression | DenseNet-121 | none |
| HisToGene | 2022 | -- | full-slide ViT | ViT | none |
| Hist2ST | 2022 | Brief. Bioinform. | CNN + ViT + GNN | mixed | none |
| EGN | 2023 | WACV | exemplar retrieval | ViT / CONCH | feature bank |
| iStar | 2024 | Nat. Biotech. | super-resolution | CNN+ViT | none |
| THItoGene | 2024 | Brief. Bioinform. | multi-scale ViT | ViT | none |
| TCGN | 2024 | IEEE TMI | transformer + GCN | mixed | none |
| BLEEP | 2024 | NeurIPS | bi-modal contrastive | DenseNet-121 | none |
| mclSTExp | 2024 | Brief. Bioinform. | multi-view contrastive | DenseNet-121 | none |
| OmiCLIP / Loki | 2025 | Nat. Methods | foundation model | ViT | OmiCLIP ckpt |
| UMPIRE | 2025 | Nat. Comput. Sci. | ST-guided pathology FM | ViT | UMPIRE ckpt |

---

## 4. Datasets

[HEST-1k](https://github.com/mahmoodlab/HEST) packages most public ST datasets into a unified format and exposes them on Hugging Face. Below are the most-used ones, grouped by tissue, with their HEST tags. Pull with,

```bash
pip install huggingface_hub
huggingface-cli download MahmoodLab/hest --include "<tag>*" --local-dir ./hest_data
```

### 4.1 Main Public ST Datasets

| Dataset | Platform | Tissue | Slides | HEST tag / source | Paper |
|---|---|---|---|---|---|
| HER2+ Breast | Spatial Transcriptomics (ST) | breast cancer | 36 | `andersson2021spatial` | Andersson et al., Nat. Commun. 2021 |
| He Breast 10x | ST | breast cancer | 68 | `he2020integrating` | He et al., Nat. Biomed. Eng. 2020 |
| cSCC | Visium | skin SCC | 12 | `ji2020multimodal` | Ji et al., Cell 2020 |
| DLPFC | Visium | brain (dorsolateral PFC) | 12 | `maynard2021transcriptome` | Maynard et al., Nat. Neurosci. 2021 |
| 10X Human Breast | Visium | breast | multiple | `10xgenomics_breast` | 10X Genomics public |
| 10X Mouse Brain | Visium | mouse brain | multiple | `10xgenomics_mouse_brain` | 10X Genomics public |
| Kidney atlas | Visium | kidney | multiple | `lake2023atlas` and related | Lake et al., Nature 2023 |
| Liver atlas | Visium | liver | multiple | various studies | -- |
| Pan-organ Xenium | Xenium | multi-tissue | multiple | `xenium_*` | 10X Xenium releases |
| HEST-Bench | mixed | mixed | 1108 ST slides | bundled with HEST-1k | Jaume et al., NeurIPS 2024 D&B |

### 4.2 Suggested Dataset Combinations for ST AL Papers

- **HER2 + He-Breast (10x)**, two breast-cancer datasets from different platforms, gives clean cross-domain generalization tests.
- **cSCC**, only 4 patients, naturally fits patient-level 4-fold CV, manageable for fast iteration.
- **DLPFC**, brain tissue, very different spatial patterns from breast, validates cross-organ transfer.
- **Xenium**, sub-spot (single-cell) resolution; the next frontier for AL once your pipeline is mature.

### 4.3 Common Data Splits and Conventions

- **Patient / donor-level k-fold CV (most common, 4-fold).** Make sure all slides of one patient stay in the same fold; otherwise leakage. HER2 has 8 donors -> 2 per fold; cSCC has 4 patients -> 1 per fold.
- **Slide-level holdout** (favored by 3D datasets), the middle slice of each patient goes to training, outer slices to test. Used in cSCC and HER2.
- **Annotation budget (AL).** Report results across multiple budget ratios, common sweep is `mask_ratio in {0.05, 0.10, 0.15, 0.30, 0.50, 0.75}` (fraction of training spots labeled). 0.10-0.30 is the most informative regime.
- **HVG target,** consistently use the top-300 HVGs for reporting (matches BLEEP, mclSTExp). Some papers report top-50 or top-1000, always state which.
- **PCC convention,** *per-gene PCC* (correlation across spots for each gene, then averaged over genes) is the most informative metric and the default in newer papers. *per-spot PCC* (correlation across genes for each spot, then averaged) is also seen but tends to over-report because spot-level correlations are dominated by housekeeping genes. **Always state which convention you use.**

### 4.4 External Single-Cell References (when methods need them)

Several modern methods (Loki/OmiCLIP, UMPIRE, cell-aware AL) use external single-cell atlases as a prior. Useful sources,

- **Breast**, Chen 2025 + Reed 2024 + Klughammer 2024 SC atlases (~3M cells combined).
- **Kidney**, Lake 2025 multi-modal kidney atlas (~2M cells).
- **General-purpose portals,**
  - [CELLxGENE](https://cellxgene.cziscience.com/) (curated, CC-BY).
  - [Human Cell Atlas](https://www.humancellatlas.org/) (multi-tissue).
  - [Tabula Sapiens](https://tabula-sapiens-portal.ds.czbiohub.org/).

Integration pipeline,
1. Map gene names to your ST HVG list (HGNC symbol vs. ENSG; use BioMart / mygene.info for aliases).
2. Compute cell embeddings with a foundation model, [scGPT](https://github.com/bowang-lab/scGPT) or [Geneformer](https://huggingface.co/ctheodoris/Geneformer).
3. K-Means cluster the embeddings into ~100-500 cluster prototypes that downstream code can use for coverage / matching.

---

## 5. End-to-End Checklist

Steps to take a fresh HEST-1k subset through a complete AL experiment,

1. **Data acquisition**
   - `pip install huggingface_hub hest`
   - `huggingface-cli download MahmoodLab/hest --include "<tag>*" --local-dir ./hest_data`
2. **Preprocessing**
   - Convert each TIF to PNG (or read level-0 directly with OpenSlide).
   - Extract spot coordinates from `obsm['spatial']` to per-slide CSVs.
   - Concat the dataset's slides into one AnnData, normalize + log1p, select top-1000 HVGs jointly. Save per-slide expression matrices.
   - Optionally pre-dump 224x224 patches; otherwise crop on the fly.
3. **(Optional) Feature cache**
   - Extract frozen features with CONCH / UNI / DINOv2 once; store as a `(N_spots, D)` numpy array indexed by global spot id. Reused by every AL method below.
4. **AL experiment**
   - Choose a backbone (start with ST-Net for a fast cycle, then move to BLEEP or EGN once the pipeline is verified).
   - Choose AL methods (always include Random; add at least one strong baseline such as BADGE or MaxHerding before reporting your own).
   - Mask-ratio sweep, `{0.10, 0.30, 0.50}` first; expand to 5-6 ratios for the final paper.
   - Patient-level 4-fold CV, average across folds.
5. **Evaluation**
   - Metrics, **PCC** (primary), MSE, MAE.
   - State the PCC convention (per-gene vs. per-spot) explicitly.
   - Statistical significance, Wilcoxon signed-rank test paired across folds or slides; report p-values for the headline comparisons.
   - Always include the full-supervision (`mask_ratio = 1.0`) upper bound so readers can see how close the AL methods get.

---

## 6. Common Pitfalls

- **`obsm['spatial']` column order is (x, y) = (pxl_col, pxl_row), not (row, col).** Source of ~90% of patch-misalignment bugs.
- **Take level 0 from multi-frame TIFs.** HEST spot pixel coordinates are aligned to the level-0 image; do not silently downsample.
- **Select HVGs jointly across slides.** Per-slide selection produces different gene sets and breaks expression-vector alignment.
- **`expression_norm` is already log1p-normalized.** Do not normalize again before training; do not compare against raw counts at evaluation time.
- **Retrieval pool leakage.** For BLEEP / mclSTExp / EGN, the retrieval pool at inference must contain only training spots. Including validation spots gives inflated numbers.
- **Fold consistency.** When comparing AL methods to a full-supervision upper bound, use the exact same fold split; otherwise the comparison is meaningless.
- **Small budgets need multi-seed averaging.** For `k < 50` (e.g., cSCC at 5% budget), AL results have high variance, average over 3-5 random seeds.
- **PCC convention discipline.** Per-gene vs. per-spot PCC differ by >10 points on the same predictions. Pick one, write it down, never silently switch.
- **Tissue masking before sampling.** Background spots have near-zero counts and are easy to predict; including them in the AL pool inflates baseline numbers and dilutes acquired-spot informativeness. Filter to `in_tissue == True` (Visium) or apply an H&E-intensity Otsu threshold (ST).
- **Gene name aliases.** A surprising number of papers' results disagree because of HGNC alias drift (`MARCH1` vs. `MARCHF1`, etc.). Resolve aliases via mygene.info before any cross-dataset comparison.

---

## 7. Reference Resources

- **HEST-1k toolkit and dataset**, https://github.com/mahmoodlab/HEST
- **DeepAL+ benchmark**, https://github.com/SineZHAN/deepALplus, implementations of BADGE, Core-Set, VAAL, and a dozen more AL methods in a single library.
- **DISTIL (submodular AL framework)**, https://github.com/decile-team/distil, the SIMILAR-family methods plus many baselines.
- **scanpy tutorials**, https://scanpy-tutorials.readthedocs.io/, the standard reference for normalization, HVG selection, and AnnData manipulation.
- **squidpy**, https://github.com/scverse/squidpy, spatial-aware scanpy add-on, useful for neighbor graphs and tissue masks.
- **scGPT** (single-cell FM), https://github.com/bowang-lab/scGPT.
- **Geneformer**, https://huggingface.co/ctheodoris/Geneformer.
- **CONCH (pathology FM)**, https://github.com/mahmoodlab/CONCH.
- **UNI (pathology FM)**, https://github.com/mahmoodlab/UNI.
- **modAL** (general AL library), https://github.com/modAL-python/modAL, a Python AL library that gives you Entropy / Margin / Committee / etc. for free if you only need standard baselines.
- **Recent surveys**,
  - Ren et al., "A Survey of Deep Active Learning", ACM Computing Surveys 2022. https://arxiv.org/abs/2009.00236
  - Zhan et al., "A Comparative Survey of Deep Active Learning", 2022. https://arxiv.org/abs/2203.13450
  - Budd et al., "A Survey on Active Learning and Human-in-the-loop Deep Learning for Medical Image Analysis", Medical Image Analysis 2021. https://arxiv.org/abs/1910.02923
