---
title: Learnings from 10+ OSS Raman/spectroscopy repos - adopted vs roadmap
type: reference
date: 2026-06-02
tags: [oss, research, roadmap, peak-fitting, preprocessing, chemometrics]
---

# Cross-repo learnings (BoxSERS, rampy, + 10 repos via 3 agents)

Three parallel research agents surveyed peak-fitting, preprocessing, and
framework/chemometrics repos. Everything below is "novel idea vs wrapper" and
"adopted vs roadmap".

## Adopted this session
- **Cosmic-ray / spike removal** (Whitaker-Hayes 2018) - `preprocessing.remove_cosmic_rays`.
  Flagged by the preprocessing agent as our #1 correctness gap (a spike wrecks
  normalisation). 
- **ModPoly / IModPoly / SNIP / rubberband baselines** (from BaselineRemoval,
  raman_tl, pybaselines) - a polynomial/clipping/hull family alongside our
  penalised-LS ALS/arPLS/airPLS. Selectable via `remove_baseline(method=...)`.
- **Peak-feature extractor** (`peaks.py`) - center/area/height/FWHM per band +
  area-ratios, from a training-discovered band registry. The peak-fitting agent's
  main recommendation; matches common practice (rampy, BoxSERS). Honest result:
  on polystyrene, peak features (80) give RF R2 0.54 vs full-spectrum 0.62 - more
  interpretable/compact, slightly less accurate; a complementary view, not a win.
- **Dimensionality reduction** (`dimensionality_reduction.py`): PCA + t-SNE +
  UMAP + MDS + Isomap + LDA with a separability metric (silhouette + kNN-acc);
  `SpectroPCA` (BoxSERS-style: loadings-as-spectra + scatter). PCA/t-SNE/UMAP/MDS
  are the standard exploratory embeddings.
- **XAI**: Grad-CAM (`interpretability.grad_cam_1d`) + SHAP
  (`shap_wavenumber_importance`, Tree/Linear/Kernel dispatch from AutoML platform) +
  the existing Integrated Gradients. SHAP flags chemically-correct bands
  (785/1006 cm-1 bacteria; 992-1025 cm-1 polystyrene).

## "Fancy wrapper" verdicts (do NOT re-implement)
- **BoxSERS** `SpectroRF/SVM/LDA/Kmeans/Gmixture/CNN`, `cf_matrix`, `clf_report`:
  thin sklearn/Keras wrappers + spectro plotting; not new algorithms.
- **raman-fitting / SpectraFit / PeakFit / raman_tl / Raman-noodles**: all
  wrappers over lmfit/scipy. The transferable assets are arPLS-before-fitting,
  the peak registry, the center/width/area triple, and ratio features (adopted).
- **BaselineRemoval ZhangFit == airPLS** (already had it). **DerekKaknes/raman**:
  thin (just resample-to-grid) + a "raw spectra into CNN" ablation idea.

## Roadmap (genuinely new capability, not yet built)
Prioritised by the framework/chemometrics agent (SpectroChemPy, ChemoSpec):
1. **MCR-ALS unmixing** (+ NMF, constraints: non-negativity/unimodality/closure;
   SIMPLISMA/EFA init) - a whole capability class we lack; standard for Raman
   mixture resolution. Impl via `scipy.optimize.nnls` ALS or `pymcr` (BSD).
2. **Self-supervised / contrastive pretraining** (SimCLR + MMD, masked-AE/SMAE)
   feeding the CNN - the highest-rated lever for the low-data regime and the path
   to push classification past the current SOTA (we sit at 0.852 vs SANet 0.861).
3. **Robust PCA + Sparse PCA** in the DR module (outlier-resistant; band-localised
   interpretable loadings) - sklearn `SparsePCA`, `MinCovDet`.
4. **EFA** (component-count estimation) and **ANOVA-PCA** (design-of-experiments:
   separate instrument/day/operator variance from chemistry).
5. **Named, citable preprocessing protocols** (RamanSPy-style versioned presets).
6. **More dataset loaders** (RRUFF, MLROD, API, SOP) + a synthetic spectrum
   generator - turns the repo into a broadly benchmarkable library and enables a
   true cross-dataset distribution-shift study.

## Honest result note (multi-scale architecture)
The SANet-style multi-scale ensemble (`MSResNet1D`) scored 0.844 - BELOW the plain
SE-ResNet ensemble (0.852) on our protocol. The paper reports multi-scale > plain
ResNet, so our quick port (3 stages, base 64) likely under-replicates SANet's
depth/width/tuning. Reported honestly; the plain SE-ResNet ensemble (0.852)
remains our best, beating Ho 2019 (0.822) and trailing SANet (0.861) / SE-ResNet
2024 (0.878). The later heterogeneous mix + TTA reached 0.862, clearing SANet.

## Honest result note (self-supervised pretraining, now implemented)
SSL is no longer a roadmap item. `SpectralMAE` masked-autoencoder pretraining on
63k unlabelled spectra + a fine-tuned 5-member ensemble reaches **0.711** (single
0.702 +/- 0.004, ensemble + TTA 0.711). The first attempt globally average-pooled
the encoder output before the head and scored only 0.36 - it discarded the local
peak structure. Replacing the head with a spatial-feature head (keep the conv
feature map, AdaptiveAvgPool to a small grid, then Linear) recovered it to 0.711.
SSL still trails the supervised ensemble (0.862): with 60k labelled reference
spectra there is little label scarcity for SSL to exploit. Value = the diagnosis
and a working config, reported honestly.
