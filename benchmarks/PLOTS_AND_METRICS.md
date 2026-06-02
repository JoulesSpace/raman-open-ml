# Plots and metrics

A map of what we visualise and how we score each model, with how that compares to
common Raman-analysis practice, the SOTA literature, and general ML platforms.

## Plot types

### Common Raman-analysis practice
The recurring plots in Raman/SERS workflows: spectra overlays (raw / smoothed /
augmented), **confusion matrices**, **predicted-vs-real concentration error
scatter**, ROC / AUC curves, bar charts, heatmaps, **PCA score plots**, and
kNN k-sweep accuracy curves.

### SOTA papers
Confusion matrices; ROC / precision-recall curves; accuracy / macro-F1 bar
charts and leaderboard tables; **t-SNE / UMAP** embedding scatter; **saliency /
Grad-CAM** peak attribution; **reliability (calibration) diagrams**; training
loss/accuracy curves.

### AutoML platform platform
XAI only: **SHAP** mean-|SHAP| feature-importance ranking (`explainability/shap.py`).
No spectroscopy-specific plots (it is a tabular/CV/NLP platform).

### This repo (`benchmarks/plots/`)
- `classification_accuracy.png` - accuracy bar per classifier
- `classification_confusion.png` - row-normalised confusion matrix (winner)
- `quantification_r2.png` - CV R² bar per regressor
- `quantification_parity.png` - predicted-vs-true parity, coloured by particle size
- `domain_shift.png` - grouped bar: in-distribution / cross-domain / adapted
- `pca_scree.png`, `pca_bacteria.png`, `pca_polystyrene.png` - PCA scree + score plots
- `calibration_transfer.png` - R² in-domain / shifted / PDS-corrected
- `shap_classification.png`, `shap_quantification.png` - **SHAP per-wavenumber peak attribution**
- `shap_classification_byclass.png` - **per-class SHAP heatmap** (which bands distinguish each of the 30 isolates)
- `gradcam_classification.png` - Grad-CAM wavenumber importance (1D-CNN)
- `dimreduction_*.png`, `pca_*.png` - embeddings + PCA score/scree plots

We cover the common set (spectra/confusion/parity/PCA), add the SOTA staples
(calibration-aware + attribution), and skip pure-aesthetic overlays.

## Metric -> model mapping

| Task / layer | Models | Metrics |
|---|---|---|
| **Classification** | LogReg, LinearSVM, RandomForest, 1D-CNN, 1D-ResNet, SE-/multi-scale ResNet, deep ensemble | accuracy, balanced accuracy, macro-F1, confusion matrix |
| **Domain shift** | LogReg, 1D-ResNet | accuracy in 3 regimes (in-distribution / cross-domain / adapted) + shift gap |
| **Calibration of a classifier** | any probabilistic classifier / ensemble | ECE (Expected Calibration Error), temperature T |
| **Conformal classification** | any classifier with probabilities | set coverage (vs 1-alpha target), average set size |
| **Open-set / OOD** | classifier + MSP / Energy / Mahalanobis | AUROC, FPR@95%TPR, closed-set accuracy |
| **Quantification** | PLSR, PLSR+VIP, PCR, SVR, kNN, RandomForest, 1D-CNN | R² (pooled CV), RMSE & MAE in log10 units, per-fold R² mean ± std |
| **Conformal / interval regression** | any regressor | interval coverage (vs 1-alpha), mean interval width |
| **Calibration transfer** | PLSR, SVR, RandomForest | R²: in-domain vs secondary-instrument vs PDS-corrected |
| **Hyperparameter tuning** | any sklearn estimator | best CV score under the chosen `scoring` (grid/random/Bayes) |
| **Explainability (XAI)** | RandomForest (SHAP), 1D-CNN (Integrated Gradients) | mean-|SHAP| per wavenumber; IG attribution with completeness |

### Why these metrics
- **30-class isolate ID is balanced**, so accuracy ~ balanced accuracy; macro-F1
  guards against any per-class collapse; the confusion matrix shows *which*
  isolates are confused.
- **Quantification spans ~6 orders of magnitude**, so we score in **log10** units
  (R² / RMSE / MAE) and use the relative-concentration target (see DATA_SOURCES).
- **Coverage / set size / ECE / AUROC** are the trust-layer metrics that make the
  models deployable, and are the ones missing from the rest of the field.
