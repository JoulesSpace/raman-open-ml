# MEMORY - agent memory index

This is the **entry point to the project's agent memory**. It is tracked in git,
read first each session, and kept current as work happens. Start here, then
follow the links. How this store is organised: [README.md](README.md).

## Start here
- [Latest handoff - polish pass (2026-06-02)](handoffs/2026-06-02-polish-and-ssl-recovery.md) - SSL recovery 0.711, regression ensemble, showcase plots, AGPL, badges/citation
- [Honest limitations](notes/honest-limitations.md) - read before quoting numbers
- [Method notes](notes/methods.md) - preprocessing, models, evaluation in one place
- [Dataset notes](notes/datasets.md) - shapes and quirks of both datasets

## Decisions (ADRs)
- [0001 - Two open datasets, one per task](decisions/0001-two-open-datasets.md) - bacteria-ID (clf) + polystyrene (reg)
- [0002 - Relative-concentration target](decisions/0002-relative-concentration-target.md) - removes the particle-size confound
- [0003 - PyTorch device-aware CNN](decisions/0003-pytorch-device-aware-cnn.md) - per-task sizing, CPU or CUDA
- [0004 - Trustworthy-ML positioning](decisions/0004-trustworthy-ml-positioning.md) - own UQ/OOD/transfer, not preprocessing breadth
- [0005 - Domain-shift-aware evaluation](decisions/0005-domain-shift-aware-evaluation.md) - in-dist / cross-domain / adapted

## Insights (gotchas and learnings)
- [bacteria-ID reference->test domain shift](insights/bacteria-id-domain-shift.md) - the headline finding (93%->50%)
- [CNN regressor needs target standardisation](insights/cnn-regressor-target-standardization.md) - R2 -53 -> 0.75 after z-scoring y
- [Linear PLS/PCR fail on cross-size pooling](insights/pls-fails-cross-size-pooling.md) - the confound that motivated decision 0002
- [Mendeley figure CSVs vary in usability](insights/mendeley-figure-csvs.md) - inspect before committing to a dataset

## Reference (external research)
- [SOTA Raman ML synthesis](reference/sota-raman-ml.md) - architectures, OSS landscape, transferable methods, open problems (cited)
- [Competitive analysis](reference/competitive-analysis.md) - criterion-by-criterion scorecard vs SOTA + OSS; where we lead and trail
- [OSS learnings (10+ repos)](reference/oss-learnings.md) - BoxSERS/rampy/peak-fitting/chemometrics; adopted vs wrapper vs roadmap (MCR-ALS, SSL, ...)
- [Plots & metrics catalog](../benchmarks/PLOTS_AND_METRICS.md) - plot types (common practice/SOTA/ours) + metric->model mapping

## Notes (domain knowledge)
- [Method notes](notes/methods.md) - preprocessing, models, evaluation, augmentation
- [Dataset notes](notes/datasets.md) - bacteria-ID and polystyrene specifics

## Handoffs (session log, newest first)
- [2026-06-02 - polish pass](handoffs/2026-06-02-polish-and-ssl-recovery.md) - SSL 0.711, regression ensemble, showcase plots, AGPL, badges/citation
- [2026-06-02 - first build](handoffs/2026-06-02-first-build.md) - scaffold, baselines, trust layer, domain-shift study
