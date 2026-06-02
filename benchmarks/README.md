# Benchmarks

Apples-to-apples comparison of algorithms for the two Raman tasks. Every number
and plot here is produced by the runner scripts in `../scripts/`; re-running
regenerates them.

## Run it

```bash
python ../scripts/run_classification.py --train-size 30000 --cnn-epochs 80
python ../scripts/run_quantification.py  --n-aug 40 --cnn-epochs 60 --cv-repeats 3
```

The 1D-CNN uses CUDA automatically when a GPU is present (here: RTX 4090 Laptop),
otherwise CPU.

## Outputs

```
results/
  classification_metrics.csv     accuracy, balanced acc, macro-F1, fit time per model
  quantification_metrics.csv     R2, RMSE, MAE (log units), fold R2 mean/std, time
plots/
  classification_accuracy.png    bar chart of test accuracy per classifier
  classification_confusion.png   row-normalised confusion matrix for the winner
  quantification_r2.png          bar chart of CV R2 per regressor
  quantification_parity.png      predicted vs true (coloured by particle size)
```

## What each task measures

| Task | Split / CV | Headline metric | Why it is fair |
|---|---|---|---|
| Classification (`run_classification.py`) | train on a stratified subsample of the 60k `reference` set, score on the 3k `test` set - this is **cross-domain** (different campaign), so accuracies are ~0.47; the in-distribution comparison is in `run_domain_shift.py` | accuracy (30-class), mean ± std over seeds | every model sees the same training subsample and the same test set |
| Quantification | repeated 5-fold CV on the 48 real spectra | R2 on relative log10 concentration | augmentation is generated inside each train fold only; test folds stay real |

See [MODEL_CARDS.md](MODEL_CARDS.md) for per-model configuration and the
best-algorithm-per-task verdict.
