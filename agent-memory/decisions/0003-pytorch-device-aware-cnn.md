---
title: PyTorch for the CNN, device-aware (CPU or CUDA)
type: decision
date: 2026-06-02
status: accepted
tags: [models, cnn, gpu]
---

# 0003 - PyTorch, device-aware CNN

## Context

The `ml-training` venv ships PyTorch but no TensorFlow, so the 1D-CNN uses
PyTorch. The venv originally had the CPU-only build (`2.10.0+cpu`); a single
classification run with 40 CNN epochs took ~5 min and quantification CV took
~12 min, dominated by CNN training.

## Decision

- Implement one compact `SpectralCNN` backbone with per-task sizing:
  classification keeps spatial detail (`pool_out=4`, channels 32/64/128);
  regression is heavily pooled and regularised (`pool_out=1`, channels 16/32/64,
  dropout 0.4) because it has only 48 samples.
- Wrap it in sklearn-style `fit`/`predict` objects so it is compared on equal
  footing with the sklearn models.
- Make `device` resolve to CUDA when available, else CPU. The machine has an
  RTX 4090 Laptop (16 GB); on user request torch was reinstalled as the CUDA
  12.8 build so the CNN trains on GPU.

## Consequences

- `models.py` moves batches to `self.device` in both the train loop and predict.
- Re-running on GPU lets us use more CNN epochs for a fairer classification
  comparison without long wall-clock.
- A fresh clone with CPU-only torch still works unchanged (device falls back).
