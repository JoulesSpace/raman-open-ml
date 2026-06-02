---
title: bacteria-ID has a severe reference->test domain shift (the headline finding)
type: insight
date: 2026-06-02
tags: [classification, domain-shift, evaluation, critical]
---

# bacteria-ID: reference and test are different measurement campaigns

Measured with a plain LogisticRegression (verified):

| protocol | accuracy |
|---|---|
| reference-internal split (in-distribution val) | **0.93** |
| train reference -> official test | **0.496** |
| train reference -> finetune | 0.525 |
| train finetune -> official test | **0.806** |
| train reference+finetune -> test | 0.775 |
| finetune-internal split | 0.897 |

Interpretation: the `reference` set (60k spectra) and the `finetune`+`test`
sets are different acquisition campaigns. A model trained on `reference`
generalises within `reference` (93%) but collapses on `test` (50%). Training on
`finetune` (same distribution as `test`) recovers ~81%, which is why Ho et al.
2019 report ~82% - they fine-tuned on `finetune` before testing.

## Why this matters

This is exactly the open problem the 2026 benchmark (arXiv:2601.16107) calls the
field's #1 issue: instrument/campaign distribution shift (they saw 99%->74%;
here it is 93%->50%). It is the stated #1 clinical concern in the open-world
paper too. Our earlier "the CNN loses to LogReg" was NOT a weak CNN - it was
every model collapsing under shift.

## Decision consequence

Reframe classification around a **domain-shift-aware protocol** and report three
numbers per model: in-distribution (reference val), cross-domain (reference ->
test), and adapted (+finetune -> test). The repo's differentiator becomes
rigorous shift handling + a trust layer (OOD rejection, conformal coverage),
which no maintained OSS Raman library provides. See
[[sota-raman-ml]] and the ADR on the evaluation protocol.
