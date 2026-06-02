---
title: Evaluate classification with a domain-shift-aware protocol
type: decision
date: 2026-06-02
status: accepted
tags: [evaluation, domain-shift, classification]
---

# 0005 - Domain-shift-aware evaluation

## Context

bacteria-ID's `reference` and `finetune`/`test` splits are different measurement
campaigns ([[bacteria-id-domain-shift]]). Reporting a single accuracy is
misleading: in-distribution it is ~94%, cross-campaign ~50%. Most papers report
only the favourable number.

## Decision

`scripts/run_domain_shift.py` reports three numbers per model:

1. **in-distribution** - train on 80% of `reference`, test on its held-out 20%.
2. **cross-domain** - train on `reference`, test on the official `test` set.
3. **adapted** - train on `finetune` (same campaign as `test`), test on `test`.

The gap (1) - (2) quantifies the shift; (3) shows how much target-domain data
recovers. The deep model under shift is additionally wrapped in temperature
scaling + conformal sets so its (in)confidence is calibrated and its prediction
sets carry a coverage guarantee.

## Consequences

- Honest headline numbers; the shift is a feature of the study, not hidden.
- Establishes the template for adding cross-instrument splits later.
- Verdict language must always state which of the three regimes a number is from.
