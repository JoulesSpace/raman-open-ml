---
title: Mendeley "data behind figures" CSVs are not always trainable matrices
type: insight
date: 2026-06-02
tags: [data, gotcha]
---

# Mendeley figure CSVs vary wildly in usability

Both candidate quantification datasets store their CSVs as per-figure exports,
fetched via the public API
(`https://data.mendeley.com/public-api/datasets/<id>/files?folder_id=root&version=1`,
then each file's `content_details.download_url`). `urllib` hits a 403 on the
redirect to the storage backend; `curl -L` works.

- **Viral set (44sgp2jvj5):** `Fig_3_data.csv` held only 2 example spectra
  (c=4.16e5, c=5.76e5). Useless for training a regressor. Rejected.
- **Polystyrene set (33wf5rtr4h):** `Fig.S1..S8` are genuine 6-point dilution
  series (10 column-blocks of [Raman Shift, mean Intensity, SD]; row 0 names,
  row 1 units, row 2 sample labels, data from row 3). Usable. Parser lives in
  `datasets._parse_polystyrene_file`.

Takeaway: before committing to a Mendeley dataset, pull the file list and
actually inspect one CSV's shape and headers. "Has Raman + concentration in the
abstract" does not mean "ships a trainable matrix".
