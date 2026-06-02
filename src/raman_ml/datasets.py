"""Dataset loaders for the two open Raman datasets used in this repo.

* ``load_bacteria_id``   -> classification (30 bacterial/yeast isolates)
* ``load_polystyrene``   -> quantification (particle concentration regression)

Both datasets are openly licensed; see DATA_SOURCES.md for provenance.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

from .preprocessing import resample_to_grid

# Repo paths -----------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")

# 30 reference isolates from Ho et al. 2019 (csho33/bacteria-ID config.py) ----
STRAINS = {
    0: "C. albicans", 1: "C. glabrata", 2: "K. aerogenes", 3: "E. coli 1",
    4: "E. coli 2", 5: "E. faecium", 6: "E. faecalis 1", 7: "E. faecalis 2",
    8: "E. cloacae", 9: "K. pneumoniae 1", 10: "K. pneumoniae 2",
    11: "P. mirabilis", 12: "P. aeruginosa 1", 13: "P. aeruginosa 2",
    14: "MSSA 1", 15: "MSSA 3", 16: "MRSA 1 (isogenic)", 17: "MRSA 2",
    18: "MSSA 2", 19: "S. enterica", 20: "S. epidermidis", 21: "S. lugdunensis",
    22: "S. marcescens", 23: "S. pneumoniae 2", 24: "S. pneumoniae 1",
    25: "S. sanguinis", 26: "Group A Strep.", 27: "Group B Strep.",
    28: "Group C Strep.", 29: "Group G Strep.",
}


# --------------------------------------------------------------------------- #
# Classification dataset                                                       #
# --------------------------------------------------------------------------- #
def load_bacteria_id(split: str = "reference", subsample: int | None = None,
                     seed: int = 0):
    """Load a split of the bacteria-ID dataset.

    Parameters
    ----------
    split : {"reference", "finetune", "test", "2018clinical", "2019clinical"}
        Which array pair to load. ``reference`` (60k spectra) is the main
        training pool; ``test`` (3k) is the held-out evaluation set.
    subsample : int, optional
        If given, return a class-stratified random subsample of this many rows.
    seed : int
        RNG seed for the subsample.

    Returns
    -------
    X : (n, 1000) float32   spectra (already baseline-corrected & [0,1] scaled)
    y : (n,) int            class index 0..29
    wavenumbers : (1000,) float
    """
    d = os.path.join(DATA_DIR, "bacteria_id")
    X = np.load(os.path.join(d, f"X_{split}.npy")).astype(np.float32)
    y = np.load(os.path.join(d, f"y_{split}.npy")).astype(np.int64)
    wn = np.load(os.path.join(d, "wavenumbers.npy")).astype(float)

    if subsample is not None and subsample < len(X):
        rng = np.random.default_rng(seed)
        idx = _stratified_indices(y, subsample, rng)
        X, y = X[idx], y[idx]
    return X, y, wn


def _stratified_indices(y: np.ndarray, n: int, rng) -> np.ndarray:
    """Pick ~n indices keeping class proportions roughly balanced."""
    classes = np.unique(y)
    per = max(1, n // len(classes))
    chosen = []
    for c in classes:
        idx = np.where(y == c)[0]
        take = min(per, len(idx))
        chosen.append(rng.choice(idx, size=take, replace=False))
    out = np.concatenate(chosen)
    rng.shuffle(out)
    return out


# --------------------------------------------------------------------------- #
# Quantification dataset                                                        #
# --------------------------------------------------------------------------- #
_CONC_RE = re.compile(r"([0-9.]+E[+\-]?[0-9]+)\s*particles/mL", re.IGNORECASE)


def _parse_polystyrene_file(path: str):
    """Parse one 'Fig.S*_data.csv' file into per-concentration spectra.

    Layout: 10 column-blocks of [Raman Shift, Intensity(mean), SD].
    Row 0 = column names, row 1 = units, row 2 = sample labels, data from row 3.
    Only blocks whose label contains 'particles/mL' are kept.

    Yields tuples (shift, intensity, sd, concentration).
    """
    raw = pd.read_csv(path, header=None, dtype=str)
    labels = raw.iloc[2].tolist()
    body = raw.iloc[3:].reset_index(drop=True)
    n_blocks = raw.shape[1] // 3
    for b in range(n_blocks):
        label = str(labels[3 * b + 1])
        m = _CONC_RE.search(label)
        if not m:
            continue
        conc = float(m.group(1))
        shift = pd.to_numeric(body.iloc[:, 3 * b], errors="coerce").to_numpy()
        inten = pd.to_numeric(body.iloc[:, 3 * b + 1], errors="coerce").to_numpy()
        sd = pd.to_numeric(body.iloc[:, 3 * b + 2], errors="coerce").to_numpy()
        mask = np.isfinite(shift) & np.isfinite(inten)
        yield shift[mask], inten[mask], np.nan_to_num(sd[mask]), conc


def load_polystyrene(n_points: int = 1000):
    """Load the polystyrene limit-of-detection dilution series.

    Each of the 8 'Fig.S*' files is a different particle size with a 6-point
    halving dilution series. All spectra are interpolated onto a common
    wavenumber grid.

    Returns
    -------
    X    : (48, n_points) float   mean spectra
    SD   : (48, n_points) float   per-point standard deviation (for augmentation)
    conc : (48,) float            concentration in particles/mL
    size_id : (48,) int           which file/particle-size each spectrum came from
    grid : (n_points,) float      common wavenumber axis (cm^-1)
    """
    d = os.path.join(DATA_DIR, "polystyrene")
    files = sorted(glob.glob(os.path.join(d, "Fig.S[1-8]_data.csv")))
    if not files:
        raise FileNotFoundError(
            f"No polystyrene Fig.S files in {d}. Run scripts/download_data.py.")

    parsed = []  # (shift, inten, sd, conc, size_id)
    for sid, f in enumerate(files):
        for shift, inten, sd, conc in _parse_polystyrene_file(f):
            parsed.append((shift, inten, sd, conc, sid))

    # Common grid over the overlapping wavenumber range of all spectra.
    lo = max(s.min() for s, *_ in parsed)
    hi = min(s.max() for s, *_ in parsed)
    grid = np.linspace(lo, hi, n_points)

    X, SD, conc, size_id = [], [], [], []
    for shift, inten, sd, c, sid in parsed:
        X.append(resample_to_grid(shift, inten, grid))
        SD.append(resample_to_grid(shift, sd, grid))
        conc.append(c)
        size_id.append(sid)

    return (np.array(X), np.array(SD), np.array(conc),
            np.array(size_id), grid)


def augment_with_sd(X: np.ndarray, SD: np.ndarray, y: np.ndarray,
                    groups: np.ndarray, n_aug: int, seed: int = 0):
    """Generate Gaussian replicate spectra ~ N(mean, SD) for each sample.

    This mirrors real measurement noise (the SD columns are the empirical
    spread across replicate acquisitions) and is used ONLY to enlarge training
    folds. Returns concatenated (X_aug, y_aug, groups_aug).
    """
    rng = np.random.default_rng(seed)
    Xs, ys, gs = [X.copy()], [y.copy()], [groups.copy()]
    for _ in range(n_aug):
        noise = rng.standard_normal(X.shape) * SD
        Xs.append(X + noise)
        ys.append(y.copy())
        gs.append(groups.copy())
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(gs)
