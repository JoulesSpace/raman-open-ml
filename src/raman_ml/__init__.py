"""raman_ml: open-source Raman spectroscopy classification & quantification.

Beyond the baseline pipelines, this package ships the trustworthy-ML layer that
maintained Raman libraries lack: uncertainty quantification (conformal),
out-of-distribution / open-set detection, calibration transfer, and spectral
augmentation.
"""
from . import (  # noqa: F401
    augment,
    calibration_transfer,
    datasets,
    dimensionality_reduction,
    interpretability,
    models,
    ood,
    peaks,
    preprocessing,
    ssl,
    tuning,
    uncertainty,
    variable_selection,
)

__all__ = ["augment", "calibration_transfer", "datasets",
           "dimensionality_reduction", "interpretability", "models", "ood",
           "peaks", "preprocessing", "ssl", "tuning", "uncertainty",
           "variable_selection"]
