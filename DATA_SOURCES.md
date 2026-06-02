# Data sources

This project uses **only openly-licensed, third-party datasets**. No proprietary
data is included. Run `python scripts/download_data.py` to fetch everything into
`./data/`.

## 1. Classification - bacteria-ID

- **What:** Surface-enhanced-free Raman spectra of 30 bacterial / yeast isolates,
  1000 wavenumbers each (381.98-1792.40 cm⁻¹), already baseline-corrected and
  min-max scaled. 60,000 reference + 3,000 fine-tune + 3,000 test spectra, plus
  two clinical sets.
- **Source:** Ho, C.-S., Jean, N., Hogan, C.A. et al. *Rapid identification of
  pathogenic bacteria using Raman spectroscopy and deep learning.*
  Nature Communications 10, 4927 (2019).
- **Code / data:** https://github.com/csho33/bacteria-ID (code: MIT). Data arrays
  hosted on the linked Dropbox folder.
- **Used for:** 30-class isolate classification.

## 2. Quantification - polystyrene limit-of-detection dilution series

- **What:** Raman spectra (Renishaw inVia, 785 nm) of polystyrene nanoparticle
  suspensions, 8 particle sizes (25-1000 nm) each measured as a 6-point halving
  dilution series, with per-wavenumber standard deviations. 48 mean spectra
  spanning 6.56×10⁸ - 7.08×10¹⁴ particles/mL.
- **Source:** "Dataset for Limit of Detection of Raman Spectroscopy Using
  Polystyrene Particles from 25 to 1000 nm in Aqueous Suspensions", Mendeley Data,
  V1, doi: 10.17632/33wf5rtr4h.1
- **License:** CC BY 4.0.
- **Used for:** regression of log10(particle concentration) from a spectrum.

### A note on the quantification target
Because the same *particle-number* concentration corresponds to very different
*amounts of polystyrene* across particle sizes, absolute concentration is not
perfectly recoverable from a single spectrum when the size is unknown. We
therefore pool all sizes and report cross-validated performance with this caveat;
the per-size colour coding in `results/quantification_parity.png` makes the
residual size structure visible.
