# API Reference

This page summarizes the public API. For full details, consult the docstrings.

## Top-Level Exports

Public exports live in `src/cvhmax/__init__.py`:

- `CVHM`: high-level model wrapper
- `CVI`: base class for conjugate variational inference readouts
- `Gaussian`, `Poisson`: built-in readouts
- `Params`: readout parameter container
- `HidaMatern`: kernel class for state-space dynamics
- `whittle`: spectral hyperparameter fitting

## CVHM

- `CVHM.fit(y, ymask=None, random_state=None)`
- `CVHM.fit_transform(y, ymask)`
- `CVHM.transform(y, ymask)` (currently not implemented)

Source: `src/cvhmax/cvhm.py`

## CVI and Readouts

- `CVI`: registry-backed base class
- `Gaussian`, `Poisson`: implementations of likelihood-specific updates
- `Params`: container of `C`, `d`, `R`, `M`

Source: `src/cvhmax/cvi.py`

## Kernels and Hyperparameters

- `HidaMatern`: kernel with `Af/Qf/Ab/Qb` and `K(tau)`
- `whittle`: fit spectral hyperparameters using the Whittle loss

Sources: `src/cvhmax/hm.py`, `src/cvhmax/hp.py`
