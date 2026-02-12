# Algorithms

This page outlines the core algorithms used by cvhmax.

## CVI-EM Loop

The `CVHM.fit` method alternates between:

1) CVI smoothing: update pseudo-observations and smooth latents
2) Readout update: refit the observation model parameters

The inner loop performs several CVI iterations before each outer EM update.

## Information-Form Filtering

Filtering is performed in information form (precision matrices). The forward and backward passes are combined by `bifilter` to obtain smoothed latents.

Key code:

- `information_filter_step` (forward update)
- `information_filter` (scan over time)
- `bifilter` (merge forward and backward results)

Source: `src/cvhmax/filtering.py`

## Hida-Matern Kernels

`HidaMatern` represents Matern kernels as linear Gaussian state-space models. The kernel order determines the state-space dimension. Only select orders are implemented.

Key code:

- `HidaMatern.Af/Qf/Ab/Qb` for transitions and noise
- `HidaMatern.K(tau)` for stationary covariances

Source: `src/cvhmax/hm.py`

## Whittle Hyperparameter Fitting

The `whittle` routine refines kernel hyperparameters in the frequency domain using the Whittle likelihood. It flattens a nested spec into a trainable vector, runs BFGS, and reconstructs the spec.

Source: `src/cvhmax/hp.py`
