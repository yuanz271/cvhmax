# Changelog

## Unreleased

### Added

- Added a loading-estimator axis, selected by `CVHM(estimator=...)`, separating
  how the readout loadings are fitted from the observation model. `RidgeEstimator`
  is the default, `FrozenEstimator` holds supplied parameters fixed, and
  `ARDEstimator` fits one loading precision per latent against the evidence by a
  closed-form fixed point. Fitted precisions are reported as `Params.alpha`.
- Added `gaussian_ard_estimate` and `gaussian_ard_evidence`: automatic relevance
  determination for the Gaussian loading matrix, conditional on the latent
  moments. The expected readout log-likelihood is quadratic in the loadings, so
  the evidence is closed form and no Laplace approximation is needed. Precisions
  and observation variances are fitted jointly against the evidence, with an
  optional Gamma hyperprior on the precisions. It reports per-latent precisions
  whose separation indicates an effective latent dimensionality; columns are
  shrunk rather than deleted, and any threshold is left to the caller.

### Changed

- Stopped penalising the Gaussian readout bias. The ridge penalty now applies to
  the loadings only, matching the Poisson readout and the flat-bias prior the
  loading regularisers are defined against.
- Changed the Poisson readout loading penalty from the L2 norm `‖C‖` to the
  squared L2 norm `‖C‖²`. The norm penalty has a constant-magnitude gradient, so
  it either left the fit unchanged or drove the loading matrix to exactly zero;
  the squared penalty shrinks smoothly and monotonically. The default `gamma`
  drops from 10.0 to 1.5 so the effective strength is unchanged: the old
  gradient magnitude was `gamma`, the new one is `2 * gamma * ‖C‖`.

## 0.3.0

### Breaking changes

- Restricted Gaussian observation noise to positive diagonal covariance.
- Renamed `pad_trials` and `unpad_trials` to `pad_sequences` and
  `unpad_sequences`; the unpadding keyword is now `sequence_lengths`.
- Renamed `trial_info_repr` to `sequence_info_repr`, `bin_info_repr` to
  `step_info_repr`, and `poisson_trial_nell` to `poisson_readout_objective`.
  The Poisson CVI `*_bin_stats` helpers are now `*_step_stats`.

### Added

- Added broadcastable element-wise observation masks for fitting and inference,
  while retaining time-step masks shaped `(sequences, time)`.
- Added a synthetic two-sequence VdP example with partially overlapping features.

### Changed

- Updated the Gaussian readout using masked posterior second moments.
- Recomputed the final posterior after fitting so it matches the final readout.
- Removed loading normalization from Gaussian and Poisson readouts and the
  convergence metric. Fixed GP kernels set the latent prior scale, and fitted
  loadings retain observation-specific gains.

## 0.2.1

### Added

- Added mandatory EM convergence detection with configurable tolerance,
  minimum iterations, patience, and post-fit diagnostics.
- Added GLM loading convergence comparison up to the latent orthogonal rotation
  gauge.

### Fixed

- Hardened convergence and serialized-parameter validation, including
  observation-specific `R` shape checks and strict iteration settings.
- Simplified positive numerical and serialization test coverage.

### Documentation

- Clarified research-code priorities and validation practices for agents.

## 0.2.0

### Breaking changes

- Removed the unvalidated symbolic kernel generator and its `kergen`
  dependency. Built-in Hida–Matérn state-space kernels now support orders 0,
  1, and 2.
- Removed the multi-device batch sharding path from `CVHM.fit`.

### Added

- Added optional CUDA 12 JAX installation through `cvhmax[cuda12]`.
- Added opt-in CUDA integration and CPU–GPU parity tests.
- Added `CVHM.get_config()`, `CVHM.from_config()`, `CVHM.save()`,
  `CVHM.load()`, and `CVHM.infer()`.
- Added a single-archive, data-only model format using JSON metadata and
  Equinox parameter serialization; posterior and latent caches are excluded.

### Fixed

- Replaced deprecated `optax.tree_utils.tree_l2_norm` with
  `optax.tree_utils.tree_norm`.
