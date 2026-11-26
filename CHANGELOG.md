# Changelog

## Unreleased

### Breaking changes

- Removed the loading-estimator classes. `CVHM(estimator=..., estimator_config=...)`
  becomes `CVHM(loading="ridge" | "ard" | "fixed", loading_config=...)`, and
  `Estimator`, `RidgeEstimator`, `ARDEstimator` and `FixedEstimator` are no
  longer exported. The observation model now owns its readout M-step, because
  the conditional loading distribution depends on the data term; `Gaussian` and
  `Poisson` implement the three loading kinds, and `CVI.supported_loadings`
  declares which are available.
- Moved `Params` and `LoadingTerms` to `cvhmax.readout`, next to the code that fills
  them; neither is exported from `cvhmax.utils` or `cvhmax.cvi` any more.
- Bumped the archive format version to 3. Manifests record `loading` and
  `loading_config` in place of `estimator` and `estimator_config`.

### Added

- Added Poisson support to ARD using a Laplace approximation for the loading
  M-step while retaining CVI for latent inference.
- Added `cvhmax.readout` with the shared readout fitting functions: penalty,
  conjugate solve, ridge fit, both ARD fits, and the loading evidence.
- Added `cvhmax.simulate` with seeded synthetic-data generators
  (`sample_gp_latents`, `simulate_vdp`, `standardize`), shared by the demos and
  tests in place of per-file copies.

### Changed

- Raised the default number of inner CVI iterations from 5 to 20. Poisson
  inference may be more accurate at additional computational cost; set
  `cvi_iter` explicitly to choose a different count.
- Renamed the training progress label to `Readout score`, since ARD reports
  evidence rather than a negative expected log-likelihood.

## 0.5.1

### Fixed

- Fixed a runaway in the Poisson readout fit. Since 0.4.0 the objective clipped the
  rate at `MAX_LOGRATE` but not the `y * lin` term, so once an optimiser step
  crossed the cap the objective fell without limit. On a 2-sequence, 20-feature
  test problem, initialization returned a bias of order `1e34`. The rate now
  continues along its tangent beyond the cap, which leaves the objective
  unchanged wherever the cap is inactive.

## 0.5.0

### Breaking changes

- Replaced `CVI.update_readout` with `Estimator.estimate` and added estimator
  fields to `CVHM`, changing positional constructor arguments. #12
- Changed the archive format to version 2, recording estimator configuration and
  optional `Params.alpha`. Version 1 archives are rejected. #12
- Changed the default CVI step size from `lr=0.1` to `lr=1.0` and replaced the
  Poisson loading penalty `gamma=1.5` with ridge precision `alpha=0.1`. #12

### Added

- Added `RidgeEstimator`, `FrozenEstimator`, and `ARDEstimator`, selected by
  `CVHM(estimator=...)`, and exported them at the package top level. #12
- Added Gaussian ARD with per-latent evidence-fitted loading precisions, a weak
  Gamma prior, and jointly fitted observation variances. ARD requires each
  feature to have more than `n_components + 1` observed time steps. #12
- Added `examples/demo_ard.py` to compare ridge and ARD on synthetic data. #12

### Changed

- Changed convergence detection to use a loading Gram comparison computed with
  reduced matrices; ARD additionally checks loading-basis and precision changes.
  The revised normalization avoids suppressing small loading changes. #12
- Applied the same Gaussian-prior precision to Poisson and Gaussian loadings,
  without penalizing the bias. Ridge reports its configured `Params.alpha`, and
  Poisson training reports the unpenalized readout objective. #12
- Made array-valued estimator configurations JSON-serializable and restricted
  archived estimators to the supported built-in classes. #12

### Fixed

- Warm-started ARD from fitted precisions and rejected invalid precision, variance,
  evidence, and insufficient-observation cases instead of silently fitting an
  invalid model. #12

## 0.4.0

### Changed

- Raised the default `max_iter` from 10 to 200. Fits stop on the convergence
  criterion, so this only lifts the cap; a converged Poisson fit on the
  two-sequence test problem needs about 100 outer iterations.
- Corrected the Poisson readout objective to the exact negative expected
  log-likelihood. It had applied the posterior-variance correction to the count
  term as well as the rate, which rewarded inflating `C`: on a two-sequence
  problem `||C||` grew from 2.8 to 21.8 over 400 iterations along the latent-scale
  gauge without converging. The exact objective converges in under 100
  iterations with stable scale.
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
