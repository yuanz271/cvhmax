# API Reference

This page summarizes the public API. For full details, consult the docstrings.

## Top-Level Exports

Public exports live in `src/cvhmax/__init__.py`:

- `CVHM`: high-level model wrapper
- `Estimator`, `RidgeEstimator`, `ARDEstimator`, `FrozenEstimator`: loading
  estimators, selected by `CVHM(estimator=...)`
- `CVI`: base class for conjugate variational inference readouts
- `Gaussian`, `Poisson`: built-in readouts
- `Params`: readout parameter container
- `HidaMatern`: kernel class for state-space dynamics
- `Ks`, `make_Ks`: raw functional kernel API and static-order JAX wrapper
- `pad_sequences`, `unpad_sequences`: utilities for variable-length sequences

## CVHM

- `CVHM.fit(y, valid_y=None, random_state=None)`
  - `cvi_iter` sets the fixed number of inner CVI iterations (default `20`)
  - mandatory outer convergence controls: `tol` (default `0.05`), `min_iter`
    (default `2`), and `convergence_patience` (default `2`)
  - post-fit diagnostics: `converged_`, `n_iter_`
- `CVHM.fit_transform(y, valid_y)`
- `CVHM.infer(y, valid_y=None)` — recompute posterior using fitted readout parameters
- `CVHM.transform(y, valid_y)` (currently not implemented)
- `CVHM.get_config()` — return JSON-compatible configuration dict
- `CVHM.from_config(config)` — classmethod reconstructing CVHM from configuration
- `CVHM.save(path)` — persist model to one self-contained ZIP archive
- `CVHM.load(path)` — classmethod restoring a saved model

Source: `src/cvhmax/cvhm.py`

`estimator` selects how the loadings are fitted, by exact class name, the same
way `observation` selects a readout. `"RidgeEstimator"` (default) holds the
loading precision fixed at `alpha` (default 0.1); `"ARDEstimator"` fits one
precision per latent column against the evidence; `"FrozenEstimator"` leaves the
supplied `params` untouched, which is how a fit is held at preset values.
`ARDEstimator` needs a conjugate loading representation, so it is not available
with the Poisson readout. Precisions are reported on the fitted parameters as
`Params.alpha`: ranking the columns by it is how effective dimensionality is
read off, and any threshold is the caller's.

`valid_y` may be `None`, a time-step mask shaped `(sequences, time)` (or `(time,)`
for one sequence), or a boolean mask broadcastable to `y`, including
`(sequences, time, 1)` and `(sequences, 1, features)`. Fitting requires each
feature to be observed in at least one time step. With ARD, each feature must
be observed at more than `n_components + 1` time steps; ARD's initial
precisions must be positive and its Gamma prior requires `alpha_shape > 1` and
`alpha_rate > 0`. Sequences have independent
latent trajectories but share kernels, `dt`, and the readout; feature indices
must refer to the same variables across sequences. Sequence-specific readouts
are not represented.

`CVHM.save(path)` and `CVHM.load(path)` persist and restore model state without
saving posterior or latent caches. The serialization uses the
`get_config()`/`from_config()` protocol on both `CVHM` and `HidaMatern`. Only
the built-in readout names and built-in estimators are supported; user-defined
estimators are rejected. `estimator_config` is written as finite JSON
primitives, so a numpy or JAX `alpha` becomes a list with its logical shape
preserved.
After fitting, `CVHM.posterior` and `CVHM.latent` describe the same posterior
under the final readout parameters. `CVHM.infer(...)` recomputes the GP
posterior without updating fitted parameters. See `serialization.md` for the
archive format and compatibility policy.

## CVI and Readouts

- `CVI`: registry-backed base class for an observation model, with stateless
  class methods
  - `initialize_params(y, valid_y, n_factors, *, random_state, penalty=None, params=None)`
    — data-driven starting readout, called by the estimator
  - `initialize_info(params, y, valid_y)` — pseudo-obs info in latent space
  - `update_pseudo(params, y, valid_y, m, V, j, J, lr)` — CVI update in latent space
  - `loading_terms(y, valid_y, m, V)` — conjugate loading statistics
    (`LoadingTerms`), or `None` when the likelihood has no conjugate form
  - `readout_objective(params, y, valid_y, m, V, penalty)` — penalised negative
    expected log-likelihood, averaged per observed time step
- `Estimator`: registry-backed base class that owns the readout M-step and the
  loading prior
  - `initialize_params(likelihood, y, valid_y, n_factors, *, random_state, params=None)`
    — drives the likelihood's start and fills `Params.alpha`
  - `estimate(likelihood, params, y, valid_y, m, V)` — returns `(Params, objective)`
- `Gaussian`, `Poisson`: built-in observation models
- `RidgeEstimator`, `ARDEstimator`, `FrozenEstimator`: built-in estimators
- `Params`: container of `C`, `d`, `R`, and the fitted loading precisions
  `alpha` (no `M` — that lives in CVHM)

The built-in readouts use the loading matrix `Params.C`. Gaussian `Params.R`
is a positive diagonal observation-noise covariance; Poisson uses a scalar zero
sentinel. Fixed GP kernel amplitudes set the latent prior scale, while `C`
retains observation-specific gains.

`ARDEstimator` fits the loadings of a Gaussian readout under a per-latent ARD
prior, holding the latent moments fixed within each M-step. The fitted
precisions `Params.alpha` separate latent columns the readout needs from columns
pruned to numerical zero. `ard_evidence(terms, log_alpha, log_variance)` in
`cvhmax.estimator` evaluates the marginal log-evidence at supplied precisions.
See `algorithms.md` for the derivation and its limits.

CVI methods work entirely in latent space `(K)`. CVHM passes readout parameters
explicitly and compares the loading Gram `C C^T`, `d`, and `R` for convergence;
the Gram is invariant to latent rotation, so the same criterion serves every
estimator. `Params.alpha` is reported, not compared.

Source: `src/cvhmax/cvi.py`

## Conversion Helpers

- `lift(j_latent, J_latent, M)` — latent→state information
- `project(z, Z, M)` — state→latent posterior (delegates to `sde2gp`)

Source: `src/cvhmax/cvhm.py`

## Kernels

- `HidaMatern`: user-facing parameter container with scalar `kernel(tau)`,
  state block `K(tau)`, `Af/Qf/Ab/Qb` convenience methods, `get_config()`,
  and `from_config()`.
- `Ks(kernelparam, tau)`: canonical JAX-compatible functional API for the
  raw, jitter-free complex state covariance. Dictionary parameter containers
  are pytrees and can be passed through `jax.jit`, `jax.vmap`, and `jax.scan`;
  use `make_Ks(order)` when the state shape must be static.
- `matern(tau, rho=..., order=...)` and `hm(tau, sigma=..., rho=...,
  order=..., omega=...)`: scalar real-valued covariance helpers.

`HidaMatern.K(tau)` is a thin wrapper around `Ks`: it packs the object fields
and applies the object's instantaneous state-space component `s I` only at
zero lag. Positive-lag cross-covariances remain raw, and with `s=0` the result
agrees with `Ks` up to dtype. Functional dynamics use the jittered stationary
block and raw positive-lag cross-covariance, with Cholesky-based stationary
solves. `HidaMatern.Af/Qf/Ab/Qb` pass the object's `s` to that policy. The raw
state covariance is positive-lag oriented; `kernel(tau)` is the real, even
scalar covariance.

For JAX transformations where the matrix shape must be static, use
`make_Ks(order)` to close over the integer order and pass only numerical
`sigma`, `rho`, and `omega` leaves in the parameter mapping.

Orders 0, 1, and 2 (Matérn-1/2, -3/2, and -5/2) use built-in closed-form
state-space implementations. Higher orders are rejected explicitly because
this package does not validate their state-space construction. Users may pass
custom kernel objects directly through `CVHM(kernels=...)`; the inference
engine does not restrict extensions to the built-in Hida–Matérn family.

Source: `src/cvhmax/hm.py`

## Utilities

- `pad_sequences(y_list, valid_y_list=None)` → `(y, valid_y, sequence_lengths)`
  Pad variable-length sequences; per-sequence masks may use `(time,)` or broadcast
  to `(time, features)`.
- `unpad_sequences(arrays, sequence_lengths)` → `list[Array]` or `list[tuple[Array, ...]]`
  Strip padding from rectangular arrays back to per-sequence slices.
  Accepts a single array or a tuple of arrays.

Source: `src/cvhmax/utils.py`

See `data-model.md` for a full usage example.
