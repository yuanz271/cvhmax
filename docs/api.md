# API Reference

This page summarizes the public API. For full details, consult the docstrings.

## Top-Level Exports

Public exports live in `src/cvhmax/__init__.py`:

- `CVHM`: high-level model wrapper
- `CVI`: base class for conjugate variational inference readouts
- `Gaussian`, `Poisson`: built-in readouts
- `Params`: readout parameter container
- `HidaMatern`: kernel class for state-space dynamics
- `Ks`, `make_Ks`: raw functional kernel API and static-order JAX wrapper
- `pad_trials`, `unpad_trials`: utilities for variable-length trials

## CVHM

- `CVHM.fit(y, valid_y=None, random_state=None)`
- `CVHM.fit_transform(y, valid_y)`
- `CVHM.transform(y, valid_y)` (currently not implemented)

Source: `src/cvhmax/cvhm.py`

## CVI and Readouts

- `CVI`: registry-backed base class with stateless abstract methods
  - `initialize_params(y, valid_y, n_factors, *, random_state)` — create params
  - `initialize_info(params, y, valid_y)` — pseudo-obs info in latent space
  - `update_pseudo(params, y, valid_y, m, V, j, J, lr)` — CVI update in latent space
  - `update_readout(params, y, valid_y, m, V)` — M-step for observation params
- `Gaussian`, `Poisson`: built-in readouts
- `Params`: convenience container of `C`, `d`, `R` (no `M` — that lives in CVHM)

CVI methods work entirely in latent space `(K)`. The params structure is
opaque to CVHM — each subclass may use any pytree-compatible container.

Source: `src/cvhmax/cvi.py`

## Conversion Helpers

- `lift(j_latent, J_latent, M)` — latent→state information
- `project(z, Z, M)` — state→latent posterior (delegates to `sde2gp`)

Source: `src/cvhmax/cvhm.py`

## Kernels

- `HidaMatern`: user-facing parameter container with scalar `kernel(tau)`,
  state block `K(tau)`, and `Af/Qf/Ab/Qb` convenience methods.
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

- `pad_trials(y_list, valid_y_list=None)` → `(y, valid_y, trial_lengths)`
  Pad variable-length trials into rectangular arrays.
- `unpad_trials(arrays, trial_lengths)` → `list[Array]` or `list[tuple[Array, ...]]`
  Strip padding from rectangular arrays back to per-trial slices.
  Accepts a single array or a tuple of arrays.

Source: `src/cvhmax/utils.py`

See `data-model.md` for a full usage example.
