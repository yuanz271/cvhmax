# API Reference

This page summarizes the public API. For full details, consult the docstrings.

## Top-Level Exports

Public exports live in `src/cvhmax/__init__.py`:

- `CVHM`: high-level model wrapper
- `CVI`: base class for conjugate variational inference readouts
- `Gaussian`, `Poisson`: built-in readouts
- `Params`: readout parameter container
- `HidaMatern`: kernel class for state-space dynamics
- `HidaMaternKernelGenerator`, `make_kernel`: kernel generator for arbitrary orders
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
  are pytrees and can be passed through `jax.jit`, `jax.vmap`, and `jax.scan`.
- `matern(tau, rho=..., order=...)` and `hm(tau, sigma=..., rho=...,
  order=..., omega=...)`: scalar real-valued covariance helpers.

`HidaMatern.K(tau)` is a thin wrapper around `Ks`: it packs the object fields
and applies the object's optional diagonal jitter `s`. It contains no
order-specific dispatch. With `s=0`, it agrees with `Ks` up to dtype.
The functional dynamics helpers accept an explicit `jitter=` argument;
`HidaMatern.Af/Qf/Ab/Qb` pass the object's `s` to stabilize consequential
linear-algebra operations. The raw state covariance is positive-lag oriented;
`kernel(tau)` is the real, even scalar covariance.

Orders 0, 1, and 2 (Matérn-1/2, -3/2, and -5/2) currently use built-in
closed-form implementations. Higher orders require the `kergen` extra.

Source: `src/cvhmax/hm.py`

## Kernel Generator

Runtime symbolic construction of Hida-Matern state-space kernel matrices
for arbitrary smoothness orders. Uses SymPy for symbolic differentiation
and `sympy2jax` to convert expressions into JIT-compatible JAX functions.

- `HidaMaternKernelGenerator(order)`: builds a generator for SSM order `M`
  - `.create_K_hat(tau, sigma, rho, omega)` — M x M complex covariance matrix
  - `.get_moments(sigma, rho, omega)` — 2M spectral moments
  - `.get_base_kernel(tau, sigma, rho, omega)` — scalar base kernel
- `make_kernel(order)`: cached factory returning a `HidaMaternKernelGenerator`

The generator order `M` corresponds to the SSM state dimension. The
Matern smoothness is `nu = (M - 1) + 0.5`:

| Generator order (M) | Matern | `HidaMatern.order` |
|---------------------|--------|--------------------|
| 1 | 1/2 | 0 |
| 2 | 3/2 | 1 |
| 3 | 5/2 | 2 |
| N | (2N-1)/2 | N-1 |

`Ks()` is the canonical raw functional dispatch path. `HidaMatern.K()`
delegates to `Ks()` and adds the object's configured covariance jitter.
Functional `Af/Qf/Ab/Qb` accept explicit stabilization jitter, and class
methods pass `HidaMatern.s` to those helpers. Orders 0, 1, and 2 use built-in
implementations; generator dispatch starts at order 3, so higher-order kernels
work transparently throughout the pipeline.

Source: `src/cvhmax/kernel_generator/`

See `kernel-generator.md` for usage examples and integration patterns.

## Utilities

- `pad_trials(y_list, valid_y_list=None)` → `(y, valid_y, trial_lengths)`
  Pad variable-length trials into rectangular arrays.
- `unpad_trials(arrays, trial_lengths)` → `list[Array]` or `list[tuple[Array, ...]]`
  Strip padding from rectangular arrays back to per-trial slices.
  Accepts a single array or a tuple of arrays.

Source: `src/cvhmax/utils.py`

See `data-model.md` for a full usage example.
