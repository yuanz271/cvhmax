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

- `CVHM.fit(y, ymask=None, random_state=None)`
- `CVHM.fit_transform(y, ymask)`
- `CVHM.transform(y, ymask)` (currently not implemented)

Source: `src/cvhmax/cvhm.py`

## CVI and Readouts

- `CVI`: registry-backed base class
- `Gaussian`, `Poisson`: implementations of likelihood-specific updates
- `Params`: container of `C`, `d`, `R`, `M`

Source: `src/cvhmax/cvi.py`

## Kernels

- `HidaMatern`: kernel with `Af/Qf/Ab/Qb` and `K(tau)`

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

`HidaMatern.K()` and `Ks()` in `hm.py` automatically dispatch to the
generator for orders >= 2, so higher-order kernels work transparently
throughout the pipeline.

Source: `src/cvhmax/kernel_generator/`

See `kernel-generator.md` for usage examples and integration patterns.

## Utilities

- `pad_trials(y_list, ymask_list=None)` → `(y, ymask, trial_lengths)`
  Pad variable-length trials into rectangular arrays.
- `unpad_trials(arrays, trial_lengths)` → `list[Array]` or `list[tuple[Array, ...]]`
  Strip padding from rectangular arrays back to per-trial slices.
  Accepts a single array or a tuple of arrays.

Source: `src/cvhmax/utils.py`

See `data-model.md` for a full usage example.
