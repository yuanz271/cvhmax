# Known Bugs

Discovered during parity testing against the PyTorch reference implementation
(`hida_matern_gp_lvms/`). Each bug is documented by one or more tests in
`tests/test_cvi.py` or `tests/test_parity.py` (marked `xfail(strict=True)` or
using explicit `assert not allclose` checks).

---

## BUG-1: `trial_info_repr` J shape when vmapped per time bin

| | |
|---|---|
| **File** | `src/cvhmax/utils.py:132` |
| **Severity** | High — produces wrong-rank output, breaks downstream filtering |
| **Affects** | `Gaussian.initialize_info` |
| **Test** | `tests/test_cvi.py::test_gaussian_initialize_info_shape` (xfail) |

### Description

`trial_info_repr` computes `J = C.T @ solve(R, C)` (shape `(L, L)`) then
tiles it with `jnp.tile(J, (T, 1, 1))` where `T = y.shape[0]`.

`Gaussian.initialize_info` (`cvi.py:366`) calls:

```python
return vmap(partial(trial_info_repr, C=H, d=d, R=R))(y, ymask)
```

This vmaps over the time axis of `y` (shape `(T, N)`), so each invocation
receives `y_t` of shape `(N,)`. Inside `trial_info_repr`, `T = y.shape[0]`
evaluates to `N` (observation dimension), producing `J` of shape `(N, L, L)`
per time bin. After the vmap, the final shape is `(T, N, L, L)` instead of
the expected `(T, L, L)`.

### Root cause

`trial_info_repr` was designed to operate on a full `(T, N)` observation
matrix. `Gaussian.initialize_info` incorrectly vmaps it over individual time
bins.

### Suggested fix

Remove the inner vmap in `Gaussian.initialize_info` and call
`trial_info_repr` directly:

```python
return trial_info_repr(y, ymask, H, d, R)
```

`trial_info_repr` already handles the full `(T, N)` tensor and tiles `J`
to `(T, L, L)`.

---

## BUG-2: `Gaussian.infer` cls binding — FIXED

| | |
|---|---|
| **File** | `src/cvhmax/cvi.py:310` |
| **Severity** | High — Gaussian CVI loop dispatches to abstract base class |
| **Affects** | `Gaussian.infer` → `CVI.infer` → `cls.update_pseudo` |
| **Test** | `tests/test_cvi.py::test_gaussian_infer_single_cvi_iter` |
| **Status** | Fixed in `4439022` |

### Description

`Gaussian.infer` overrides the base `CVI.infer` to force `cvi_iter=1`
(Gaussian likelihood is conjugate, so one iteration suffices). It did this
via:

```python
return CVI.infer(params, j, J, y, ymask, z0, Z0, smooth_fun, smooth_args, 1, lr)
```

This called `CVI.infer` as an **unbound classmethod** with `params` as the
first positional argument. Inside `CVI.infer`, `cls` was bound to `CVI`
(the base class), not `Gaussian`. When the `fori_loop` body called
`cls.update_pseudo(...)`, it dispatched to the abstract `CVI.update_pseudo`
instead of `Gaussian.update_pseudo` (which is a no-op that returns `j, J`
unchanged).

### Root cause

Calling `CVI.infer(params, ...)` instead of `super().infer(params, ...)`.
The `super()` form preserves the `cls` binding so that `cls.update_pseudo`
resolves to `Gaussian.update_pseudo`.

### Fix

Changed to `super().infer(...)` which preserves the correct `cls` binding:

```python
return super().infer(params, j, J, y, ymask, z0, Z0, smooth_fun, smooth_args, 1, lr)
```

---

## BUG-3: `Gaussian.initialize_params` creates 1D `R`

| | |
|---|---|
| **File** | `src/cvhmax/cvi.py:434` |
| **Severity** | High — crashes the Gaussian E2E pipeline |
| **Affects** | `Gaussian.initialize_params` → `trial_info_repr` |
| **Test** | `tests/test_cvi.py::test_gaussian_e2e` (xfail) |

### Description

`Gaussian.initialize_params` creates the observation noise as:

```python
R = jnp.zeros(y.shape[-1])  # shape (N,) — 1D vector
```

This `R` is later passed to `trial_info_repr` which calls
`jnp.linalg.solve(R, C)`. `solve` requires a 2D matrix for its first
argument; a 1D vector raises a shape error.

### Root cause

`R` should be a 2D covariance matrix `(N, N)`, not a 1D vector.

### Suggested fix

Initialize `R` as a diagonal matrix:

```python
R = jnp.eye(y.shape[-1])
```

or, if a zero initialization is desired (updated later by `update_readout`):

```python
R = jnp.eye(y.shape[-1]) * eps  # small positive diagonal for numerical stability
```

Using `jnp.zeros` for a covariance matrix is numerically degenerate regardless
of dimensionality.

---

## BUG-4: `poisson_cvi_bin_stats` convention mismatch — FIXED

| | |
|---|---|
| **File** | `src/cvhmax/cvi.py:525` |
| **Severity** | Medium — wrong posterior moments, converges to different fixed point |
| **Affects** | Poisson CVI pseudo-observation updates |
| **Test** | `tests/test_parity.py::test_poisson_cvi_bin_stats_convention` |
| **Status** | Fixed in `4439022` |

### Description

The information filter stores `(z, Z)` in **information form**:

- `Z = J = Sigma^{-1}` (precision matrix)
- `z = h = Sigma^{-1} mu` (information vector)

The correct moment recovery is:

```
mu    = Z^{-1} z     = J^{-1} h
Sigma = Z^{-1}       = J^{-1}
```

This is confirmed by `sde2gp` (`cvhm.py:333`) which correctly uses
`m = solve(Z, z)` and `V = inv(Z)`.

However, `poisson_cvi_bin_stats` (`cvi.py:525`) recovered moments as:

```python
m = -0.5 * cho_solve(Zcho, z)   # i.e. m = -0.5 * Z^{-1} z
```

and the quadratic term (`cvi.py:528`):

```python
quad = einsum("nl, ln -> n", H, -0.5 * cho_solve(Zcho, H.mT))  # V = -0.5 * Z^{-1}
```

The `-0.5` factor is the natural-parameter-to-moment mapping for
exponential families where:

```
eta_1 = Sigma^{-1} mu        eta_2 = -0.5 * Sigma^{-1}
mu    = -0.5 * eta_2^{-1} eta_1
Sigma = -0.5 * eta_2^{-1}
```

But the filter uses **information form**, not natural parameters. In
information form, `Z = Sigma^{-1}` (not `-0.5 * Sigma^{-1}`). The
`-0.5` factor was therefore incorrect, producing:

- `m = -0.5 * mu` (half the true mean, wrong sign)
- `V = -0.5 * Sigma` (negative covariance)

### Root cause

The code was written assuming natural-parameter convention but the
rest of the codebase uses information form.

### Fix

Removed the `-0.5` factor from the moment recovery:

```python
m = cho_solve(Zcho, z)          # mu = Z^{-1} z
quad = einsum("nl, ln -> n", H, cho_solve(Zcho, H.mT))  # Sigma = Z^{-1}
```

---

## BUG-5: `trial_info_repr` broadcasting with `d`

| | |
|---|---|
| **File** | `src/cvhmax/utils.py:130` |
| **Severity** | Medium — silent wrong results when `N != T` |
| **Affects** | `trial_info_repr`, `Gaussian.initialize_info` |
| **Test** | `tests/test_utils.py::test_trial_info_repr_analytic` (uses workaround), `tests/test_parity.py::test_observation_info_parity` (uses workaround) |

### Description

`trial_info_repr` computes `y.T - d` on line 130:

```python
j = C.T @ jnp.linalg.solve(R, y.T - d)
```

`y.T` has shape `(N, T)` and `d` (the bias) has shape `(N,)`. JAX broadcasts
`(N,)` as `(1, N)`, producing `(N, T) - (1, N)` which only works when
`N == T`. When `N != T`, this either raises a shape error or silently
produces wrong results via broadcasting.

### Root cause

`d` should be reshaped to `(N, 1)` (column vector) so that it broadcasts
correctly against `(N, T)`.

### Workaround

Callers can pass `d` as `d[:, None]` (shape `(N, 1)`). The existing parity
tests use this workaround.

### Suggested fix

Inside `trial_info_repr`, reshape `d` before the subtraction:

```python
d = d[:, None] if d.ndim == 1 else d  # ensure (N, 1) for broadcasting
j = C.T @ jnp.linalg.solve(R, y.T - d)
```

---

## Summary table

| ID | Location | Severity | Category | Status | xfail test |
|----|----------|----------|----------|--------|------------|
| BUG-1 | `utils.py:132` / `cvi.py:366` | High | Shape | **Open** | `test_gaussian_initialize_info_shape` |
| BUG-2 | `cvi.py:310` | High | Dispatch | **Fixed** (`4439022`) | `test_gaussian_infer_single_cvi_iter` |
| BUG-3 | `cvi.py:434` | High | Shape | **Open** | `test_gaussian_e2e` |
| BUG-4 | `cvi.py:525` | Medium | Convention | **Fixed** (`4439022`) | `test_poisson_cvi_bin_stats_convention` |
| BUG-5 | `utils.py:130` | Medium | Broadcasting | **Open** | (workaround in tests) |

### Interaction between bugs

BUG-1 and BUG-3 together make the Gaussian end-to-end pipeline non-functional:
BUG-3 crashes during parameter initialization, and even if `R` were fixed,
BUG-1 would produce wrong-rank `J` tensors during filtering.

~~BUG-2 affects the Gaussian CVI loop but is partially masked by the conjugate
structure.~~ **Fixed** — `Gaussian.infer` now uses `super().infer(...)` which
preserves the correct `cls` binding to `Gaussian`.

~~BUG-4 is independent and only affects the Poisson likelihood.~~ **Fixed** —
the erroneous `-0.5` factor in `poisson_cvi_bin_stats` has been removed,
aligning moment recovery with the information-form convention used by the
rest of the codebase.

BUG-5 affects both likelihoods through `trial_info_repr` but is easily
worked around by passing `d` as `d[:, None]`.
