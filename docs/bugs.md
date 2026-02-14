# Known Bugs

Discovered during parity testing against the PyTorch reference implementation
(`hida_matern_gp_lvms/`). Each bug is documented by one or more tests in
`tests/test_cvi.py` or `tests/test_parity.py` (marked `xfail(strict=True)` or
using explicit `assert not allclose` checks).

---

## BUG-1: Redundant vmap in `Gaussian.initialize_info` — FIXED

| | |
|---|---|
| **File** | `src/cvhmax/cvi.py:385` (inner vmap) + `src/cvhmax/cvhm.py:215` (outer vmap) |
| **Severity** | High — produces wrong-rank output, breaks downstream filtering |
| **Affects** | `Gaussian.initialize_info` → `bin_info_repr` |
| **Test** | `tests/test_cvi.py::test_gaussian_initialize_info_shape` |
| **Status** | Fixed |

### Description

`initialize_info` is called from `CVHM.fit` (`cvhm.py:215`) wrapped in an
outer vmap over the trial axis:

```python
j, J = vmap(self.cvi.initialize_info, in_axes=(None, 0, 0, None, None))(
    params, y, ymask, Af, Qf
)
```

With `y` shaped `(trials, T, N)`, the outer vmap strips the trial axis, so
each `initialize_info` call receives `y` of shape `(T, N)`.

`Gaussian.initialize_info` (`cvi.py:385`) then applied a second vmap:

```python
return vmap(partial(bin_info_repr, C=H, d=d, R=R))(y, ymask)
```

This mapped over the leading axis of `(T, N)`, passing per-time-bin slices of
shape `(N,)` to `bin_info_repr`. The old `bin_info_repr` (then called
`trial_info_repr`) interpreted its first argument as `(T, N)` and did
`jnp.tile(J, (T, 1, 1))` where `T = y.shape[0]` evaluated to `N`
(observation dimension, not time), producing `(T, N, L, L)` instead of
`(T, L, L)`.

### Root cause

`bin_info_repr` was designed to operate on a full `(T, N)` matrix for a
single trial.  `Gaussian.initialize_info` incorrectly vmapped it over
individual time bins, but the trial axis was already stripped by the outer
vmap in `cvhm.py:215`.

### Fix

Refactored the information-representation functions into a clear hierarchy:

- `bin_info_repr(y, ymask, C, d, R)` — single bin, `y` shape `(N,)`
- `trial_info_repr(y, ymask, C, d, R)` — vmaps `bin_info_repr` over time,
  `y` shape `(T, N)`
- `batch_info_repr(y, ymask, C, d, R)` — vmaps `trial_info_repr` over
  trials, `y` shape `(trials, T, N)`

`Gaussian.initialize_info` now calls `trial_info_repr` directly:

```python
return trial_info_repr(y, ymask, H, d, R)
```

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
| **Affects** | `Gaussian.initialize_params` → `bin_info_repr` |
| **Test** | `tests/test_cvi.py::test_gaussian_e2e` (xfail) |

### Description

`Gaussian.initialize_params` creates the observation noise as:

```python
R = jnp.zeros(y.shape[-1])  # shape (N,) — 1D vector
```

This `R` is later passed to `bin_info_repr` which calls
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

## BUG-5: `bin_info_repr` broadcasting with `d` — FIXED

| | |
|---|---|
| **File** | `src/cvhmax/utils.py:130` |
| **Severity** | Medium — silent wrong results when `N != T` |
| **Affects** | `bin_info_repr`, `Gaussian.initialize_info` |
| **Test** | `tests/test_utils.py::test_trial_info_repr_analytic`, `tests/test_parity.py::test_observation_info_parity` |
| **Status** | Fixed |

### Description

The old `bin_info_repr` (then called `trial_info_repr`) computed `y.T - d`
where `y.T` had shape `(N, T)` and `d` had shape `(N,)`. JAX broadcast
`(N,)` as `(1, N)`, producing `(N, T) - (1, N)` which only worked when
`N == T`. When `N != T`, this either raised a shape error or silently
produced wrong results.

### Root cause

The function operated on a full `(T, N)` matrix and transposed `y` before
subtracting `d`, creating a broadcasting mismatch.

### Fix

`bin_info_repr` now operates on a single bin: `y` has shape `(N,)` and `d`
has shape `(N,)`, so `y - d` is element-wise with no broadcasting ambiguity.
The `d[:, None]` workarounds in tests have been removed.

---

## Summary table

| ID | Location | Severity | Category | Status | Test |
|----|----------|----------|----------|--------|------|
| BUG-1 | `cvi.py:385` / `cvhm.py:215` | High | Shape | **Fixed** | `test_gaussian_initialize_info_shape` |
| BUG-2 | `cvi.py:310` | High | Dispatch | **Fixed** (`4439022`) | `test_gaussian_infer_single_cvi_iter` |
| BUG-3 | `cvi.py:434` | High | Shape | **Open** | `test_gaussian_e2e` (xfail) |
| BUG-4 | `cvi.py:525` | Medium | Convention | **Fixed** (`4439022`) | `test_poisson_cvi_bin_stats_convention` |
| BUG-5 | `utils.py:130` | Medium | Broadcasting | **Fixed** | `test_trial_info_repr_analytic` |

### Interaction between bugs

**BUG-3** is the only remaining open bug. It crashes during
`Gaussian.initialize_params` by creating a 1D `R` vector instead of a 2D
covariance matrix, which blocks the Gaussian end-to-end pipeline.

~~BUG-1 and BUG-3 together make the Gaussian end-to-end pipeline
non-functional.~~ **BUG-1 fixed** — `bin_info_repr` now operates per-bin on
`(N,)` input and `Gaussian.initialize_info` calls `trial_info_repr` directly.

~~BUG-2 affects the Gaussian CVI loop but is partially masked by the conjugate
structure.~~ **Fixed** — `Gaussian.infer` now uses `super().infer(...)` which
preserves the correct `cls` binding to `Gaussian`.

~~BUG-4 is independent and only affects the Poisson likelihood.~~ **Fixed** —
the erroneous `-0.5` factor in `poisson_cvi_bin_stats` has been removed,
aligning moment recovery with the information-form convention used by the
rest of the codebase.

~~BUG-5 affects both likelihoods through `bin_info_repr`.~~ **Fixed** —
`bin_info_repr` now operates on a single bin where `y` and `d` are both
`(N,)`, eliminating the broadcasting issue.
