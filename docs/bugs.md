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

## BUG-3: `Gaussian.initialize_params` creates 1D `R` — FIXED

| | |
|---|---|
| **File** | `src/cvhmax/cvi.py:460` |
| **Severity** | High — crashes the Gaussian E2E pipeline |
| **Affects** | `Gaussian.initialize_params` → `bin_info_repr` |
| **Test** | `tests/test_cvi.py::test_gaussian_e2e` |
| **Status** | Fixed |

### Description

`Gaussian.initialize_params` created the observation noise as:

```python
R = jnp.zeros(y.shape[-1])  # shape (N,) — 1D vector
```

This `R` was later passed to `bin_info_repr` which calls
`jnp.linalg.solve(R, C)`. `solve` requires a 2D matrix for its first
argument; a 1D vector raises a shape error.

### Root cause

`R` should be a 2D covariance matrix `(N, N)`, not a 1D vector.

### Fix

Initialize `R` as an identity matrix:

```python
R = jnp.eye(y.shape[-1])
```

The initial value is overwritten by `ridge_estimate` in the first
`update_readout` call, so the exact scale does not matter — it only
affects the initial `initialize_info` call.

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

## BUG-6: `update_readout` uses `vstack` on `ymask` — FIXED

| | |
|---|---|
| **File** | `src/cvhmax/cvi.py:413–414` (Gaussian), `cvi.py:678–679` (Poisson) |
| **Severity** | High — shape mismatch crashes Gaussian pipeline |
| **Affects** | `Gaussian.update_readout`, `Poisson.update_readout` |
| **Test** | `tests/test_cvi.py::test_gaussian_e2e` |
| **Status** | Fixed |

### Description

Both `Gaussian.update_readout` and `Poisson.update_readout` used
`jnp.vstack` to flatten the trial dimension:

```python
y = jnp.vstack(y)           # (trials, T, N) → (trials*T, N) ✓
ymask = jnp.vstack(ymask)   # (trials, T) → (trials, T) ✗
m = jnp.vstack(m)           # (trials, T, L) → (trials*T, L) ✓
```

`jnp.vstack` on a 3D array treats the first axis as a list of 2D arrays
and concatenates along axis 0 → correct 2D result. But `jnp.vstack` on
a **2D** array treats the first axis as a list of 1D arrays, promotes
each to 2D (adding a leading axis), then stacks — returning the **same
2D shape**. For `ymask` with shape `(1, T)`, `vstack` returns `(1, T)`
instead of the expected `(T,)`.

When this `(1, T)` mask reached `ridge_estimate`, `expand_dims` produced
`(1, T, 1)` instead of `(T, 1)`, causing a shape mismatch in the
`m1.T @ y` matrix multiply (contracting `(1,)` vs `(T,)`).

### Fix

Replaced `vstack` with explicit `reshape`/`ravel`:

```python
y = y.reshape(-1, y.shape[-1])
ymask = ymask.ravel()
m = m.reshape(-1, m.shape[-1])
```

---

## BUG-7: `poisson_trial_nell` spurious `expand_dims` — FIXED

| | |
|---|---|
| **File** | `src/cvhmax/cvi.py:507` |
| **Severity** | Low — masked by all-ones mask in tests |
| **Affects** | `poisson_trial_nell` masking |
| **Test** | Latent; would produce wrong loss with partial masks |
| **Status** | Fixed |

### Description

`poisson_trial_nell` applied the mask with:

```python
bin_nells = jnp.where(jnp.expand_dims(ymask, -1), bin_nells, 0)
```

`bin_nells` is `(T,)` (one scalar per bin from `vmap`). With `ymask`
shape `(T,)`, `expand_dims(ymask, -1)` produces `(T, 1)`. Broadcasting
`(T, 1)` with `(T,)` yields `(T, T)` — a square matrix instead of a
vector. With all-ones masks the numerical result was correct (all entries
True → identity), but with partial masks the sum would be wrong.

### Fix

Removed the unnecessary `expand_dims`:

```python
bin_nells = jnp.where(ymask, bin_nells, 0)
```

---

## BUG-8: `em_step` carry unpack order mismatch — FIXED

| | |
|---|---|
| **File** | `src/cvhmax/cvhm.py:220` |
| **Severity** | Critical — smoother uses wrong information, posterior is all zeros |
| **Affects** | Entire EM loop for both Gaussian and Poisson |
| **Test** | `tests/test_cvi.py::test_gaussian_e2e` |
| **Status** | Fixed |

### Description

The EM carry tuple has order:

```python
carry = (params, z, Z, j, J, m, V, nell)
#        0       1  2  3  4  5  6  7
```

But `em_step` unpacked it as:

```python
params, j, J, *_ = carry
```

This assigned `z` (smoothed information vectors) to `j` and `Z`
(smoothed information matrices) to `J`. The smoother then received
wrong pseudo-observations, producing a zero posterior.

For Poisson, this bug was masked because `update_pseudo` overwrites
`j, J` based on the current smoothed latents inside `CVI.infer`, so
the wrong initial values were corrected within the CVI iterations.

For Gaussian, `update_pseudo` is a no-op (conjugate structure), so the
wrong values propagated unchanged and the smoother always used stale
information from the initial `initialize_info` call.

### Fix

Corrected the unpack to match the carry order:

```python
params, _, _, j, J, *_ = carry
```

---

## BUG-9: Missing info refresh after readout update — FIXED

| | |
|---|---|
| **File** | `src/cvhmax/cvhm.py:240` |
| **Severity** | High — Gaussian EM does not converge |
| **Affects** | Gaussian EM loop |
| **Test** | `tests/test_cvi.py::test_gaussian_e2e` |
| **Status** | Fixed |

### Description

For Gaussian readouts, the pseudo-observations `(j, J)` are a
deterministic function of the readout parameters `(C, d, R)`:

```
J = H^T R^{-1} H       j = H^T R^{-1} (y - d)
```

After `update_readout` changes `C, d, R` in the M-step, the old `j, J`
become stale. The carry passed these stale values to the next EM
iteration, so the smoother always used the initial FA-derived
information. The M-step updated the readout parameters but the E-step
never reflected those updates → the posterior was stuck at the initial
estimate.

For Poisson, this was not an issue because the CVI iterations inside
`infer` maintain pseudo-observations via `update_pseudo`.

### Fix

Added a `initialize_info` call at the end of `em_step` to refresh
`(j, J)` from the updated params:

```python
j, J = vmap(self.cvi.initialize_info, in_axes=(None, 0, 0, None, None))(
    params, y, ymask, Af, Qf
)
```

---

## Summary table

| ID | Location | Severity | Category | Status | Test |
|----|----------|----------|----------|--------|------|
| BUG-1 | `cvi.py:385` / `cvhm.py:215` | High | Shape | **Fixed** | `test_gaussian_initialize_info_shape` |
| BUG-2 | `cvi.py:310` | High | Dispatch | **Fixed** (`4439022`) | `test_gaussian_infer_single_cvi_iter` |
| BUG-3 | `cvi.py:460` | High | Shape | **Fixed** | `test_gaussian_e2e` |
| BUG-4 | `cvi.py:525` | Medium | Convention | **Fixed** (`4439022`) | `test_poisson_cvi_bin_stats_convention` |
| BUG-5 | `utils.py:130` | Medium | Broadcasting | **Fixed** | `test_trial_info_repr_analytic` |
| BUG-6 | `cvi.py:413` | High | Shape | **Fixed** | `test_gaussian_e2e` |
| BUG-7 | `cvi.py:507` | Low | Masking | **Fixed** | Latent |
| BUG-8 | `cvhm.py:220` | Critical | Logic | **Fixed** | `test_gaussian_e2e` |
| BUG-9 | `cvhm.py:240` | High | Logic | **Fixed** | `test_gaussian_e2e` |

### Interaction between bugs

All nine bugs have been fixed. The Gaussian and Poisson end-to-end
pipelines are now fully functional.

BUG-3, BUG-6, BUG-8, and BUG-9 together made the Gaussian pipeline
completely non-functional. Fixing BUG-3 alone exposed BUG-6 (shape
mismatch in `vstack`), and fixing that exposed BUG-8 (carry order) and
BUG-9 (missing info refresh). All four had to be fixed together to get
`test_gaussian_e2e` passing.

BUG-8 also affected Poisson, but was masked by the CVI iteration loop
which corrects stale pseudo-observations via `update_pseudo`.

BUG-7 is latent — it would only manifest with partial observation masks
in Poisson readout fitting.
