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
    params, y, valid_y, Af, Qf
)
```

With `y` shaped `(trials, T, N)`, the outer vmap strips the trial axis, so
each `initialize_info` call receives `y` of shape `(T, N)`.

`Gaussian.initialize_info` (`cvi.py:385`) then applied a second vmap:

```python
return vmap(partial(bin_info_repr, C=H, d=d, R=R))(y, valid_y)
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

- `bin_info_repr(y, valid_y, C, d, R)` — single bin, `y` shape `(N,)`
- `trial_info_repr(y, valid_y, C, d, R)` — vmaps `bin_info_repr` over time,
  `y` shape `(T, N)`
- `batch_info_repr(y, valid_y, C, d, R)` — vmaps `trial_info_repr` over
  trials, `y` shape `(trials, T, N)`

`Gaussian.initialize_info` now calls `trial_info_repr` directly:

```python
return trial_info_repr(y, valid_y, H, d, R)
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
return CVI.infer(params, j, J, y, valid_y, z0, Z0, smooth_fun, smooth_args, 1, lr)
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
return super().infer(params, j, J, y, valid_y, z0, Z0, smooth_fun, smooth_args, 1, lr)
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

## BUG-6: `update_readout` uses `vstack` on `valid_y` — FIXED

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
valid_y = jnp.vstack(valid_y)   # (trials, T) → (trials, T) ✗
m = jnp.vstack(m)           # (trials, T, L) → (trials*T, L) ✓
```

`jnp.vstack` on a 3D array treats the first axis as a list of 2D arrays
and concatenates along axis 0 → correct 2D result. But `jnp.vstack` on
a **2D** array treats the first axis as a list of 1D arrays, promotes
each to 2D (adding a leading axis), then stacks — returning the **same
2D shape**. For `valid_y` with shape `(1, T)`, `vstack` returns `(1, T)`
instead of the expected `(T,)`.

When this `(1, T)` mask reached `ridge_estimate`, `expand_dims` produced
`(1, T, 1)` instead of `(T, 1)`, causing a shape mismatch in the
`m1.T @ y` matrix multiply (contracting `(1,)` vs `(T,)`).

### Fix

Replaced `vstack` with explicit `reshape`/`ravel`:

```python
y = y.reshape(-1, y.shape[-1])
valid_y = valid_y.ravel()
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
bin_nells = jnp.where(jnp.expand_dims(valid_y, -1), bin_nells, 0)
```

`bin_nells` is `(T,)` (one scalar per bin from `vmap`). With `valid_y`
shape `(T,)`, `expand_dims(valid_y, -1)` produces `(T, 1)`. Broadcasting
`(T, 1)` with `(T,)` yields `(T, T)` — a square matrix instead of a
vector. With all-ones masks the numerical result was correct (all entries
True → identity), but with partial masks the sum would be wrong.

### Fix

Removed the unnecessary `expand_dims`:

```python
bin_nells = jnp.where(valid_y, bin_nells, 0)
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
    params, y, valid_y, Af, Qf
)
```

---

## BUG-10: `latent_mask` stride-2 indexing assumes `order=1` — FIXED

| | |
|---|---|
| **File** | `src/cvhmax/cvhm.py:129` |
| **Severity** | Medium — wrong posterior extraction for multi-component models with `order != 1` |
| **Affects** | `CVHM.latent_mask()` → `sde2gp` → posterior mean/covariance |
| **Status** | Fixed |

### Description

`CVHM.latent_mask` constructs the selection matrix `M` that maps latent
components to their corresponding real-valued state-space coordinates.
After `block_diag` of N complex kernel blocks and `real_repr`, the state
vector is laid out as:

```
[Re(kernel_0), ..., Re(kernel_{N-1}), Im(kernel_0), ..., Im(kernel_{N-1})]
```

where each kernel occupies `nple = order + 1` consecutive entries in the
real and imaginary halves.

The current code uses a hardcoded stride of 2:

```python
for i in range(self.n_components):
    M = M.at[i, i * 2].set(1.0)
```

This places 1s at columns 0, 2, 4, … which is only correct when every
kernel has `nple = 2` (i.e., `order = 1`). For other orders or mixed
orders, the mask selects the wrong state-space dimensions.

### Example

Two `order=0` kernels (`nple=1` each). `ssm_dim = 2`, state vector has
4 entries: `[Re(k0), Re(k1), Im(k0), Im(k1)]`.

| Latent | Current (`i*2`) | Selects | Correct column | Should select |
|--------|-----------------|---------|----------------|---------------|
| 0 | 0 | Re(k0) | 0 | Re(k0) |
| 1 | 2 | Im(k0) | 1 | Re(k1) |

Latent 1 picks the imaginary part of kernel 0 instead of the real part
of kernel 1.

### Scope

| Condition | Correct? |
|-----------|----------|
| `n_components = 1` (any order) | Yes — `0 * 2 = 0` matches offset 0 |
| `n_components >= 2`, all `order = 1` | Yes — stride 2 matches `nple = 2` |
| `n_components >= 2`, any `order != 1` | **No** |

### Root cause

Hardcoded stride of 2 instead of accumulating each kernel's `nple`.

### Proposed fix

Replace the stride-2 loop with `nple`-based offset accumulation,
consistent with `utils.latent_mask` (`utils.py:265`):

```python
offset = 0
for i, kernel in enumerate(self.kernels):
    M = M.at[i, offset].set(1.0)
    offset += kernel.nple
```

---

## BUG-11: `ridge_estimate` divides residual covariance by total `T` instead of valid count — FIXED

| | |
|---|---|
| **File** | `src/cvhmax/utils.py:396` |
| **Severity** | Medium — systematically underestimates observation noise when data is partially masked |
| **Affects** | `ridge_estimate` → `Gaussian.update_readout` → observation information `(j, J)` |
| **Test** | — |
| **Status** | Fixed |

### Description

`ridge_estimate` zeros out masked rows of `y` and `m1` before computing
the regression (lines 386–388), so the residual `r = y - m1 @ w` is zero
for masked bins. The sum of squared residuals `r.T @ r` therefore only
accumulates contributions from valid bins — this is correct.

However, the observation noise covariance was computed as:

```python
R = r.T @ r / T
```

where `T` is the total number of rows (valid + masked). Masked rows
contribute zero to the numerator but inflate the denominator, causing `R`
to be underestimated in proportion to the fraction of masked bins.

### Example

With `T = 100` bins and 50 masked (`valid_y` has 50 ones), the sum of
squared residuals comes from 50 bins but is divided by 100, making `R`
roughly half its true value.

### Downstream impact

`R` feeds into `bin_info_repr` (`utils.py:132`):

```python
J = C.T @ solve(R, C)
j = C.T @ solve(R, y - d)
```

An underestimated `R` inflates `J` and `j`, making the filter
overconfident in observations relative to the dynamics prior. The
posterior is biased toward noisy observations and away from the smoothed
latent trajectory. The effect scales with the masked fraction.

### Root cause

Divisor should be the number of valid bins, not the total bin count.

### Fix

Replaced the divisor with the valid count:

```python
n_valid = jnp.maximum(jnp.sum(valid_y), 1.0)
R = r.T @ r / n_valid
```

`jnp.maximum(..., 1.0)` guards against division by zero when all bins
are masked. At this point `valid_y` has shape `(T, 1)` (from
`expand_dims` on line 386), so `jnp.sum(valid_y)` counts the valid bins.

### Note

With fully-observed data (`valid_y` all ones), `n_valid == T` and the
result is unchanged. This is likely why the bug was not caught — tests
typically use fully-observed data.

---

## BUG-12: README example uses wrong parameter name `likelihood` — FIXED

| | |
|---|---|
| **File** | `README.md:73` |
| **Severity** | Low — runtime `TypeError` if user copies the example |
| **Affects** | Documentation only |
| **Status** | Fixed |

### Description

The README quickstart passes `likelihood="Poisson"` to the `CVHM`
constructor:

```python
model = CVHM(
    n_components=n_latents,
    dt=dt,
    kernels=kernels,
    likelihood="Poisson",  # or "Gaussian"
    max_iter=5,
)
```

The `CVHM` dataclass (`cvhm.py:52`) defines the field as
`observation: str = "Gaussian"`. Using `likelihood=` raises:

```
TypeError: CVHM.__init__() got an unexpected keyword argument 'likelihood'
```

`docs/quickstart.md` already uses the correct `observation=` kwarg.

### Fix

Change `likelihood="Poisson"` to `observation="Poisson"` on
`README.md:73`.

---

## BUG-13: Version mismatch between `pyproject.toml` and `__about__.py` — FIXED

| | |
|---|---|
| **File** | `pyproject.toml:3` / `src/cvhmax/__about__.py:4` |
| **Severity** | Low — no runtime impact but confusing for packaging/release |
| **Affects** | Version reporting |
| **Status** | Fixed |

### Description

`pyproject.toml` declares `version = "0.1.0"` while `__about__.py`
declares `__version__ = "0.0.1"`. The build system (hatchling) uses
`pyproject.toml` as the authoritative source, so installed packages get
`0.1.0`. The `__about__.py` file is not imported by any module, so this
is purely a maintenance inconsistency.

### Fix

Align `__about__.py` to `"0.1.0"`.

---

## BUG-14: Impossible `torch>=2.10.0` dev dependency — FIXED

| | |
|---|---|
| **File** | `pyproject.toml:40` |
| **Severity** | Low — prevents installing dev dependencies; all parity tests skipped |
| **Affects** | Dev environment setup, `tests/test_parity.py` |
| **Status** | Fixed |

### Description

The dev dependency group pins `torch>=2.10.0`. PyTorch has not released
version 2.10 (latest stable is ~2.5). This prevents
`uv sync --group dev` or equivalent from resolving torch, which means
the parity tests (all guarded by `@requires_ref`) are always skipped.

### Fix

Lower to a version that exists, e.g. `torch>=2.1.0`.

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
| BUG-10 | `cvhm.py:129` | Medium | Indexing | Fixed | `099c818` |
| BUG-11 | `utils.py:396` | Medium | Masking | **Fixed** | — |
| BUG-12 | `README.md:73` | Low | Docs | **Fixed** | — |
| BUG-13 | `pyproject.toml:3` / `__about__.py:4` | Low | Config | **Fixed** | — |
| BUG-14 | `pyproject.toml:40` | Low | Config | **Fixed** | — |

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
