# Algorithms

This page outlines the core algorithms used by cvhmax.

## CVI-EM Loop

The `CVHM.fit` method alternates between:

1) CVI smoothing: update pseudo-observations and smooth latents
2) Readout update: refit the observation model parameters

The inner loop performs several CVI iterations before each outer EM update.

## Gaussian Parameterizations

The codebase uses two parameterizations of multivariate Gaussians:
**information form** for filtering/smoothing and **mean-variance (moment)
form** for observation model updates and loss computation.

### Information form `(z, Z)`

| Symbol | Definition | Properties |
|---|---|---|
| `Z = Σ⁻¹` | Precision matrix | Positive definite |
| `z = Σ⁻¹ μ` | Precision-weighted mean | — |

Convert to moments:

```
μ = Z⁻¹ z       (code: solve(Z, z))
Σ = Z⁻¹          (code: inv(Z))
```

Where it appears:

- `filtering.py` — predict/update steps, `bifilter` smoother
- `cvhm.py` — initial conditions (`z0 = 0`, `Z0 = inv(Q0)`), `sde2gp` (`solve(Z, z)`, `inv(Z)`)
- `cvi.py` — CVI loop inputs/outputs, pseudo-observation updates

Key property: the filter update is additive.

```
z_post = z_pred + j
Z_post = Z_pred + J
```

### Mean-variance (moment) form `(m, V)`

| Symbol | Definition | Properties |
|---|---|---|
| `m` (or `μ`) | Mean vector | — |
| `V` (or `Σ`) | Covariance matrix | Positive definite |

Where it appears:

- `sde2gp` output — converts information form to `(m, V)` for downstream use
- Readout/observation model updates (`update_readout`)
- Loss computation (`poisson_trial_nell`, `gaussian_trial_nell`)

### Observation information increments `(j, J)`

Same convention as `(z, Z)`:

```
J = Hᵀ R⁻¹ H       observation information matrix (positive semi-definite)
j = Hᵀ R⁻¹ (y − d)  observation information vector
```

Computed by `bin_info_repr` in `utils.py` (with `trial_info_repr` and `batch_info_repr` as vmapped wrappers over time and trial axes respectively). In the Poisson CVI path, `(j, J)` are pseudo-observations updated iteratively rather than computed from a closed-form likelihood.

When a bin is masked (`ymask = 0`), both `j` and `J` are set to zero by
`bin_info_repr`, so the filter update `Z_post = Z_pred + J`,
`z_post = z_pred + j` reduces to a no-op at that bin — the posterior
equals the prediction.

### Variable name glossary

| Variable | Form | Meaning |
|---|---|---|
| `z, Z` | Information | Posterior information vector / matrix |
| `z0, Z0` | Information | Prior (initial) information vector / matrix |
| `zp, Zp` | Information | Predicted (prior for current step) |
| `zf, Zf` | Information | Forward-filtered |
| `zpb, Zpb` | Information | Backward-predicted |
| `j, J` | Information | Observation information increments |
| `k, K` | Information | CVI pseudo-observation gradient updates |
| `m, V` | Moment | Posterior mean / covariance |
| `P` | Information | State noise precision `Q⁻¹` |

### Note on exponential-family natural parameters

The exponential-family natural parameters `(η₁, η₂)` for a Gaussian are
`η₁ = Σ⁻¹ μ` and `η₂ = −½ Σ⁻¹`. These differ from information form by a
factor of `−½` on the matrix component (`η₂ = −½ Z`). The moment recovery
formulas are `μ = −½ η₂⁻¹ η₁` and `Σ = −½ η₂⁻¹`.

**This codebase does not use natural parameters.** The `−0.5` factor in
`poisson_cvi_bin_stats` (`cvi.py:525`) is a bug (BUG-4, see `docs/bugs.md`)
where the natural-parameter formula was applied to information-form inputs.
The correct conversion is `μ = Z⁻¹ z` with no `−0.5` factor.

## Information-Form Filtering

Filtering is performed in information form (precision matrices). The forward and backward passes are combined by `bifilter` to obtain smoothed latents.

Key code:

- `information_filter_step` (forward update)
- `information_filter` (scan over time)
- `bifilter` (merge forward and backward results)

Source: `src/cvhmax/filtering.py`

## Hida-Matern Kernels

`HidaMatern` represents Matern kernels modulated by a complex exponential
as linear Gaussian state-space models:

    k(tau) = sigma^2 * matern(nu, tau, rho) * exp(i * omega * tau)

The kernel order determines the SSM state dimension `M = order + 1`.
Orders 0 and 1 use hand-coded closed-form expressions (`Ks0`, `Ks1`).
Higher orders are handled by the `kernel_generator` subpackage, which
symbolically differentiates the kernel and converts the result to JAX
functions at runtime via `sympy2jax`.

### Kernel generator internals

For an order-M kernel, the generator:

1. Builds the symbolic Hida-Matern covariance using SymPy.
2. Computes successive derivatives `d^p k / d tau^p` for `p = 0..2M-1`
   and their limits at `tau -> 0+`.
3. Assembles the M x M state-space covariance matrix `K_hat(tau)`:
   - Outer entries (row 0, last column): `K_hat[r,c] = (-1)^c * d^{r+c}k/dtau^{r+c}`
   - Inner entries via antisymmetry: `K_hat[r,c] = -K_hat[r-1, c+1]`
4. Converts each entry to a JAX-traceable function using `sympy2jax.SymbolicModule`.
5. Handles the `tau = 0` limit case with `jnp.where`.

The SSM dynamics are derived from the kernel blocks:
- Forward transition: `A = K(tau) @ K(0)^{-1}`
- Process noise: `Q = K(0) - K(tau) @ K(0)^{-1} @ K(tau)^H`
- These satisfy the Lyapunov equation: `A @ K(0) @ A^H + Q = K(0)`

Generator instances are cached by order via `make_kernel(order)`, so the
symbolic computation cost is paid once per order per process lifetime.

Key code:

- `HidaMatern.Af/Qf/Ab/Qb` for transitions and noise
- `HidaMatern.K(tau)` for stationary covariances
- `kernel_generator.HidaMaternKernelGenerator` for arbitrary-order kernels
- `kernel_generator.make_kernel(order)` cached factory

Sources: `src/cvhmax/hm.py`, `src/cvhmax/kernel_generator/`

See `kernel-generator.md` for usage examples and integration patterns.


