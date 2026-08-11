# Algorithms

This page outlines the core algorithms used by cvhmax.

> **Dimension conventions.** This page uses `N` (observation dimension),
> `K` (latent dimension, number of GP kernels), and `L` (state dimension,
> SDE state = `2 * sum(nple)`).  See `data-model.md` for the full glossary.

## Three-Way Separation

CVI works in latent space `(K)`, filtering works in state space `(L)`,
and CVHM bridges the two via the selection mask `M`.  See
`architecture.md` for the full design rationale, data flow diagram, and
conversion formulas.

## CVI-EM Loop

The `CVHM.fit` method alternates between:

1) CVI pseudo-observation refresh: `initialize_info` in latent space
2) Forward-filter warm-up: lift → forward filter → project predictions → `update_pseudo`
3) CVI iterations: lift → bifilter → project → `update_pseudo` (repeated)
4) Readout update: `update_readout` in latent space

```
CVHM loop:
    j, J = CVI.initialize_info(params, y, valid_y)              # latent (per-bin)
    j_s, J_s = lift(j, J, M)                                    # latent → state
    zp, Zp = forward_filter(j_s, J_s, ...)                      # state (predicted)
    m_w, V_w = project(zp, Zp, M)                               # state → latent
    j, J = CVI.update_pseudo(params, y, valid_y, m_w, V_w, ...) # warm-up
    for each CVI iteration:
        j_s, J_s = lift(j, J, M)                                # latent → state
        z, Z = bifilter(j_s, J_s, ...)                          # state
        m, V = project(z, Z, M)                                  # state → latent
        j, J = CVI.update_pseudo(params, y, valid_y, m, V, ...) # latent
    params = CVI.update_readout(params, y, valid_y, m, V)        # latent
```

### Convergence detection

`CVHM.fit` always performs convergence detection and stops either when the
configured criteria are met or when it reaches the hard `max_iter` cap. Set
`tol` to a positive value to control the threshold.
Stopping uses training-fit state only; held-out likelihood or predictive
metrics are not used.

After each outer iteration, the implementation compares three readout
parameter quantities:

- Procrustes-aligned loading matrix change (`C`), accounting for the GLM
  latent sign/permutation/orthogonal rotation gauge;
- readout bias change (`d`); and
- observation-noise covariance change (`R`); `R` is always a JAX-compatible
  array/scalar, with Poisson using a preserved scalar zero sentinel so this
  metric is exactly zero.

The latent GP prior, kernels, time step, and data are fixed during `fit`. Thus,
identical readout parameters deterministically imply the same posterior up to
numerical error. The loading comparison aligns successive `C` matrices with
orthogonal Procrustes; the detector does not compare full latent posterior
arrays or map them into observation space.

Each change is normalized by `max(1, ||reference||)`. Convergence requires the
maximum of these three changes to be at most `tol` for
`convergence_patience` consecutive iterations after at least `min_iter`
iterations. The first iteration is always unconditional because no previous
posterior exists for comparison. Non-finite metrics never pass. If the cap is
reached first, the fit is marked non-converged.

The diagnostics are available after fitting as `model.converged_` and
`model.n_iter_`. There is no fixed-iteration mode that disables convergence
detection.

The forward-filter warm-up provides sequentially coherent initial
pseudo-observations — each bin's starting point incorporates causal
information from all prior bins.  This is a state-space operation owned
by CVHM, keeping CVI free of dynamics knowledge.

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

When a bin is masked (`valid_y = 0`), both `j` and `J` are set to zero by
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

**This codebase uses information form, not natural parameters.** The
correct conversion is `μ = Z⁻¹ z` with no `−0.5` factor.

## Information-Form Filtering

Filtering is performed in information form (precision matrices). The forward and backward passes are combined by `bifilter` to obtain smoothed latents.

Key code:

- `information_filter_step` (forward update)
- `information_filter` (scan over time)
- `bifilter` (merge forward and backward results)

Source: `src/cvhmax/filtering.py`

## Hida-Matern Kernels

`HidaMatern.kernel(tau)` evaluates the real scalar covariance

    sigma^2 * matern(abs(tau), rho, order) * cos(omega * tau)

without state derivatives or jitter. The state-space implementation uses the
related complex auxiliary kernel

    k_z(tau) = sigma^2 * matern(abs(tau), rho, order) * exp(i * omega * abs(tau))

and recovers the real process through `Re(k_z)`. This is an exact
representation of the real oscillatory covariance, since
`cos(omega * tau) = Re(exp(i * omega * tau))`.

The complex form is not an assumption that the observed GP is complex. It is
a compact realization of the two conjugate spectral components centered at
`+omega` and `-omega`. A direct real realization carries both components
explicitly, whereas one complex state represents the pair. The number of
real degrees of freedom is unchanged: one complex state corresponds to two
real states. In this repository, `real_repr` converts the complex state
matrices to an equivalent real block representation for real-valued filtering.

The numerical benefit comes from the resulting algebraic structure and from
state scaling, rather than from complex arithmetic by itself. The covariance
blocks are Hermitian and obey conjugate-symmetry constraints, which avoids
redundant calculations and enables structure-preserving operations. The
stationary block is factored with Cholesky before dynamics are derived; a
bounded machine-scale jitter ladder is used only if that factorization is not
finite. The paper also proposes a correlation transform that normalizes
derivative states using
the stationary variances, reducing the condition number of `K(0)` before
filtering. Without this scaling, high-order derivative covariance matrices can
be badly conditioned even in the complex representation. `CVHM` applies this transform when constructing its filtering dynamics and
stationary prior. The standalone `HidaMatern.Af/Qf/Ab/Qb/K` methods remain
unscaled so they continue to expose the raw kernel state coordinates. Their
dynamics use the jittered stationary block and raw positive-lag block; the
Cholesky fallback is a numerical safeguard, not a replacement for correlation
scaling.

The kernel order determines the per-kernel complex state dimension
`nple = order + 1`. The total state dimension is `L = 2 * sum(nple)`
across all kernels (factor of 2 from the real-valued representation).

The kernel API has two layers with deliberately different roles:

- `Ks(kernelparam, tau)` is the canonical JAX functional path. Its mapping
  parameter container is a pytree, so it is suitable for `jax.jit`, `jax.vmap`,
  and `jax.scan`. It returns the raw complex state covariance.
- `HidaMatern.K(tau)` is a convenience wrapper. It packs the dataclass fields,
  delegates to `Ks`, and adds the object's optional instantaneous state-space
  component `s I` only at zero lag. Positive-lag cross-covariances remain raw;
  with `s=0`, it agrees with `Ks` up to dtype.
- Functional dynamics use the jittered stationary block and raw positive-lag
  cross-covariance. Cholesky solves are used for the stationary block, with a
  bounded machine-scale fallback ladder. Derived process-noise blocks are
  Hermitian-symmetrized but are not eigenvalue-clipped. Correlation scaling
  remains the primary high-order conditioning mechanism. The default practical
  path is `JAX_ENABLE_X64=1`, correlation scaling, and `s=1e-5`.

Orders 0, 1, and 2 use built-in closed-form implementations. Higher orders
are rejected explicitly because this package does not validate their
state-space construction. The inference engine remains extensible through
direct kernel-object injection: users can pass custom objects implementing the
kernel interface required by `CVHM` without modifying the built-in Hida–Matern
implementation.

This representation follows Dowling, Sokół, and Park, “Hida–Matérn Kernel,”
arXiv:2107.07098, especially the complex decomposition and state-space
construction in Eqs. 25–31 and the conditioning discussion in Section 6.
