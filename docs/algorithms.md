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
4) Readout update: `Estimator.estimate` in latent space

```
CVHM loop:
    j, J = CVI.initialize_info(params, y, valid_y)              # latent (per-time-step)
    j_s, J_s = lift(j, J, M)                                    # latent → state
    zp, Zp = forward_filter(j_s, J_s, ...)                      # state (predicted)
    m_w, V_w = project(zp, Zp, M)                               # state → latent
    j, J = CVI.update_pseudo(params, y, valid_y, m_w, V_w, ...) # warm-up
    for each CVI iteration:
        j_s, J_s = lift(j, J, M)                                # latent → state
        z, Z = bifilter(j_s, J_s, ...)                          # state
        m, V = project(z, Z, M)                                  # state → latent
        j, J = CVI.update_pseudo(params, y, valid_y, m, V, ...) # latent
    params = Estimator.estimate(CVI, params, y, valid_y, m, V)   # latent
```

### Convergence detection

`CVHM.fit` always performs convergence detection and stops either when the
configured criteria are met or when it reaches the hard `max_iter` cap. Set
`tol` to a positive value to control the threshold.
Stopping uses training-fit state only; held-out likelihood or predictive
metrics are not used.

After each outer iteration, the implementation compares three readout
parameter quantities:

- change in the loading Gram matrix `C C^T`, which is unchanged by any latent
  rotation `C -> C Q` and so is blind to the GLM sign/permutation/orthogonal
  rotation gauge. Predictions depend on `C` only through `C C^T`, so this asks
  whether the fitted model has stopped changing rather than the basis it is
  written in;
- readout bias change (`d`); and
- observation-noise covariance change (`R`); `R` is always a JAX-compatible
  array/scalar, with Poisson using a preserved scalar zero sentinel so this
  metric is exactly zero.

The latent GP prior, kernels, time step, and data are fixed during `fit`. Thus,
identical readout parameters deterministically imply the same posterior up to
numerical error. The detector does not compare full latent posterior arrays or
map them into observation space.

The Gram comparison applies to every estimator. A per-column prior such as ARD
breaks the rotation symmetry of the objective and so rotates the loadings slowly
toward orthogonal columns; that rotation changes `C` and the fitted precisions
without changing the fitted mean, and the Gram ignores it. A relative change
`delta` in `C` appears as roughly `2 * delta` in `C C^T`, so `tol` is about twice
as strict on the loadings as a direct comparison of `C` would be.

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
pseudo-observations — each time step's starting point incorporates causal
information from all prior time steps.  This is a state-space operation owned
by CVHM, keeping CVI free of dynamics knowledge.

Gaussian and Poisson readouts use the fitted latent-space loading `Params.C`.
The state-space observation matrix is `H = C @ M`: CVHM applies `M` when
lifting the latent information increments into the SDE state space. The fixed
GP kernel amplitudes set the prior scale of the latent processes; `C` retains
observation-specific gains. The convergence metric compares the same loading
matrix. After the last readout update, `fit()` performs a final inference pass:
`latent` and `posterior` describe the posterior under the final parameters.

Gaussian observation noise is diagonal. For observed feature mask `w` and
variance vector `r = diag(R)`, Gaussian information increments are
`J = C.T @ diag(w / r) @ C` and `j = C.T @ (w * (y - d) / r)`. The readout
update, under the default `RidgeEstimator`, solves a ridge objective per
feature from the masked first and second moments, then refreshes that feature's
noise variance from its expected residual squares. The penalty is `alpha_k` on
each loading and **zero on the bias**: the prior is
`C[:, k] ~ N(0, alpha_k^-1)` with a flat bias, so the fit does not shrink the
baseline rate. Poisson CVI and the Poisson readout objective likewise sum only
observed feature contributions.
Initialization fills missing entries with per-feature observed means for Factor
Analysis only; subsequent updates use the actual observation mask. For Poisson,
FA count-space loadings are scaled to log-rate space, with a half-count per
feature used only to initialize a finite baseline rate. The Poisson
readout objective is the exact negative expected log-likelihood,
`exp(eta + q/2) - y * eta` with `eta = C m + d` and `q = diag(C V C^T)`; only
the rate carries the variance correction. Masking does not change it. It
adds a squared L2 penalty on the loading matrix, averaged with the data term,
and clamps the log rate at `MAX_LOGRATE` to bound the rate.

## Gaussian ARD Loadings

`ARDEstimator` fits Gaussian loadings under a per-latent automatic relevance
determination prior. It is a conditional analysis: the latent moments
are held fixed, and it answers which latent columns the readout needs, not which
latents the latent prior supports.

The construction uses the fact that the expected readout log-likelihood is
quadratic in `beta = [d, C]`. With `q(z) = N(m, V)`,

```
E_q[log p(y | z, C, d)] = -0.5 sum_t [log(2 pi R) + (y - C m - d)^2 / R + C V C / R]
```

so `exp(E_q[log p])` is a Gaussian kernel in `beta` and the integral against
`C[:, k] ~ N(0, alpha_k^-1 I)` is available in closed form. The prior on the
bias is flat. The evidence is therefore

```
log p(y | alpha) = sum_n [ -0.5 N_n log(2 pi R_n) - 0.5 sum_t y^2 / R_n
                           - 0.5 log|Prec_n / R_n + diag(0, alpha)|
                           + 0.5 c_n' (Prec_n / R_n + diag(0, alpha))^-1 c_n ]
                   + 0.5 N (sum_k log(alpha_k / 2 pi) + (K + 1) log 2 pi)
```

where `Prec_n` and `c_n` are the sufficient statistics built from `y`, `m`, and
`V`, and `R_n` is the diagonal observation variance. Because `E_q[log p]` is a
Jensen *lower* bound on `log E_q[p]`, the value is a lower bound on the evidence
at the supplied moments.

Precisions and observation variances are both maximised against the evidence,
jointly, mirroring the evidence-based noise update used in the ARD literature.
The maximisers are reached by the fixed-point iteration described under
*Loading Estimators*, so no optimiser runs inside the EM loop.

Sparsity is *not* exact. A fitted precision plateaus at a large but finite
value and the column is shrunk towards zero rather than deleted, because the
evidence tends to the reduced model's evidence as a precision grows, and with
fitted latent moments a small non-zero loading can beat removal. Exact zeros
require an explicit selection step; this package reports the precisions and
leaves that choice to the caller. The reference ARD implementation behaves the
same way: with pruning disabled its precisions plateau, and its exact zeros
come from a hard precision threshold rather than from divergence.

ARD and ridge regularise the same block of loadings, so choosing between them is
the choice of `estimator`, not a second update step.

With `V = 0` the Jensen bound is tight: `E_q[log p]` becomes the ordinary
Gaussian log-likelihood at the latent means, so the evidence is the exact
marginal likelihood of an ARD linear regression on that design and the fit
matches a standard ARD regression. With `V > 0` the estimator maximises a lower
bound instead, because the latents are uncertain.

The bias is a parameter of this model, integrated under a flat prior, so the
fitted observation variance carries residual degrees of freedom
`n - sum(gamma) - 1`. An implementation that centers the design away has no bias
column in its Gram and uses `n - sum(gamma)`; the two agree as `n` grows. The
relevance precisions are shared across features here, so the structure is
hierarchical rather than one ARD regression per feature.

## Loading Estimators

The observation model and the loading estimator are separate axes. A `CVI`
subclass is an observation model: it supplies the E-step, a conjugate loading
representation when one exists, and a readout objective. An `Estimator` owns the
readout M-step and the loading prior. Ridge and ARD share one prior,
`C[:, k] ~ N(0, alpha_k^-1 I)` with a flat bias, and differ in whether `alpha` is
fitted -- so choosing an estimator chooses a prior as well as an algorithm.

An estimator uses the closed form when the likelihood exposes conjugate loading
terms and a generic gradient solve otherwise. `Gaussian` exposes them; `Poisson`
does not, because `exp(eta)` breaks conjugacy in the loadings, so
`ARDEstimator` is rejected with Poisson at construction.

`RidgeEstimator` solves the penalised conjugate objective directly. Note that
solving each feature against its own noise variance brings an `R_nn` factor into
the regulariser; that is the unit conversion for an absolute precision, not a
different penalty strength.

`ARDEstimator` maximises the evidence over the precisions and the observation
variances jointly, with the loadings integrated out. Both stationary points have
closed forms and are reached by a fixed-point iteration, so no optimiser is
involved:

```
gamma_nk = 1 - alpha_k * Sigma_n[k, k]
alpha_k  = (sum_n gamma_nk + 2a - 2) / (sum_n C[n, k]^2 + 2b)
R_n      = RSS_n / (N_n - sum_k gamma_nk)
```

with `Sigma_n = (Prec_n / R_n + diag(0, alpha))^-1` and `RSS_n` the expected
residual sum of squares including latent uncertainty. The observation variance
is estimated from expected residuals alone, with the effective parameter count
removed; it needs no prior, and its denominator is bounded away from zero
whenever a feature has more observed time steps than parameters.

The loading precisions do need one. `gamma_k` and `||C[:, k]||^2` vanish
together as a column is pruned, so the bare rule is a `0 / 0` limit; measured
over a grid of problems it diverges or reaches precisions of order `1e17` in a
majority of cases. `(a, b)` is a weak Gamma prior -- the reference ARD
regression's defaults -- that removes the corner. It is a guard on the updater
rather than a modelling claim, which is why the shape and rate are not meant to
be tuned: the fit is insensitive to them across many orders of magnitude, and
their only visible effect is how large an inactive column's precision
plateaus. Precisions are shared across features, because a latent is one shared object here, so this is a
hierarchical ARD rather than one ARD regression per feature.

An irrelevant column's loadings are driven to numerical zero -- as small as
`1e-130` in a full fit -- but the column is not deleted from `C`, and its precision
plateaus at a large finite value that the guard prior's constants set, not the
data, because the evidence tends to the reduced model's evidence as a precision
grows. The precisions are reported on `Params.alpha` and any threshold is left
to the caller. What makes a threshold workable is the size of the gap, from one
to several orders of magnitude; the absolute value is gauge dependent, so a
fixed threshold does not transfer between fits. Without the ARD estimator, fixed
ridge only shrinks surplus latents to a small nonzero norm.

At `V = 0` the Jensen bound is tight, so the evidence is the exact marginal
likelihood of an ARD linear regression on the latent means and the fit matches a
standard ARD regression up to one residual degree of freedom: the bias is a
parameter of this model and costs one, whereas an implementation that centers
the design away uses one fewer.

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
- Readout updates (`Estimator.estimate`)
- Poisson readout objective (`poisson_readout_objective`)

### Observation information increments `(j, J)`

Same convention as `(z, Z)`:

```
J = Hᵀ R⁻¹ H       observation information matrix (positive semi-definite)
j = Hᵀ R⁻¹ (y − d)  observation information vector
```

Computed by `step_info_repr` in `utils.py` (with `sequence_info_repr` and
`batch_info_repr` as vmapped wrappers over the time and sequence axes,
respectively). In the Poisson CVI path, `(j, J)` are pseudo-observations updated
iteratively rather than computed from a closed-form likelihood.

When an entire time step is masked (`valid_y = 0` for every feature), `j` and `J`
are zero, so the filtering update is a no-op. Smoothing may still use
observations from other time steps. A partially masked time step contributes
information only from its observed features.

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
