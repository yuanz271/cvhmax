# Likelihood + Loading-Prior Composition

## Goal

Separate the observation model from the loading regularization so the two are
chosen independently:

```text
y ~ Likelihood(C, d, R; z)        z ~ GP(kernels)
C ~ N(0, alpha^-1)   via an Estimator that fits alpha and the loadings
```

Today a `CVI` subclass bundles both. `Gaussian.update_readout` and
`Poisson.update_readout` each carry their own hand-rolled ridge, and the two
disagree. Making the estimator a first-class object gives one convention, makes
ridge the fixed equal-precision special case of ARD, and puts the ARD/ridge
choice in the user's hands instead of in the readout's implementation.

The ARD estimator body already exists on this branch
(`gaussian_ard_estimate`, `gaussian_ard_evidence`), with its conventions settled
and checked against a reference ARD regression. What remains is structural:
moving it, and the ridge solve, behind a common estimator interface.

## Why: the disagreement this started from

Before step 1, the same object — a squared L2 prior on the loadings — was
implemented twice:

| | Gaussian (`utils.ridge_estimate`) | Poisson (`poisson_readout_objective`) |
|---|---|---|
| Bias `d` | penalised | not penalised |
| Form | separate closed-form solver | inside the LBFGS objective |
| Hyperparameter | `lam = 0.1` | `gamma = 1.5` |

One real disagreement, on the bias. An earlier draft of this plan also claimed
the scales disagreed, noise-relative against absolute; that was wrong. Solving
each feature against its own noise variance brings an `R_nn` factor into the
regulariser, so `lam * R_nn` is the unit conversion for an absolute precision
`lam`, not a different penalty strength. Verified by re-deriving the normal
equations.

`lam = 0.1` is inert in practice (~0.05% of the objective), so the Gaussian bias
penalty was small, but it was a modelling claim nobody chose. Step 1 removed
it; what remains is that `alpha` is still a scalar rather than per-column.

## The convention

Settle on the ARD prior as the definition, since it is the general case:

```text
C[:, k] ~ N(0, alpha_k^-1 I)      bias d flat      alpha absolute
```

Then:

- **Gaussian** regularizer becomes `R_nn * diag(0, alpha)` where the current
  code has `lam * R_nn * I_{K+1}`. The closed form is unchanged in structure;
  only the diagonal changes.
- **Poisson** penalty becomes `sum_k alpha_k * ||C[:, k]||^2`, which is the
  current `gamma * ||C||^2` with a per-column precision and a rename.
- **Ridge** is this prior with `alpha` fixed; **ARD** is this prior with
  `alpha` fitted by the evidence.

## Interface

The second axis is an **estimator**, not a prior. Ridge and ARD share one prior
(`C[:, k] ~ N(0, alpha_k^-1 I)`) and differ in whether `alpha` is fitted, which
is an estimation choice rather than a distributional one. Naming them priors
would imply a distinction that is not there, and both carry a solve and, for
ARD, an optimiser. Accordingly the M-step moves out of `CVI`:

```python
class Estimator:
    requires_conjugate_terms: bool
    isotropic: bool                      # equal precisions across columns

    def initialize_params(self, likelihood, y, valid_y, n_factors, *, random_state,
                          params=None)
    def estimate(self, likelihood, params, y, valid_y, m, V) -> tuple[Params, float]
```

`initialize_params` keeps its `params=None` escape hatch: supplying parameters
short-circuits initialisation, which is how a fit is currently frozen at preset
values. The data-driven start itself stays on the likelihood
(`CVI.initialize_params`); the estimator drives it and supplies the penalty, so
only the M-step left `CVI`. See Migration.

Estimation is selected with the exact class name, exactly as
`observation="Gaussian"` selects a likelihood:

```python
CVHM(..., observation="Gaussian", estimator="RidgeEstimator")
```

Registry and serialization therefore key on `cls.__name__`, the same mechanism
`CVI.registry` already uses, and no new registration machinery is needed. The
lowercase short forms (`ridge`, `ard`) are a presentation convenience for a
future command-line entry point, not a serializable value; they belong in that
wrapper if and when one exists.

Internally an estimator holds the precision diagonal in log-density units, with
a zero at the bias index so `d` stays flat, and updates it in `estimate`. Both
the closed-form Gaussian path and the Poisson objective path consume the same
diagonal; the likelihood owns its own averaging, as it does now.

Likelihoods become observation models only — `initialize_info`, `update_pseudo`,
and an optional conjugate representation:

```python
    def loading_terms(y, valid_y, m, V) -> tuple[Array, Array, Array] | None
```

The estimator drives the solve: it uses the closed form when the likelihood
offers conjugate terms, and a generic gradient solve otherwise. `Gaussian`
implements `loading_terms` (`_gaussian_loading_moments`, already present);
`Poisson` does not, because `exp(eta)` breaks conjugacy in `C`.

Swapping estimators swaps the prior as well as the algorithm. That is intended
here — the algorithm is determined by whether `alpha` is fitted — but it should
be stated in the docs so nobody expects to keep the prior and change the solver.

## Migration

Moving `update_readout` out of `CVI` (and routing `initialize_params` through
the estimator) breaks four call sites, all of which must move in the same
change:

| Site | Today | Becomes |
|---|---|---|
| `examples/demo_vdp.py` `FrozenPoisson` | overrides `update_readout` to a no-op | `estimator="FrozenEstimator"` |
| `tests/test_benchmark.py` `FrozenPoisson` | same | same |
| `tests/test_cvhm.py` | monkeypatches both methods on `Poisson` | patch the estimator instead |
| `tests/test_masking.py` | calls `Gaussian.update_readout` directly | calls an estimator |

The frozen-readout pattern is the interesting one. It is not a likelihood
variant at all — it holds `C` and `d` fixed while the latents are still
inferred — so under the split it becomes an estimator whose `estimate` is the
identity, usable with either likelihood, and the two `FrozenPoisson` subclasses
disappear. That is a small instance of the general point: freezing the readout
and choosing its prior are both properties of the estimation step.

## Capability, not a stub

`RidgeEstimator` needs only a penalty diagonal, so it works with every
likelihood. `ARDEstimator` has `requires_conjugate_terms = True` because it
forms the evidence.

The unsupported pair `("Poisson", "ARDEstimator")` is rejected in
`CVHM.__post_init__`, which already resolves `CVI.registry[self.observation]`.
No placeholder class and no registered estimator that always raises: the gap is
carried by the interface and reported at the boundary, which is where input
validation belongs.

## Where `alpha` lives

Put it in `Params` as a fourth field. Reasons:

- it is fitted state for ARD and configuration for Ridge, and `Params` is
  already the container for readout state;
- it persists across M-steps through the existing `_FitState` carry with no new
  plumbing;
- it serialises with the params payload instead of needing a second channel;
- `readout_change` can compare it, which it should once `alpha` is fitted.

Cost: `serialization.py` must record and validate it, and `_params_manifest`
and friends grow a field. That is contained and preferable to threading a
second pytree through `fit` and the archive.

Store the precision itself, not its log: it is representable directly, reads
better in a manifest, and the prior parameters are static estimator
configuration rather than fitted state, so `alpha` is the only new quantity
that has to persist and serialise.

## Convergence metric

The gauge is set by the estimator's prior, not the likelihood. An isotropic
prior leaves the orthogonal rotation gauge, so `readout_change`'s Procrustes
alignment is correct. A per-column prior leaves only permutation and sign, so
the alignment would rotate away differences the prior treats as real and must
be dropped. Drive this from `Estimator.isotropic` rather than a call-site flag.

When `alpha` is fitted it is part of the converged quantity and should enter
the metric alongside `C`, `d`, and `R`.

## Status

Implemented and green on `feature/loading-prior`: `Estimator`, `RidgeEstimator`,
`ARDEstimator`, `FrozenEstimator`; `CVI.loading_terms` / `readout_objective`
replacing `update_readout`; the config key, boundary check, serialization, and
gauge-aware `readout_change`; all call sites migrated. 135 tests pass, the
benchmark R-squared is unchanged, and the estimated-readout demo figure is
identical to before the refactor, which is the ridge identity check in
end-to-end form.

Findings that changed the plan while implementing:

- ARD had to become a closed-form fixed point, because the standalone function's
  scipy driver cannot run inside `lax.while_loop`. Verified that the fixed point
  is the stationary point of this evidence (gradient norm 6e-14) and that
  L-BFGS cannot improve on it.
- The noise prior was removed. Over a 32-configuration grid, a loading prior
  alone matches having both on every count (32/32 finite, 32/32 correct rank,
  same worst-case precision), while dropping the loading prior collapses to
  13/32 finite and 7/32 correct, with precisions reaching `1e54`. So the loading
  prior is load-bearing and the noise prior is redundant; the observation
  variance is now estimated from expected residuals with no prior.
- Two earlier claims here were wrong and were corrected. The noise prior was
  first credited with fixing pruned precisions that had been out by seven orders
  of magnitude -- the real cause was replacing the scipy driver with the fixed
  point, the optimiser having under-converged along a very flat direction
  (`log p = -283.3` against `-280.3`). It was then described as inert, which was
  true on the single problem tested and false in general.
- `Params.alpha` turned out to be necessary rather than merely tidy: ranking
  the precisions is the deliverable, so they have to be reachable after a fit.
- The Poisson latent recovery that had been reported as a separate pre-existing
  issue was a bug in the Poisson readout objective (the variance correction had
  leaked into the count term, rewarding growth of `C` along the latent-scale
  gauge), together with an under-converged inner CVI loop at the old `lr`. Both
  are fixed -- the objective in 0.4.0, `lr` on this branch -- and the Poisson
  joint-sequence test asserts latent recovery again.
- ARD was only exercised on fixed latents until it was run inside the full EM
  loop. There it recovers the true dimensionality: with five fitted latents it
  leaves exactly `K_true` finite precisions and drives the surplus loadings to
  numerical zero (`1e-130`), where ridge only shrinks them to about `0.1`. The
  pruned precision plateaus near `1e5`, which is set by the guard prior's
  constants rather than by the data, so only the ranking is meaningful. A
  three-latent problem had not converged by 300 iterations but still ranked
  correctly.

## Steps

1. **Reconcile the convention.** Drop the bias penalty from the Gaussian ridge;
   the loading penalty already has the right scale and `lam` already is the
   precision. Behaviour change, so gate on the benchmark. Done: benchmark and
   full suite pass, and the reference objective in `test_masking.py` was
   updated to the flat-bias convention.
2. **Extract `Estimator` and `RidgeEstimator`**, reproducing step 1 exactly. No
   ARD yet — this step must not change any number. This is where the M-step
   moves out of `CVI`; likelihoods keep the E-step, the data term, and the
   data-driven start (`initialize_params`, now called through the estimator),
   and `initialize_info`/`update_pseudo` are untouched.
3. **Expose `Gaussian.loading_terms`** by moving `_gaussian_loading_moments`
   from module level onto the readout so the estimator can request it.
4. **Add `ARDEstimator`** wrapping the existing `gaussian_ard_estimate` body.
   Nothing calls that function today, so this is a move rather than a
   replacement: the evidence, the prior, and the jointly fitted variances
   are already written and verified.

   **Blocker found while implementing.** The standalone function fits the
   precisions with `scipy.optimize.minimize` on host values. That cannot run
   inside the EM loop, because `CVHM.fit` traces the loop with
   `jax.lax.while_loop` and the starting point depends on `params.R`, which is a
   tracer there. So step 4 is not a move: the precision fit has to be expressed
   in JAX. The natural choice is the MacKay fixed point already used by the
   reference implementation,

   ```
   gamma_k = N - alpha_k * sum_n R_nn * Sigma_n[k, k]
   alpha_k = gamma_k / ||C[:, k]||^2
   ```

   with `Sigma = (Prec / R + diag(0, alpha))^-1` taken from the conjugate terms,
   run for a fixed number of iterations in a `lax.fori_loop`. Here that inverse
   is exact, unlike the Laplace route that diverged earlier, so `gamma` is
   well conditioned. Two further consequences: timescales would then need to
   persist in `Params` (see "Where `alpha` lives") or be re-fitted from the
   estimator default each M-step, and `scipy` drops out of the estimator
   entirely.

   Work in progress is parked on `wip/estimator-refactor`, which has the
   estimator axis, `LoadingTerms`, and the likelihood split working for ridge
   and frozen estimators with the boundary check in place.
5. **Thread the estimator through `CVHM`** — config key, `_FitState`, boundary
   check, serialization, `readout_change`.
6. **Migrate the call sites** listed above, including `FrozenEstimator`, in the
   same change as step 2. Nothing that overrides `update_readout` may be left
   behind.
7. **Docs and changelog.**

## Validation

- Benchmark `R² >= 0.96` after step 1 and again after step 5; the ridge path
  must be numerically unchanged from the reconciled convention onward.
- Step 2 must reproduce step 1's fitted `C`, `d`, `R` to tight tolerance — a
  pure refactor check.
- New tests: ARD rank recovery on the existing sweep; `("Poisson", "ARD")`
  rejected at construction; `Params` round-trip carrying `alpha`;
  `readout_change` skipping the alignment when `isotropic` is false.
- Re-run the rank and kernel-amplitude sweeps through the composed API and
  confirm the earlier separation survives.
- **Reference-implementation check.** With `V = 0` the Jensen bound is tight, so
  `ARDEstimator` must reproduce a standard ARD regression on the same design:
  signal precisions to ~1e-3 and coefficients to ~1e-2 at `N = 1`, with `R`
  differing only by the single degree of freedom the bias costs. This is the
  strongest guard against convention drift and belongs in the suite, not as a
  one-off. Already written against the standalone function
  (`test_gaussian_ard_matches_reference_ard_regression`); sklearn is a runtime
  dependency, so no optional-import guard is needed.
- **Threshold robustness.** Assert that the retained rank is unchanged across
  thresholds spanning decades rather than fixing a threshold, which would be
  circular. Already written (`test_gaussian_ard_rank_is_threshold_robust`).
- **Estimator identity check.** `RidgeEstimator` with equal precisions must
  reproduce `ridge_estimate`'s closed form, including the `R_nn` unit
  conversion, so the refactor is verifiably behaviour-preserving.

## Out of scope

- **Poisson + ARD.** Needs a Laplace or variational treatment over `C` in the
  conditional Poisson GLM. Rejected at the boundary, documented, no stub. It is
  materially easier than the earlier attempt because the latents are already
  conditioned on, so only `C` needs approximating.
- **ARD on the latent prior** (per-latent kernel hyperparameters). Separate
  axis, and it would require estimating kernel hyperparameters, which is a
  different problem.
- **Structured or hierarchical loadings.** The `precision` interface is
  deliberately a diagonal; widening it is a later decision.

## Settled

- **The bias is unpenalised**: a flat prior on `d`, penalty on `C` only.
- **A Gamma prior on the loading precisions instead of a floor.** `alpha` takes
  a Gamma prior (`alpha_shape`, `alpha_rate`), which bounds it at both ends
  rather than by an ad-hoc floor and is the standard formulation. It is a guard
  on the updater, not a modelling choice: without it the fixed point diverges
  or reaches precisions near `1e17` on most of a 32-problem grid. The defaults
  are the reference implementation's weak `Gamma(1 + 1e-6, 1e-6)`, enabled by
  default, and the fit is insensitive to them over many orders of magnitude.
- **Observation variances are fitted by the evidence**, jointly with `alpha`,
  not frozen. An earlier revision of this plan froze them because the evidence
  appeared to diverge as a variance vanished; that was an error in the
  evidence's normalisation, since fixed, and the maximum in `R` is interior and
  clean (`log R* = -2` at `V = 0`, `-1` at `V = 0.3`). The variance itself takes
  no prior: the residual rule has no `0 / 0` corner.
- **No hard threshold.** Report `alpha` and document that a large value marks a
  shrunk column; leave any cut to the caller.
- **The noise rule follows what the estimator fits.** `RidgeEstimator` keeps the
  EM/expected-residual update, which is correct when the penalty is fixed.
  `ARDEstimator` fits the variances against the evidence, jointly with `alpha`.
  Pairing an EM residual with an evidence-based precision would put the two on
  different criteria, and the joint fixed point would maximise neither, so
  "which residual rule" is a property of the estimator rather than a detail.
- **Precisions are shared across features**, making this a hierarchical ARD
  rather than one ARD regression per feature. Deliberate: relevance is a
  property of the latent dimension, not of each feature. It is the main
  structural difference from the reference implementation, which fits one
  output at a time and so carries per-feature precisions.
- **The bias costs one residual degree of freedom.** The fitted variance uses
  `n - sum(gamma) - 1`, the classical account for a model with an intercept; an
  implementation that centers the design away uses `n - sum(gamma)`. The two
  agree as `n` grows.

- Second axis is an estimator, not a prior: classes `RidgeEstimator` and
  `ARDEstimator`.
- Class names are CamelCase throughout, acronyms included (`ARDEstimator`).
- Serializable config values are exact class names, so `estimator` reads like
  `observation` and both key on `cls.__name__`.
- Lowercase short aliases (`ridge`, `ard`) are reserved for a command-line
  wrapper. No such entry point exists in this repository, so nothing is built
  for them now.
- **Freezing the readout is an estimator**, not a likelihood subclass, so the
  two `FrozenPoisson` subclasses in the example and benchmark are retired by
  the split.

## Open questions

- The config key itself is `estimator`, which does not say what is estimated;
  `loading` would be more precise. Worth one line of docstring either way.
- Whether `RidgeEstimator` accepts a scalar or a per-column vector. A vector
  costs nothing and is the natural degenerate case of ARD.
- Whether `alpha` should enter `readout_change` scaled or in log space; raw
  precisions span many orders of magnitude.
- Whether `CVI` keeps a `data_term` method, or the estimator calls
  `loading_terms` for the conjugate case and a generic objective otherwise.
  Resolved in the parked work: the likelihood keeps `readout_objective`, which
  takes the penalty diagonal so that it can own its own averaging.
- Whether fitted precisions persist in `Params` or are re-fitted from the
  estimator default each M-step. Re-fitting is stateless and reproduces the
  validated standalone behaviour; persisting is a warm start whose benefit is
  unmeasured.
