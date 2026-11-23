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
disagree. Making the prior a first-class object gives one convention, makes
ridge the fixed equal-precision special case of ARD, and puts the ARD/ridge
choice in the user's hands instead of in the readout's implementation.

This depends on the conditional ARD estimator already added on
`feature/ard-loading` (`gaussian_ard_estimate`, `gaussian_ard_evidence`).

## Why: the current disagreement

The same object — a squared L2 prior on the loadings — is implemented twice:

| | Gaussian (`utils.ridge_estimate`) | Poisson (`poisson_readout_objective`) |
|---|---|---|
| Bias `d` | penalised | not penalised |
| Scale | `lam * R_nn` (noise-relative) | absolute |
| Form | separate closed-form solver | inside the LBFGS objective |
| Hyperparameter | `lam = 0.1` | `gamma = 1.5` |

Three disagreements on one concept. `lam = 0.1` is inert in practice (~0.05% of
the objective), so the Gaussian bias penalty is small, but it is a modelling
claim nobody chose.

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

    def initialize_params(self, likelihood, y, valid_y, n_factors, *, random_state)
    def estimate(self, likelihood, params, y, valid_y, m, V) -> tuple[Params, float]
```

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

Store the precision itself, not its log: the floored range is `[e^-12, inf)`,
which is representable directly and reads better in a manifest.

## Convergence metric

The gauge is set by the estimator's prior, not the likelihood. An isotropic
prior leaves the orthogonal rotation gauge, so `readout_change`'s Procrustes
alignment is correct. A per-column prior leaves only permutation and sign, so
the alignment would rotate away differences the prior treats as real and must
be dropped. Drive this from `Estimator.isotropic` rather than a call-site flag.

When `alpha` is fitted it is part of the converged quantity and should enter
the metric alongside `C`, `d`, and `R`.

## Steps

1. **Reconcile the convention.** Gaussian regularizer to `R_nn * diag(0, alpha)`
   with the bias unpenalised; Poisson to per-column `alpha`. Behaviour change,
   so gate on the benchmark.
2. **Extract `Estimator` and `RidgeEstimator`**, reproducing step 1 exactly. No
   ARD yet — this step must not change any number. This is where the M-step and
   `initialize_params` move out of `CVI`; likelihoods keep only the E-step and
   the data term, and `initialize_info`/`update_pseudo` are untouched.
3. **Expose `Gaussian.loading_terms`** by moving `_gaussian_loading_moments`
   onto the readout so the estimator can request it.
4. **Add `ARDEstimator`** and wire the evidence update into its `estimate`,
   replacing the standalone `gaussian_ard_estimate` call path.
5. **Thread the estimator through `CVHM`** — config key, `_FitState`, boundary
   check, serialization, `readout_change`.
6. **Docs and changelog.**

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

## Open questions

- Config key: `estimator="RidgeEstimator"` is vague about what is estimated;
  `loading="RidgeEstimator"` trades precision for brevity. Class names
  `RidgeEstimator` and `ARDEstimator` are settled.
- Whether an `Estimator.registry` mirrors `CVI.registry`, or `CVHM` resolves the
  estimator some other way.
- Whether `RidgeEstimator` accepts a scalar or a per-column vector. A vector
  costs nothing and is the natural degenerate case of ARD.
- Whether `alpha` should enter `readout_change` scaled or in log space; raw
  precisions span many orders of magnitude.
- Whether `CVI` keeps a `data_term` method, or the estimator calls
  `loading_terms` for the conjugate case and a generic objective otherwise.
