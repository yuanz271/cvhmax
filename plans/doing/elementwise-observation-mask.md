# Element-wise Observation Mask

## Goal

Extend CVHMAX to fit and infer from partially observed neuron populations.
Sessions are stacked on the trial axis; there is no new session axis:

```text
y:       (trials, time, neurons)
valid_y: (trials, time, neurons)
```

`valid_y[b, t, n]` is true exactly when neuron `n` is observed in trial `b`
at time `t`. A neuron absent from an entire session has a false mask for all
its time bins. A padded time bin has a false mask for every neuron. Inputs use
a common neuron ordering across trials, with matched neurons mapped to the
same index.

The mask represents availability, not a zero-valued observation. Inference,
initialization, and readout updates must depend only on entries marked observed.
Changing a missing entry's finite placeholder value must not change the result.

## Scope and model assumptions

- Each trial has an independent GP trajectory. Stacking sessions as trials
  does not align their time points or force their trajectories to match.
- Retain the current single shared readout `C, d, R` across all trials. In
  particular, this assumes a matched neuron's loading is shared across its
  sessions.
- Session-specific loading drift is **not** implemented by this mask alone.
  If a session contains multiple trials that should share a drifting readout,
  a later model must add session-specific readouts and a trial-to-session map.
- The current model uses one `dt` and one set of fixed GP kernels across
  trials. Resample recordings to a common time grid or extend the dynamics
  separately.
- A common latent coordinate system is only identifiable under the shared
  readout/prior assumptions and sufficient cross-session anchors. Feature
  masking alone does not identify a shared space when sessions are unlinked.

## Current limitation

`CVHM.fit()` currently creates a `(trials, time)` mask. `CVI.initialize_info`
passes one scalar mask per bin to `bin_info_repr` or
`poisson_cvi_bin_stats_latent`. Each helper combines the whole observation
vector before applying the mask. Gaussian `ridge_estimate`, Poisson readout
updates, and Factor Analysis initialization likewise assume whole-bin masks.
Passing a feature-level mask to this code is not a valid partial-observation
likelihood.

## Public data contract

1. Make `valid_y` match `y` exactly: `(trials, time, neurons)`, or
   `(time, neurons)` for a single-trial `(time, neurons)` input. `None` means
   every supplied entry is observed.
2. Use boolean masks internally. Masked values may be filled with any finite
   placeholder; sanitize them with `where` **before** likelihood calculations,
   rather than multiplying missing values by zero after arithmetic.
3. Update `pad_trials` to return a mask with the same shape as its padded `y`.
   Per-trial input masks have shape `(time_i, neurons)`. An unpadded bin defaults
   to all neurons observed; every padded entry is masked.
4. Update local callers, tests, and examples to use the new shape. Do not add a
   separate session dimension or silently interpret missing-neuron zeros as
   observations.

## Masked inference

### Gaussian readout

For one bin, let $\mathcal O$ be the observed neuron indices. Compute latent
information using the observed rows of `C`, `d`, and the observed principal
submatrix of `R`:

$$
J=C_{\mathcal O}^\top R_{\mathcal O\mathcal O}^{-1}C_{\mathcal O},
\qquad
j=C_{\mathcal O}^\top R_{\mathcal O\mathcal O}^{-1}
(y_{\mathcal O}-d_{\mathcal O}).
$$

Keep JAX shapes static. For the diagonal mask matrix $D$, define

$$
\widetilde C=DC,\qquad
\widetilde R=DRD+(I-D),\qquad
\widetilde y=D(y-d).
$$

Solves with $\widetilde R$ give the same $j$ and $J$ as the observed-submatrix
formulation, including a zero information increment when all neurons are
missing. Construct $\widetilde y$ with a mask-aware `where`. CVHM continues to
lift latent information using its existing state mask `M`; the state-space
observation matrix is still `H = C @ M` on observed rows.

The current Gaussian model has a **full** observation covariance `R`. Do not
replace $R_{\mathcal O\mathcal O}^{-1}$ with an observed block of $R^{-1}$:
those are generally different. If masks are constant within a session, reuse
per-pattern calculations where practical; measure performance before adding
caching machinery.

### Poisson readout

Apply the neuron mask to individual log-likelihood, gradient, and curvature
contributions **before** summing over neurons. A fully masked bin contributes
zero information and zero likelihood. Preserve the `Params.R` JAX scalar zero
sentinel.

For the readout objective, aggregate observed-neuron terms per valid time bin;
retain the existing normalization convention for all-observed data. Ensure
that its gradient and the CVI statistics use a consistent stated objective.
The current `poisson_trial_nell` multiplies the observed count by its
variance-adjusted `eta`; review that expression against the expected Poisson
log-likelihood before claiming objective consistency. Any correction that
changes the all-observed objective must be documented and tested rather than
hidden inside the masking change.

## Initialization and readout fitting

### Initialization

The current `sklearn.decomposition.FactorAnalysis` initialization expects
complete observations. Initialize from training data only:

1. Compute each neuron's mean from its observed entries.
2. For **initialization only**, replace missing entries with that neuron's
   observed mean and omit fully masked bins from Factor Analysis.
3. Use the resulting loading and bias as a starting point. The actual E-step,
   M-step, and objective must always use the original mask, never the filled-in
   entries as observations.
4. Continue allowing callers to provide initial `Params` directly.

This is a deterministic initialization heuristic, not missing-data Factor
Analysis. Evaluate sensitivity to initialization on synthetic sessions. A
neuron never observed anywhere in training cannot have its readout parameters
estimated from that dataset.

### Poisson fitting

Mask each neuron's terms in the readout objective and its derivatives. Fit each
neuron's loading and bias from the bins in which that neuron is observed. Keep
the existing regularization and finite-rate policy explicit, and test the
all-observed case against the existing implementation.

### Gaussian fitting

The existing `ridge_estimate()` fits one complete observation vector per valid
bin, ignores posterior covariance `V`, and estimates a full residual covariance
from those complete residuals. It cannot simply be reused with per-neuron
masks.

Keep the full-covariance model unless a separate modeling decision changes it.
Design a masked Gaussian readout update from the observed-data expected
objective, using posterior means and covariances. For full `R`, this requires
accounting for each observed covariance submatrix and for missing-observation
uncertainty; zero-filling residuals or pairwise covariance estimates are not
sufficient in general and may yield an invalid covariance. A missing-data EM
update that uses conditional moments of the missing observations is one
candidate. State the objective and regularization explicitly and verify the
update against a small dense calculation before integrating it.

**Decision gate:** the full-covariance masked M-step must be specified and
validated before claiming Gaussian training support. The existing all-observed
ridge update is an approximation that omits `V`; a mathematically correct
masked update may therefore change all-observed training behavior. Measure and
document that change rather than claiming exact training parity. If two neurons
are never co-observed, their noise cross-covariance is not identified by those
data; a diagonal or session-specific noise structure would be a separate
explicit modeling decision, not a silent fallback.

## Implementation sequence

1. Update the `y`/`valid_y` contract at `CVHM.fit`, `CVHM.infer`, the `CVI`
   interface, and `pad_trials`; migrate local examples and tests.
2. Implement masked Gaussian information increments with the fixed-shape full
   covariance formulation above. Keep the `lift`/`M` bridge unchanged.
3. Implement masked Poisson CVI increments and per-neuron objective contributions.
4. Implement the documented mask-aware initialization heuristic.
5. Implement and validate masked Poisson readout fitting.
6. Resolve the full-covariance Gaussian M-step decision gate, then implement and
   validate its readout fitting update. Do not silently switch to diagonal `R`.
7. Update `README.md`, `docs/api.md`, `docs/data-model.md`,
   `docs/algorithms.md`, and affected docstrings. Document the shared-readout
   assumption and the separate need for session-specific loading drift.

## Mathematical and integration tests

- **All observed:** Gaussian information increments and Poisson CVI increments
  reproduce the current all-observed calculations. Test the existing Poisson
  objective's all-observed behavior unless its discrepancy above is resolved
  explicitly.
- **Gaussian observed subset:** with a non-diagonal `R`, compare `j` and `J`
  against a dense computation using the explicitly selected observed rows and
  `R` submatrix. Verify the state-space result using `H = C @ M`.
- **Poisson observed subset:** compare pseudo-observation information and the
  readout objective/gradient with a direct sum over observed neurons.
- **Time padding:** a fully masked bin contributes no observation information;
  `pad_trials` returns the correct element-wise mask for variable trial lengths.
- **Missing-value invariance:** changing finite placeholder values for masked
  neurons leaves inference and readout updates unchanged.
- **Gaussian M-step:** compare the selected masked update against a small dense
  reference objective and validate finite symmetric positive-definite `R`.
- **Joint sessions:** on synthetic sessions with partial neuron overlap and a
  shared readout, recover finite latent posteriors and useful held-out
  predictions. Evaluate latent recovery only up to the model's applicable
  gauge; do not require the sessions' independent trajectories to match.
- **Numerical stability:** posterior moments remain finite and covariance
  matrices symmetric positive-semidefinite within tolerance under the intended
  masks and fixed kernel scales.

Keep held-out neurons/time bins out of initialization and parameter fitting. Use
fixed seeds and compare at the trial/session level, not individual entries as
independent subjects.

## Validation

Run focused readout, filtering, and padding tests; then the full x64 CPU suite,
the slow VdP benchmark, and the existing CPU–GPU parity test. The all-observed
VdP benchmark must continue to meet its existing $R^2\ge 0.96$ floor. Report
Gaussian all-observed behavior changes if its M-step is corrected as part of
this work, and record runtime/memory for representative session-constant masks.

```bash
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu uv run --no-sync pytest tests/test_cvi.py tests/test_filtering.py tests/test_utils.py -q
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu uv run --no-sync pytest -q
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu uv run --no-sync pytest tests/test_benchmark.py --run-slow -q
JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda,cpu uv run --no-sync pytest tests/test_gpu.py --run-gpu -q
uv run --no-sync ruff check src/cvhmax tests/test_cvi.py tests/test_filtering.py
git diff --check
```

## Done when

`CVHM.fit()` and `infer()` accept an element-wise `valid_y` with the same shape
as `y`; missing neurons affect neither inference nor fitted readouts; Gaussian
and Poisson paths meet the mathematical tests above; all-observed behavior is
accounted for; documentation states the shared-readout and session-identifiability
limits; and the validation commands pass.
