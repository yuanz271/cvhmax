# Element-wise Observation Mask

## Goal

Extend CVHMAX to fit and infer from partially observed neuron populations.
Sessions are stacked on the trial axis; there is no new session axis:

```text
y:       (trials, time, neurons)
valid_y: broadcastable to y, or None
```

For example, `valid_y` may have shape `(trials, time, neurons)`,
`(trials, time, 1)` to share each time-bin mask across neurons, or
`(trials, 1, neurons)` to share each trial-neuron availability across time.
The shared axis has size 1 in an actual JAX array; a zero-sized axis is not a
broadcast wildcard. `None` means all supplied entries are observed.

After broadcasting, `valid_y[b, t, n]` is true exactly when neuron `n` is
observed in trial `b` at time `t`. A neuron absent from an entire session has a
false mask for all its time bins. A padded time bin has a false mask for every
neuron. Inputs use a common neuron ordering across trials, with matched neurons
mapped to the same index.

The mask represents availability, not a zero-valued observation. Inference,
initialization, and readout updates must depend only on entries marked observed.
Changing a missing entry's finite placeholder value must not change the result.

## Scope and model assumptions

- Each trial has an independent GP trajectory. Stacking sessions as trials
  does not align their time points or force their trajectories to match.
- Assume no loading drift: a matched neuron has the same loading row `C`
  across sessions. Retain one shared readout `C, d, R` across all trials.
  Shared `d` and `R` are additional model assumptions; no loading drift alone
  does not imply that they are shared.
- Constrain Gaussian observation-noise covariance `R` to be diagonal for all
  fits, including fully observed data. The latent processes and loadings
  explain cross-neuron covariability. This is a change from the current
  full-covariance Gaussian model.
- Session-specific readout parameters are outside this plan. If needed later,
  they require a trial-to-session map when a session has multiple trials.
- The current model uses one `dt` and one set of fixed GP kernels across
  trials. Resample recordings to a common time grid or extend the dynamics
  separately.
- Require every neuron to be observed in at least one time bin across all
  training trials. Reject a fit with an observation dimension that is never
  present, rather than trying to estimate its readout parameters.
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

1. Accept `valid_y=None` or a mask broadcastable to the supplied `y` using
   ordinary JAX broadcasting. For batched `y`, explicitly support the full
   shape, `(trials, time, 1)`, and `(trials, 1, neurons)`. A two-dimensional
   `(trials, time)` mask must use an explicit trailing singleton axis to mean
   a time-bin mask; do not infer a missing axis from ambiguous dimensions.
2. At the `fit()`/`infer()` boundary, convert the mask to boolean and broadcast
   it to `y.shape` once. For a single-trial `(time, neurons)` input, broadcast
   to that shape before adding the trial axis. `None` creates an all-true mask.
   Core routines receive the canonical full-shape mask.
3. Masked values may be filled with any finite placeholder; sanitize them with
   `where` **before** likelihood calculations, rather than multiplying missing
   values by zero after arithmetic. After broadcasting, `CVHM.fit()` must raise
   an error before initialization if any neuron is masked at every time bin
   across all trials. `CVHM.infer()` may use any observed subset because the
   readout has already been fitted.
4. Update `pad_trials` to return a mask with the same shape as its padded `y`.
   Per-trial input masks must broadcast to `(time_i, neurons)`; a
   `valid_y_list` of `None` means all supplied entries are observed. Every
   padded entry is masked.
5. Update local callers, tests, and examples for the broadcastable-mask
   contract. Do not add a separate session dimension or silently interpret
   missing-neuron zeros as observations.

## Masked inference

### Gaussian readout

The Gaussian residual covariance is $R=\operatorname{diag}(r)$, where
$r_n>0$ is neuron $n$'s observation-noise variance. This is the covariance of
the residual $\varepsilon$ conditional on the latent, not the total observation
covariance.

For mask value $w_n\in\{0,1\}$ and loading row $c_n^\top$, compute the
latent information by summing observed neurons:

$$
J=\sum_n \frac{w_n}{r_n}c_n c_n^\top,
\qquad
j=\sum_n \frac{w_n}{r_n}c_n(y_n-d_n).
$$

Use fixed-shape JAX array operations and sanitize missing values with `where`
before arithmetic. A fully masked bin contributes zero information. CVHM
continues to lift latent information using its existing state mask `M`; the
state-space observation matrix is still `H = C @ M` on observed rows.

Keep `Params.R` as an `(N, N)` diagonal matrix for Gaussian readouts; use its
diagonal variance vector for the calculations above. Poisson retains its
scalar zero sentinel. Validate caller-supplied Gaussian `R` at the public API
boundary as a positive diagonal covariance rather than silently discarding
its off-diagonal entries.

### Poisson readout

Apply the neuron mask to individual log-likelihood, gradient, and curvature
contributions **before** summing over neurons. A fully masked bin contributes
zero information and zero likelihood. Preserve the `Params.R` JAX scalar zero
sentinel.

Preserve the current `poisson_trial_nell` objective, including its
variance-adjusted `eta` in the count term and its existing rate clipping. The
CVI update and this objective are not the same expected-log-likelihood
formulation; do not claim objective consistency or change either formulation as
part of masking.

The current objective divides the summed per-neuron terms and loading penalty
by the number of valid time bins. After broadcasting to the full shape, count a
time bin as valid when at least one neuron is observed:

```python
n_valid_bins = valid_y.any(axis=-1).sum()
```

Sum only observed neuron terms before dividing. This preserves the existing
all-observed scaling, but it does not give sessions with different numbers of
neurons equal weight.

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
Analysis. Evaluate sensitivity to initialization on synthetic sessions. The
input contract assumes every neuron is observed in at least one training
session.

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

Use the diagonal-noise expected objective and update each neuron using only
its observed bins. For posterior mean $m_{bt}$ and covariance $V_{bt}$, let

$$
\widetilde m_{bt}=\begin{pmatrix}1\\m_{bt}\end{pmatrix},\qquad
\widetilde V_{bt}=\begin{pmatrix}0&0\\0&V_{bt}\end{pmatrix},\qquad
\beta_n=\begin{pmatrix}d_n\\c_n\end{pmatrix}.
$$

With $w_{btn}$ the mask and $N_n=\sum_{bt}w_{btn}>0$, form the sufficient
statistics

$$
A_n=\sum_{bt}w_{btn}
(\widetilde m_{bt}\widetilde m_{bt}^\top+\widetilde V_{bt}),\qquad
b_n=\sum_{bt}w_{btn}y_{btn}\widetilde m_{bt}.
$$

For the existing ridge strength $\lambda=0.1$, use the conditional objective

$$
\mathcal Q_n(\beta_n,r_n)=\frac{N_n}{2}\log r_n+
\frac{1}{2r_n}\sum_{bt}w_{btn}
\left[(y_{btn}-\beta_n^\top\widetilde m_{bt})^2
+c_n^\top V_{bt}c_n\right]
+\frac{\lambda}{2}\lVert\beta_n\rVert_2^2.
$$

The ridge penalty includes the bias, as in the existing `ridge_estimate()`.
Holding the current $r_n$ fixed, solve for the updated loading and bias:

$$
(A_n+\lambda r_n I)\beta_n^{+}=b_n.
$$

Then update the noise variance conditionally using the new $d_n,c_n$:

$$
r_n^{+}=\frac{1}{N_n}\sum_{bt}w_{btn}
\left[(y_{btn}-d_n^{+}-(c_n^{+})^\top m_{bt})^2
+(c_n^{+})^\top V_{bt}c_n^{+}\right].
$$

Form the residual expression directly rather than subtracting large sufficient
statistics. Solve the systems rather than forming inverses. This is a
conditional loading update followed by a conditional noise update, not a claim
of joint closed-form optimization over both.

The existing all-observed ridge update ignores `V` and estimates full `R`;
the new objective and diagonal-noise model intentionally change that
behavior. Document and measure the all-observed change rather than claiming
training parity. At least one observation per neuron makes each $N_n$ positive,
but does not guarantee precise loading estimation or a strictly positive noise
estimate. If the expected residual variance degenerates to zero or becomes
non-finite, report the numerical failure; do not silently introduce a noise
floor that changes the objective.

## Posterior consistency after fitting

After the final readout update, `CVHM.fit()` must perform one final inference
pass using the final parameters, training data, and training mask. This pass
must not update the readout again. It populates both representations of the
same posterior:

- `model.latent = (z, Z)`: SDE state posterior in information form;
- `model.posterior = (m, V)`: GP latent posterior in moment form.

For the data and mask passed to `fit()`, require these invariants up to
numerical tolerance:

```python
import numpy as np
from cvhmax import project

projected = project(*model.latent, model.latent_mask())
inferred = model.infer(y, valid_y=training_mask)
for cached, from_state, fresh in zip(model.posterior, projected, inferred):
    np.testing.assert_allclose(cached, from_state, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(cached, fresh, rtol=1e-6, atol=1e-8)
```

Implement one shared inference helper for the filtering, CVI, and projection
path. The helper receives the readout parameters explicitly and returns both
posterior representations. The EM E-step, the final post-training pass, and
`infer()` use that helper, with fixed dynamics reused where practical.
`infer()` remains non-mutating and returns only `(m, V)`. Keep initialized and
updated parameters local during training; publish the fitted parameters and
cached outputs only after final inference succeeds, so a failure does not leave
a partially updated fit.

## Implementation sequence

1. Update the `y`/`valid_y` contract at `CVHM.fit`, `CVHM.infer`, the `CVI`
   interface, and `pad_trials`; migrate local examples and tests.
2. Constrain Gaussian `Params.R` to a positive diagonal matrix and implement
   masked Gaussian information increments as weighted sums over observed
   neurons. Keep the `lift`/`M` bridge unchanged.
3. Implement masked Poisson CVI increments and per-neuron objective contributions.
4. Implement the documented mask-aware initialization heuristic.
5. Implement and validate masked Poisson readout fitting.
6. Implement the diagonal-Gaussian conditional loading and noise updates
   above. Include posterior covariance in the sufficient statistics, and
   document the intentional all-observed behavior change.
7. Use one shared inference helper for the EM E-step, `infer()`, and a final
   post-training pass. Store consistent `model.latent` and `model.posterior`
   under the final fitted parameters.
8. Update `README.md`, `docs/api.md`, `docs/data-model.md`,
   `docs/algorithms.md`, and affected docstrings. Document posterior
   consistency, the shared-readout assumption, and the separate need for
   session-specific loading drift.

## Mathematical and integration tests

- **All observed:** with diagonal `R`, Gaussian information increments
  reproduce the current calculation. Poisson CVI increments and objective
  reproduce the current all-observed calculations, including the documented
  objective discrepancy.
- **Gaussian observed subset:** compare `j` and `J` against a dense
  computation using explicitly selected observed rows and diagonal `R`.
  Verify the state-space result using `H = C @ M`.
- **Poisson observed subset:** compare pseudo-observation information and the
  readout objective/gradient with a direct sum over observed neurons.
- **Broadcast masks:** for each of `(trials, time, 1)` and
  `(trials, 1, neurons)`, compare inference and readout results with that mask
  explicitly broadcast to the full shape. `None` matches an all-observed mask.
  Check the Poisson valid-bin count after broadcasting a time-shared neuron mask.
- **Time padding:** a fully masked bin contributes no observation information;
  `pad_trials` returns the correct element-wise mask for variable trial lengths
  and accepts per-trial broadcastable masks.
- **Missing-value invariance:** changing finite placeholder values for masked
  neurons leaves inference and readout updates unchanged.
- **Gaussian M-step:** compare the masked conditional updates against a small
  dense reference for the objective above. Verify that the returned `R` is
  diagonal with finite positive variances.
- **Posterior consistency:** for Gaussian and Poisson fits with a partial mask,
  compare `model.posterior` with `model.latent` projected using
  `model.latent_mask()`, and with `model.infer(y, valid_y)`. Check that the
  final inference does not update the fitted readout parameters.
- **Joint sessions:** run the reproducible shared-latent synthetic experiment
  below. Evaluate latent recovery only up to the model's applicable gauge.
- **Observation availability:** fitting fails at the public boundary when
  any neuron is missing at every time bin in every trial; inference may use a
  subset of the neurons from an already fitted model.
- **Numerical stability:** posterior moments remain finite and covariance
  matrices symmetric positive-semidefinite within tolerance under the intended
  masks and fixed kernel scales.

Keep held-out neurons/time bins out of initialization and parameter fitting. Use
fixed seeds and compare at the trial/session level, not individual entries as
independent subjects.

### Shared-latent synthetic sessions

Construct two trials as sessions with the **same realized latent trajectory**
but different observed neuron subsets:

- Using seed `42`, generate two independent latent coordinates from the same
  fixed Matérn-3/2 GP prior (`sigma=1`, `rho=6`, `dt=1`) for 256 time bins.
  Reuse the resulting trajectory in both sessions.
- Use 12 neurons with one shared loading `C`. For neuron $n=0,\ldots,11$,
  set its loading row to $0.7(\cos(2\pi n/12),\sin(2\pi n/12))$.
  Session 1 records neurons 0–7; session 2 records neurons 4–11. Thus four
  matched neurons anchor the shared space, and every neuron is recorded in at
  least one session.
- Generate Gaussian observations using `d=0`, noise covariance `R=0.05 I`, and
  independent noise draws between sessions. Generate Poisson observations with
  `d=log(8)` and independent count draws between sessions.
- Hold out recorded entry `(s, t, n)` when `(s + t + n) % 5 == 0`, where
  session indices start at zero. This retains more than `K` observed neurons
  per time bin. Derive preprocessing, initialization, and fitted parameters
  only from the remaining observed entries.
- Fit with the fixed GP prior, `random_state=42`, `max_iter=100`, `tol=1e-4`,
  `min_iter=2`, `convergence_patience=2`, `cvi_iter=5`, and `lr=0.1`. Report
  the iteration count and convergence status.
- After fitting, use `model.posterior` with the final `model.params` for all
  held-out predictions and latent-recovery scores. Verify it matches
  `model.infer(y, valid_y=training_mask)` and the projection of `model.latent`.
  Keep held-out entries masked during both fitting and inference.
- For Gaussian, predict each held-out neuron by $d_n+c_n^\top m_{st}$. For
  Poisson, use the posterior expected rate
  $\exp(d_n+c_n^\top m_{st}+\tfrac12 c_n^\top V_{st}c_n)$ with the existing
  clipping policy. For each session, score held-out predictions against that
  session's per-neuron training means. Require held-out $R^2 > 0$ for each
  session, alongside finite posterior moments.
- For latent recovery, fit **one common** orthogonal alignment from the pooled
  posterior means of both sessions to the known trajectory. Apply that same
  alignment to both sessions and require each session's latent $R^2$ against a
  zero-mean baseline to exceed zero. Do not fit a different rotation per
  session: that would not test for a shared coordinate system.

The identical trajectory is a controlled synthetic alignment check, not an
assumption that independently recorded real sessions are time-aligned. CVHM
still treats trials as independent realizations when fitting. This experiment
therefore does not by itself test posterior-uncertainty calibration under the
independent-trial generative assumption.

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

`CVHM.fit()` and `infer()` accept `valid_y=None` or a mask broadcastable to
`y`; the canonical internal mask has the same shape as `y`. Missing neurons
affect neither inference nor fitted readouts. After `fit()`, `model.latent`,
`model.posterior`, and `infer()` agree under the final fitted parameters.
Gaussian and Poisson paths meet the mathematical tests above; all-observed
behavior is accounted for; documentation states the shared-readout and
session-identifiability limits; and the validation commands pass.
