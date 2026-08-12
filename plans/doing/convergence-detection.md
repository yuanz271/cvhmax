# CVHM Convergence Detection Implementation Plan

## Goal

Add mandatory outer-loop convergence detection to `CVHM.fit`, with
`max_iter` retained as a hard iteration cap.

The stopping rule must be model-fit based, not validation based: held-out data
must never influence EM stopping. The latent GP prior is fixed during `fit`, so
convergence is detected solely from changes in the CVHM-owned GLM readout
parameters.

## Proposed public API

Add these mandatory `CVHM` configuration fields:

```python
tol: float = 0.05
min_iter: int = 2
convergence_patience: int = 2
```

Convergence detection is always enabled; there is no `tol=None` escape hatch.
The `max_iter` setting remains a hard iteration cap. `tol` must be finite and
strictly positive.
- `min_iter` is the minimum number of completed outer iterations before a stop
  is allowed. The default is 2.
- `convergence_patience` is the number of consecutive iterations satisfying
  all convergence criteria. The default is 2.
- `min_iter <= max_iter` and `convergence_patience >= 1` are required.
- Defaults are deliberately conservative: `min_iter=2` and
  `convergence_patience=2`.

Expose fitted diagnostics:

```python
model.converged_  # bool
model.n_iter_     # completed outer iterations
```

These are fit diagnostics, not required model parameters. `fit()` continues to
return `self`.

## Convergence criterion

Let the previous and current outer-iteration readout states be denoted by `-`
and `+`. Because the GP kernels, prior, time step, and data are fixed during
`fit`, identical readout parameters deterministically imply the same posterior
(up to numerical error). Therefore the initial detector compares only
CVHM-owned readout parameters; it does not compare latent posteriors. Loading
changes are compared after GLM orthogonal Procrustes alignment.

For the GLM loading matrix, align the new loading to the old loading by
orthogonal Procrustes:

$$
Q^* = \arg\min_{Q^TQ=I}\|C^+Q-C^-\|_F,
\qquad
\delta_C = \frac{\|C^+Q^*-C^-\|_F}
{\max(1,\|C^-\|_F)}.
$$

This handles the latent sign/permutation/orthogonal rotation gauge while
assuming the current fixed-prior GLM representation.

For the readout bias:

$$
\delta_d = \frac{\|d^+ - d^-\|_2}
{\max(1,\|d^-\|_2)}.
$$

Compare the observation-noise field unconditionally:

$$
\delta_R = \frac{\|R^+ - R^-\|_F}
{\max(1,\|R^-\|_F)}.
$$

`Params.R` is always a JAX-compatible array or scalar, never `None`. Gaussian
readouts store their observation covariance matrix. Poisson readouts do not use
`R`; they initialize it to the JAX scalar sentinel `jnp.asarray(0.0)` and
preserve that sentinel on every update, so `delta_R` is exactly zero without a
Python optional-value branch. The sentinel is not an observation covariance and
must not be passed to Gaussian likelihood calculations.

The scalar convergence metric is:

```text
max(delta_C, delta_d, delta_R) <= tol
```

where `delta_R` is always included; it is zero for the Poisson scalar zero
sentinel. `delta_C` is computed from the Procrustes-aligned loading, while `d`
and `R` are compared directly. Do not compare full latent posterior arrays or
map them into observation space.

The first outer iteration runs unconditionally because no previous
readout-parameter state exists for comparison. Track whether a previous
parameter state exists; the first metric is `jnp.inf`. Convergence
is eligible only after the second completed outer iteration and after at least
`min_iter` completed iterations. It then requires the criterion to pass for
`convergence_patience` consecutive completed iterations. Any NaN or Inf metric
is non-converged and resets the consecutive-passing count.

Do not use held-out likelihood, validation R², or the Gaussian `nell` value as
the sole criterion. Gaussian `nell` is intentionally `NaN` in the current
implementation.

## Implementation steps

### 1. Add configuration and validation

Add `tol`, `min_iter`, and `convergence_patience` to `CVHM`.

Validate at the Python API boundary:

- `tol` is finite and strictly positive;
- `min_iter >= 1`;
- `convergence_patience >= 1`;
- `min_iter <= max_iter`.

Include these fields in `get_config()` / `from_config()` and the model
serialization manifest, since they affect fitting behavior. The repository
does not promise cross-version archive compatibility; no migration mechanism is
needed.

### 2. Implement pure metric helpers

Add small JAX-compatible helpers in `cvhm.py` or a focused utility module,
scoped initially to the built-in `Params` structure and built-in-compatible
readouts (`Gaussian`, `Poisson`, and subclasses using `Params`). Treat `R` as a
JAX array/scalar in all branches; do not use `None` checks in the convergence
path:

- relative Frobenius change with `max(1, norm(reference))` denominator;
- Procrustes-aligned GLM loading (`C`) change;
- bias (`d`) change;
- unconditional JAX-compatible noise (`R`) change;
- aggregation to a scalar readout-parameter convergence metric.

Helpers must be pure, shape-stable, and usable inside `jax.lax.while_loop` or
`jax.lax.fori_loop`. Assume the documented `Params` structure and finite
numerical leaves inside the metric and JAX loop.

### 3. Refactor the outer loop

Preserve the existing EM step numerics, but expose both the previous and new
outer states around `em_step`.

Use a fixed-shape `jax.lax.while_loop` carry containing:

- iteration count;
- model parameters;
- filtering/CVI state;
- posterior moments needed for the fit state and final result (not used by
  convergence metrics);
- objective value;
- consecutive-convergence count;
- convergence flag; and
- latest convergence metric.

The loop condition is:

```text
iteration < max_iter and not converged
```

A stop is accepted only when a previous readout-parameter state exists,
`iteration >= min_iter`, and the finite convergence metric is at or below `tol`
for
`convergence_patience` consecutive iterations.

Do not use a Python break inside the JAX loop. Keep array shapes static.

### 4. Preserve progress and diagnostics

The training progress callback should continue to report each completed
iteration. When convergence detection is enabled, include the scalar
convergence metric and, where practical, the current patience count in the
progress fields.

After fitting, set:

```python
self.converged_ = bool(converged)
self.n_iter_ = int(iteration)
```

If the iteration cap is reached first, set `converged_ = False` and
`n_iter_ == max_iter`. Do not create `converged_` or `n_iter_` before `fit()`
completes. Non-finite convergence metrics are always non-converged.

### 5. Add tests

Add discriminative tests for:

- the default tolerance (`0.05`) is mandatory and allows stopping before
  `max_iter`;
- a loose tolerance stops before `max_iter`;
- a strict tolerance reaches the cap and reports non-convergence;
- `min_iter` prevents premature stopping;
- `convergence_patience` requires consecutive passing iterations;
- convergence evaluates the JAX-compatible `R` field;
- Poisson convergence has the scalar zero sentinel on every iteration and
  therefore `delta_R == 0`;
- fixed readout parameters imply deterministic repeated posterior inference;
- latent posterior arrays are not required by the detector;
- `converged_` and `n_iter_` are correct after fitting;
- serialization round-trips the convergence configuration fields;
- `Params.R` is always a JAX-compatible leaf, including Poisson fits; and
- GLM loading change is invariant to an orthogonal latent rotation.

Use tiny deterministic data and `max_iter` values that keep tests fast. Avoid
asserting a particular convergence iteration for numerically marginal data;
assert the specified numerical and algorithmic invariants instead.

### 6. Update documentation

Update `docs/algorithms.md` with the convergence design, including:

- the distinction between predictive evaluation and training convergence;
- the three readout-parameter changes;
- the fixed-prior rationale for excluding posterior comparisons;
- patience and minimum-iteration semantics; and
- the fact that held-out metrics are not stopping criteria.

Update `docs/api.md`, README configuration examples, and
`docs/troubleshooting.md` with the new fields and diagnostics. State that
convergence compares pre/post EM outer-iteration states, not raw latent
coordinates, and excludes held-out metrics.

Update `docs/serialization.md` and the completed serialization plan to list
`tol`, `min_iter`, and `convergence_patience` as CVHM-owned configuration. Do
not serialize posterior, convergence counters, or convergence caches as model
state.

## Validation commands

```bash
uv run ruff check src/cvhmax tests/test_cvhm.py tests/test_serialization.py
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu uv run pytest tests/test_cvhm.py -q
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu uv run pytest tests/test_serialization.py -q
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu uv run pytest -q
JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda,cpu \
  uv run pytest tests/test_gpu.py --run-gpu -q
```

Run the benchmark with `--run-slow` before declaring no inference regression.

## Definition of done

Convergence detection is mandatory. Every fit stops only after the configured
minimum iteration count and patience, or at the hard `max_iter` cap, using the
three finite readout-parameter metrics above. The result exposes accurate convergence
diagnostics, is JAX-compatible, survives model serialization, and passes the
focused, full, benchmark, and GPU checks.
