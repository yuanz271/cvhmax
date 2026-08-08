# Kernel API and State-Covariance Refactor Plan

## Goal

Maintain a stable, validated covariance-form Hida–Matérn implementation for
built-in orders 0, 1, and 2, while preserving JAX transformation support and
current numerical behavior.

## Plan status

The supported covariance-form implementation is complete for built-in orders
0, 1, and 2. The square-root/QR implementation is deliberately an optional
follow-up, not a prerequisite for the current supported x64,
correlation-scaled CVHM path. Higher orders and new kernel families are outside
this repository's scope.

## Design decision

Keep both kernel interfaces, but assign them different roles:

- `Ks(kernelparam, tau)` is the canonical JAX-compatible functional API.
  Mapping/dictionary parameter containers are JAX pytrees and work with
  `jax.jit`, `jax.vmap`, and `jax.scan`.
- `HidaMatern.K(tau)` is a thin user-facing wrapper. It packs dataclass fields,
  delegates to `Ks`, and applies the object's optional diagonal covariance
  jitter `s` only at zero lag. Positive-lag cross-covariances remain those of
  the Hida–Matérn component.
- Functional dynamics accept explicit `jitter=` stabilization. Class dynamics
  pass `HidaMatern.s`, because process-noise subtraction and covariance solves
  can be numerically singular even when the raw covariance is valid.
- Treat `s` as a small instantaneous state-space composite component as well
  as a numerical regularizer. The intended exact nugget semantics are
  `K0 = K_HM(0) + s I` and `Kt = K_HM(tau)` for positive lags, rather than
  adding `s I` to cross-time covariance blocks. Retain the empirically chosen
  default `s=1e-5` unless realistic-example and benchmark regressions justify
  changing it.
- Use correlation/state scaling as the primary conditioning mechanism. Prefer
  Cholesky-based solves for positive-definite stationary blocks; reserve
  adaptive jitter for failed or marginal factorizations rather than increasing
  a fixed jitter globally. A square-root/QR conditional-covariance path is a
  later option for high-order or float32 support, not an immediate requirement.
- `HidaMatern.kernel(tau)` remains the real scalar, even covariance.
- `CVHM` remains responsible for correlation/state scaling because scaling is
  a state-coordinate conditioning transform, not part of the raw kernel.

The central invariant is:

```text
HidaMatern(s=0).K(tau) == Ks(params, tau)
```

up to dtype. There must be one mathematical implementation and one order
dispatch point.

## Target layers

1. **Scalar covariance**
   - Public: `matern`, `hm`, `HidaMatern.kernel`.
   - Real-valued, even in lag, jitter-free.

2. **Matérn polynomial / complex auxiliary kernel**
   - Internal positive-lag representation:
     `sigma**2 * P_order(tau; rho) * exp((1j*omega - decay)*tau)`.
   - Hand-coded orders should provide polynomial data or recurrence parameters,
     not complete state covariance matrices where possible.

3. **Derivative and state assembly**
   - Shared internal derivative recurrence and matrix assembly.
   - Own derivative signs, state orientation, and complex covariance layout.

4. **Raw state-covariance dispatch**
   - One `_raw_state_covariance(...)` dispatcher.
   - Built-in implementations registered for orders 0, 1, and 2.
   - Higher orders are rejected explicitly.
   - Returns raw, jitter-free complex covariance.
   - `order` is static metadata; numerical parameters and lag may be traced.

5. **Canonical functional API**
   - `Ks` extracts parameters and calls the raw dispatcher.
   - `Af`, `Qf`, `Ab`, and `Qb` use shared functional dynamics helpers with an
     explicit `jitter=` argument.

6. **Object convenience API**
   - `HidaMatern.K` delegates to `Ks` and applies explicit object-level
     instantaneous jitter only when `tau == 0`.
   - `make_Ks(order)` closes over static matrix-shape metadata for JAX use.
   - Class dynamics methods delegate to shared functional dynamics.
   - No class-level order-specific branches.

7. **CVHM conditioning layer**
   - Construct and reuse correlation-scaled dynamics bundles.
   - Preserve the observation-mask inverse coordinate transform.
   - Keep the default practical path `x64 + correlation scaling + s=1e-5`.
   - Validate the default and `s=0` behavior on the realistic Van der Pol
     example; the current empirical result is indistinguishable pooled R²
     (approximately `0.9722`) for `s=0` and `s=1e-5`.

## Implementation phases

### Phase 1 — Contract and regression tests

Add or finalize tests around the existing structure before changing covariance
semantics:

- `HidaMatern(s=0).K(tau)` equals `Ks(params, tau)` for built-in and generated
  orders.
- Nonzero `s` changes only the documented diagonal jitter in `K(0)` and
  leaves positive-lag `K(tau)` unchanged.
- Functional and class dynamics agree for the same raw covariance policy.
- Scalar `kernel(tau)` remains real and even.
- State-space lag contract is explicit and tested.
- Existing CVHM scaling and benchmark behavior remains unchanged.
- Add stress coverage for raw versus stabilized dynamics at high orders, small
  lags, high frequencies, and both x64 and x32 modes.

### Phase 2 — Single state-covariance dispatcher

- Confirm the existing registry and shared polynomial/derivative assembly are
  the sole built-in order dispatch path.
- Keep `Ks` as the raw, jitter-free functional API.
- Do not reintroduce public `Ks0`, `Ks1`, or `Ks2`; document their removal or
  add compatibility aliases only if an explicit downstream compatibility
  requirement appears.

### Phase 3 — Shared dynamics and covariance stabilization

- Add a shared `_dynamics_from_covariances(K0, Kt)` helper returning a JAX-
  compatible `NamedTuple`.
- Route functional `Af/Qf/Ab/Qb` through raw `Ks` and this helper.
- Route class dynamics methods through the functional helpers.
- Keep raw covariance and stabilized covariance as explicit layers.
- Implemented the instantaneous-component convention explicitly: add `s I` to
  the stationary block used for `K0`, but do not add it to positive-lag
  cross-covariance `Kt`; zero-step dynamics use `Kt=K0` and therefore have
  identity transition and zero process noise. Tests cover zero lag, positive
  lag, forward dynamics, and backward dynamics.
- Keep the raw functional API JAX-safe by making the zero-lag/stationary
  choice explicit in the dynamics helper; do not rely on a Python `tau == 0`
  branch for traced values. `HidaMatern.K(0)` may expose the convenient object
  behavior, while `Af/Qf/Ab/Qb` must explicitly request jittered `K0` and raw
  positive-lag `Kt`.
- Preserve `s=1e-5` as the conservative default while documenting that it is
  part of the covariance model and also improves numerical conditioning.
- Hermitian-symmetrize covariance inputs and derived process-noise blocks.
- Implemented Cholesky solves for stationary blocks and a fixed six-candidate
  machine-scale jitter ladder selected by finite-factor diagnostics. The
  fallback does not eigenvalue-clip derived process noise; exhausted failure
  propagates non-finite factors for explicit downstream diagnosis.
- Apply stabilization before consequential solves and covariance subtraction,
  not by adding a correction after a derived process-noise matrix.
- Preserve raw versus stabilized paths as separately testable policies.

### Phase 4 — Consolidate supported built-in orders

Status: complete. Orders 0, 1, and 2 use the shared polynomial, derivative, and
state-covariance assembly with one built-in dispatcher. No higher-order kernel
implementation or new kernel family is planned in this repository.

- Keep order-specific polynomial data behind the shared assembly.
- Keep the supported-order boundary explicit and tested.
- Compare supported behavior through algebraic identities, dynamics tests,
  benchmarks, and realistic examples.

### Phase 5 — Consolidate CVHM dynamics preparation

Status: implemented for `fit`: CVHM prepares one scaled dynamics bundle and
reuses the assembled matrices through filtering. The public matrix accessors
remain independently callable and prepare their own bundle.

- Prepare one scaled dynamics bundle per kernel for the configured time step.
- Reuse it across `CVHM.Af`, `Qf`, `Ab`, `Qb`, `Q0`, and `latent_mask` where
  semantics permit.
- Preserve the state-coordinate transform:
  `A_scaled = D @ A @ D^-1`, `Q_scaled = D @ Q @ D.H`,
  `K0_scaled = D @ K0 @ D.H`.
- Verify that the instantaneous component is transformed consistently with
  the state scaling and inverse latent observation mask.
- Add diagnostics/tests for stationary-block conditioning, Cholesky success,
  realified process-noise PSD, and Lyapunov residuals. Do not use PSD
  eigenvalue clipping in the default CVHM path.

### Phase 6 — Validation, dtype boundaries, and fallback paths

Status: implemented for the current covariance-form scope. Static order and
scalar-parameter validation, `make_Ks`, and supported-order x64/x32 regression
coverage are implemented. Orders above 2 remain explicitly unsupported.

- Keep dtype handling explicit at kernel boundaries and reject unsupported
  state-space orders above 2 rather than attempting unvalidated construction.
- Validate nonnegative static integer `order` and positive `rho` at Python
  API boundaries.
- Do not use `abs` to hide invalid stationary variances in `state_scale`.
- Document that state covariance calls use nonnegative, transition-oriented
  lags unless signed-lag support is explicitly implemented.
- Maintain stress coverage for supported orders 0–2, short lags, high
  frequencies, x64 and x32, and a jitter ladder. Record when fixed `s=1e-5`
  is sufficient,
  when adaptive jitter is activated, and when the calculation should fail
  rather than silently alter the covariance.
- Defer square-root/QR filtering and conditional-covariance propagation to a
  separate follow-up phase unless high-order or float32 requirements make the
  covariance-form path inadequate.

### Phase 7 — Realistic-case regression, documentation, and cleanup

Status: implemented. Documentation and a reproducible slow VdP jitter
regression are updated; full-suite evidence is recorded below. The reference
comparison is `s=0` and `s=1e-5`, both above the benchmark floor and near
pooled `R²=0.972`.

- Run the Van der Pol example with `s=0`, `1e-8`, `1e-5`, and `1e-3` under
  x64; compare pooled R², posterior finiteness, covariance PSD, and training
  diagnostics. Treat the current result (`R² ≈ 0.9722` with negligible
  difference between `s=0` and `s=1e-5`) as a regression reference, not as a
  universal guarantee.
- Add a small reproducible stabilization comparison test or benchmark helper
  without committing generated PDF outputs.
- Document the distinction between a state-space instantaneous component,
  observation noise, correlation scaling, adaptive jitter, and a future
  square-root implementation.

Update:

- docs/api.md;
- docs/algorithms.md;
- __init__.py exports and docstrings if public symbols change;
- tests and examples.

The built-in boundary is order 2; orders above 2 and new kernel families are
explicitly unsupported.

### Phase 8 — Optional square-root path (deferred follow-up)

Only undertake this phase if future workloads require support beyond the
validated covariance-form regime (for example, reliable very-high-order or
float32 inference):

- Implement square-root/QR conditional-covariance construction to avoid direct
  subtraction of nearly equal covariance matrices.
- If required, propagate covariance factors through filtering rather than
  repeatedly forming dense covariance matrices.
- Compare factor and covariance paths on accuracy, PSD residuals, Lyapunov
  consistency, runtime, memory, and inferred latent trajectories.
- Keep the factor path opt-in until it matches the established benchmark and
  realistic-example behavior.

## Definition of ready / acceptance criteria

The stabilization stage is complete when:

- `HidaMatern(s=0).K(tau) == Ks(params, tau)` remains true.
- For `s>0`, only the stationary/zero-lag block changes; positive-lag raw
  cross-covariances do not.
- Class and functional dynamics agree under the same explicit `K0`/`Kt`
  policy, for forward and backward directions.
- Correlation-scaled CVHM remains finite, realified process noise is PSD within
  a documented floating-point tolerance, and Lyapunov residuals remain small.
- The benchmark and full test suite pass.
- The realistic Van der Pol comparison reproduces the current reference near
  `R²=0.9722` and shows no material regression for `s=1e-5`.
- x64 behavior is supported and documented; x32/high-order failures or
  fallback activation are explicit diagnostics rather than silent corruption.

## Current validation evidence

The current repository validation is:

```text
full CPU suite: 133 passed, 39 skipped
GPU CPU–GPU parity: 1 passed
VdP demo: frozen order 0 -> 0.8677, order 1 -> 0.9721,
           order 2 -> 0.9819, estimated -> 0.9723,
           variable-length -> 0.9721
```

The benchmark remains the regression guard for inference quality and should be
run with `--run-slow` before future numerical changes.

The optional square-root/QR path remains deferred: the x64, correlation-scaled
covariance-form path is adequate for the current benchmark and realistic case.

## Validation commands

```bash
JAX_ENABLE_X64=1 uv run pytest -q tests/test_hm.py tests/test_cvhm.py
JAX_ENABLE_X64=1 uv run pytest -q tests/test_benchmark.py --run-slow
JAX_ENABLE_X64=1 uv run pytest -q
```

Also run `git diff --check` and review all public API/documentation changes.
