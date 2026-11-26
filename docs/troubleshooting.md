# Troubleshooting

## Shape Errors

- Verify `y` is `(sequences, time, features)` or `(time, features)`.
- Use `(sequences, time)` or `(sequences, time, 1)` for time-step masks, or a mask
  broadcastable to `y`, such as `(sequences, 1, features)`, for feature masks.
- During fitting, each feature must be observed in at least one time step.

## Numerical Instability

- In float32, keep a non-zero `HidaMatern(s=...)`: the default `1e-5` is what
  keeps the process noise positive definite. The dynamics form
  `Q = K(0) - K(t) K(0)^{-1} K(t)^H`, a difference of nearly equal covariances,
  which loses positive definiteness in float32 when `s=0` and `dt` is small
  relative to `rho`; a Gaussian or ARD fit then stops with `FloatingPointError`,
  which is raised when the fitted readout parameters go non-finite. The returned
  posterior is not checked, so a Poisson fit at a fixed precision can instead
  report `converged_ = False` with a non-finite `posterior`; treat that as a
  failed fit. `JAX_ENABLE_X64=1` removes the constraint by computing
  the kernel blocks in float64, and the benchmark and examples reach the same
  accuracy in either precision, with roundoff about `1e-7` instead of `1e-16`.
  Note that `utils.TAU = 1e-6` is only eight float32 epsilons.
- On an accelerator, float32 matmuls default to a reduced-precision mode (TF32
  on CUDA). The kernel dynamics are computed at full precision regardless, so
  process noise stays positive definite, but the rest of the pipeline inherits
  TF32 rounding (around `1e-3` relative). Set
  `JAX_DEFAULT_MATMUL_PRECISION=highest` there if you need float32 to match the
  float64 results within the tolerances the test suite uses, or stay in float64.
- CVHM applies correlation scaling internally; retain this transformation
  when implementing custom filtering or kernel code because it improves
  conditioning of derivative-state covariance blocks.
- `HidaMatern(s=1e-5)` is the conservative default. It represents a small
  instantaneous state-space covariance component and is added only to `K(0)`;
  it is not a universal replacement for correlation scaling.
- Dynamics use Cholesky stationary solves and a bounded machine-scale fallback
  ladder. Derived process noise is symmetrized, not eigenvalue-clipped. The
  built-in Hida–Matérn state-space path supports orders 0, 1, and 2.
- If float64 is disabled, expect larger roundoff at small lags. For custom
  higher-order kernels, validate numerical conditioning separately rather than
  silently increasing jitter indefinitely.
- Poisson rates are capped at `exp(MAX_LOGRATE)`, about 1100 counts per time
  step, as a guard against overflow. It should never be active at a solution.
  If a fit needs it, check the units of `y` (counts per bin, not per second) and
  the scale of the data; the posterior covariance is not meaningful at the
  affected time steps.
- Start with smaller `max_iter`/`cvi_iter` and gradually increase. Convergence
  detection is mandatory; adjust the positive `tol`, `min_iter`, and
  `convergence_patience` settings and inspect `model.converged_` and
  `model.n_iter_`. Held-out metrics are not used as the stopping criterion.

## Training progress shows `Readout score n/a`

`n/a` indicates a non-finite or unreported readout score. Gaussian ridge and
fixed-readout models do not report a score. For Poisson ridge, the score is the
unpenalized negative expected log-likelihood per observed time step; for ARD,
it is the conditional evidence plus the precision-prior term
(Laplace-approximated for Poisson). These scores
have different scales and directions and should not be compared across
loading kinds. Inspect fitted parameters and posterior moments rather than treating
`n/a` as a posterior-finiteness check.
