# Troubleshooting

## Shape Errors

- Verify `y` is `(sequences, time, features)` or `(time, features)`.
- Use `(sequences, time)` or `(sequences, time, 1)` for time-step masks, or a mask
  broadcastable to `y`, such as `(sequences, 1, features)`, for feature masks.
- During fitting, each feature must be observed in at least one time step.

## Numerical Instability

- Enable 64-bit precision in JAX for the supported kernel path:
  `JAX_ENABLE_X64=1`.
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
estimators. Inspect fitted parameters and posterior moments rather than treating
`n/a` as a posterior-finiteness check.
