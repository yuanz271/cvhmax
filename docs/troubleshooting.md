# Troubleshooting

## Shape Errors

- Verify `y` is `(trials, time, features)` or `(time, features)`.
- Ensure `valid_y` has shape `(trials, time)`, or shape `(time,)` when `y`
  is provided as `(time, features)`.

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

## Training progress shows `Negative ELL n/a`

`n/a` means the observation model did not return an M-step objective, as in a
frozen-readout model. It is not a NaN posterior diagnostic. A finite objective
is displayed numerically when the readout update provides one.
