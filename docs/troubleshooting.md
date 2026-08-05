# Troubleshooting

## Shape Errors

- Verify `y` is `(trials, time, features)` or `(time, features)`.
- Ensure `valid_y` is broadcastable to the first two axes of `y`.

## Numerical Instability

- Enable 64-bit precision in JAX for the supported kernel path:
  `JAX_ENABLE_X64=1`.
- Keep CVHM correlation scaling enabled; it is the primary conditioning
  transform for derivative-state covariance blocks.
- `HidaMatern(s=1e-5)` is the conservative default. It represents a small
  instantaneous state-space covariance component and is added only to `K(0)`;
  it is not a universal replacement for correlation scaling.
- Dynamics use Cholesky stationary solves and a bounded machine-scale fallback
  ladder. Derived process noise is symmetrized, not eigenvalue-clipped. The
  built-in Hida–Matérn state-space path supports orders 0, 1, and 2.
- If float64 is disabled, expect larger roundoff at small lags. For custom
  higher-order kernels, validate numerical conditioning separately rather than
  silently increasing jitter indefinitely.
- Start with smaller `max_iter`/`cvi_iter` and gradually increase.

## Training progress shows `Negative ELL n/a`

`n/a` means the observation model did not return an M-step objective, as in a
frozen-readout model. It is not a NaN posterior diagnostic. A finite objective
is displayed numerically when the readout update provides one.

## JAX Warnings

Some JAX linear algebra warnings indicate future behavior changes. Prefer `jnp.linalg.solve(A, b[..., None])[..., 0]` for batched 1D solves.
