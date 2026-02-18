# Troubleshooting

## Shape Errors

- Verify `y` is `(trials, time, features)` or `(time, features)`.
- Ensure `valid_y` is broadcastable to the first two axes of `y`.

## Numerical Instability

- Enable 64-bit precision in JAX for maximum stability (kernel blocks are
  computed in float64 and then cast back).
- If float64 is disabled, increase kernel jitter via `HidaMatern(s=...)`.
- Start with smaller `max_iter`/`cvi_iter` and gradually increase.

## JAX Warnings

Some JAX linear algebra warnings indicate future behavior changes. Prefer `jnp.linalg.solve(A, b[..., None])[..., 0]` for batched 1D solves.
