# Troubleshooting

## Shape Errors

- Verify `y` is `(trials, time, features)` or `(time, features)`.
- Ensure `valid_y` is broadcastable to the first two axes of `y`.

## Numerical Instability

- Enable 64-bit precision: `export JAX_ENABLE_X64=1`.
- Start with smaller `max_iter`/`cvi_iter` and gradually increase.

## JAX Warnings

Some JAX linear algebra warnings indicate future behavior changes. Prefer `jnp.linalg.solve(A, b[..., None])[..., 0]` for batched 1D solves.

## Test Artifacts

Some tests emit plots (e.g., `cvhm.pdf`). Remove them if they are not needed.
