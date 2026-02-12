# Recipes

This page collects common patterns and tips.

## Single-Trial Data

If your observations are `(time, features)`, you can pass them directly. The model will expand them to `(1, time, features)`.

## Unequal Trial Lengths

Pad shorter trials with zeros and set the corresponding `ymask` entries to `0`. The filter will ignore masked bins.

## Choosing Likelihoods

- Use `likelihood="Gaussian"` for real-valued observations.
- Use `likelihood="Poisson"` for count data.

## Debugging Convergence

- Reduce `max_iter` and `cvi_iter` to test workflows quickly.
- Start with smaller latent dimension or shorter sequences.
- Ensure `JAX_ENABLE_X64=1` is set.

## Performance Tips

- Prefer `vmap` over Python loops for per-trial operations.
- Minimize host-device transfers once arrays are on device.
