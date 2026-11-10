# Recipes

This page collects common patterns and tips.

## Single-Sequence Data

If your observations are `(time, features)`, you can pass them directly. The model will expand them to `(1, time, features)`.

## Unequal Sequence Lengths

Use `pad_sequences` and `unpad_sequences` to handle sequences of different lengths:

```python
from cvhmax import pad_sequences, unpad_sequences

y_list = [y_sequence_0, y_sequence_1, y_sequence_2]  # (T_0, N), (T_1, N), (T_2, N)
y, valid_y, sequence_lengths = pad_sequences(y_list)

model.fit(y, valid_y=valid_y)

m_list = unpad_sequences(model.posterior[0], sequence_lengths)
```

Padded time steps are marked `valid_y=0` so the filter skips them automatically.
See `data-model.md` for the full workflow and API details.

## Choosing Observation Models

- Use `observation="Gaussian"` for real-valued observations.
- Use `observation="Poisson"` for count data.

## Debugging Convergence

- Reduce `max_iter` and `cvi_iter` to test workflows quickly.
- Start with fewer latent components (`K`) or shorter sequences.
- Enable 64-bit precision in JAX if you need maximum numerical stability.

## Performance Tips

- Prefer `vmap` over Python loops for per-sequence operations.
- Minimize host-device transfers once arrays are on device.
