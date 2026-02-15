# Recipes

This page collects common patterns and tips.

## Single-Trial Data

If your observations are `(time, features)`, you can pass them directly. The model will expand them to `(1, time, features)`.

## Unequal Trial Lengths

Use `pad_trials` and `unpad_trials` to handle trials of different lengths:

```python
from cvhmax import pad_trials, unpad_trials

y_list = [y_trial_0, y_trial_1, y_trial_2]  # (T_0, N), (T_1, N), (T_2, N)
y, ymask, trial_lengths = pad_trials(y_list)

model.fit(y, ymask=ymask)

m_list = unpad_trials(model.posterior[0], trial_lengths)
```

Padded bins are marked `ymask=0` so the filter skips them automatically.
See `data-model.md` for the full workflow and API details.

## Choosing Observation Models

- Use `observation="Gaussian"` for real-valued observations.
- Use `observation="Poisson"` for count data.

## Debugging Convergence

- Reduce `max_iter` and `cvi_iter` to test workflows quickly.
- Start with fewer latent components (`K`) or shorter sequences.
- Ensure `JAX_ENABLE_X64=1` is set.

## Performance Tips

- Prefer `vmap` over Python loops for per-trial operations.
- Minimize host-device transfers once arrays are on device.
