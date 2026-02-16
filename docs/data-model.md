# Data Model

This page defines the canonical shapes and masking conventions used by cvhmax.

## Observations

- `y`: observations shaped `(trials, time, features)`
- Single-trial inputs are allowed as `(time, features)` and will be expanded to `(1, time, features)`

## Masking

- `valid_y`: binary mask aligned to the first two axes of `y`
- Shape is typically `(trials, time)` and must be broadcastable to `y`
- `1` indicates observed entries, `0` indicates missing/padded bins

### How masking works

A missing observation should contribute no information to the latent state
estimate. The information filter update is additive:

```
Z_post = Z_pred + J
z_post = z_pred + j
```

When `valid_y[t] = 0`, both `j[t]` and `J[t]` are set to zero, so the
filter update at that time bin reduces to:

```
Z_post = Z_pred
z_post = z_pred
```

The posterior at a masked bin equals the prediction — no observation
information is incorporated. This is implemented in `bin_info_repr`
(`utils.py`), which applies `jnp.where(valid_y, ·, 0)` to both `j` and
`J` after computing them. Since `bin_info_repr` operates on a single bin
where `valid_y` is a scalar, the mask zeros out the entire vector/matrix
rather than individual entries.

## Dimensions

The codebase uses three core dimensions.  The canonical symbols are
defined here and referenced throughout docstrings as parenthetical
annotations (e.g. `latent_dim (K)`).

| Symbol | Name | Meaning | Typical code variables |
|--------|------|---------|------------------------|
| `N` | observation dimension | number of observed features (e.g. neurons) | `obs_dim`, `n_obs`, `y_dim` |
| `K` | latent dimension | number of GP kernels / latent factors | `latent_dim`, `n_components`, `n_factors` |
| `L` | state dimension | real-valued SDE state = `2 * sum(nple)` | `state_dim`, `L`, `d_z` |

Each kernel of order `p` contributes `nple = p + 1` complex state
dimensions and `2 * nple` real dimensions (after `real_repr`).  The
total state dimension is therefore `L = 2 * sum(kernel.nple)` over all
`K` kernels.

### Key matrix shapes

| Matrix | Shape | Space | Role |
|--------|-------|-------|------|
| `y` | `(trials, T, N)` | observation | observations |
| `C` | `(N, K)` | latent | loading matrix |
| `M` | `(K, L)` | bridge | latent mask / selection matrix (owned by CVHM) |
| `Af`, `Qf`, etc. | `(L, L)` | state | SDE transition / process noise |
| `j`, `J` (CVI) | `(trials, T, K)` / `(trials, T, K, K)` | latent | pseudo-observation info |
| `j`, `J` (filter) | `(trials, T, L)` / `(trials, T, L, L)` | state | pseudo-observation info (after lift) |
| `z`, `Z` | `(trials, T, L)` / `(trials, T, L, L)` | state | information-form SDE posterior |
| `m`, `V` | `(trials, T, K)` / `(trials, T, K, K)` | latent | GP posterior mean / covariance |

## Latent Outputs

After fitting:

- `m`: posterior mean, `(trials, time, latent_dim (K))`
- `V`: posterior covariance, `(trials, time, latent_dim (K), latent_dim (K))`

## Variable-Length Trials

JAX requires rectangular arrays, but experimental trials often have
different lengths.  Use `pad_trials` before fitting and `unpad_trials`
after to handle this transparently — the filter never sees the padding
logic.

```python
from cvhmax import CVHM, HidaMatern, pad_trials, unpad_trials

# Per-trial arrays of different lengths
y_list = [y_trial_0, y_trial_1, y_trial_2]  # (300, N), (500, N), (250, N)

# Pad to rectangular arrays
y, valid_y, trial_lengths = pad_trials(y_list)
# y.shape == (3, 500, N),  valid_y.shape == (3, 500)

# Fit as usual — padded bins have valid_y=0 so the filter skips them
model = CVHM(n_components=2, dt=0.01, kernels=[HidaMatern() for _ in range(2)])
model.fit(y, valid_y=valid_y)

# Strip padding from posterior
m_list = unpad_trials(model.posterior[0], trial_lengths)
V_list = unpad_trials(model.posterior[1], trial_lengths)
# m_list[0].shape == (300, K), m_list[1].shape == (500, K), ...

# Or unpad both at once as tuples
mv_list = unpad_trials(model.posterior, trial_lengths)
# mv_list[0] == (m_trial_0, V_trial_0)
```

### How it works

`pad_trials` zero-pads shorter trials along the time axis and sets
`valid_y = 0` for padded bins.  Because the information filter adds `j`
and `J` at each bin (see [Masking](#masking) above), padded bins
contribute zero information and the filter falls through to a pure
prediction step.

Pre-existing missing values (`valid_y = 0`) within original trials are
preserved — padding is right-sided (zeros appended after the last
original time bin).

### API

| Function | Signature | Returns |
|----------|-----------|---------|
| `pad_trials` | `(y_list, valid_y_list=None)` | `(y, valid_y, trial_lengths)` |
| `unpad_trials` | `(arrays, trial_lengths)` | `list[Array]` or `list[tuple[Array, ...]]` |

`unpad_trials` accepts either a single array or a tuple of arrays.
When given a tuple, each list element is a tuple of the unpadded
arrays for that trial.
