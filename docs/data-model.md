# Data Model

This page defines the canonical shapes and masking conventions used by cvhmax.

## Observations

- `y`: observations shaped `(sequences, time, features)`
- Single-sequence inputs are allowed as `(time, features)` and will be expanded to `(1, time, features)`

Each sequence is an independent latent GP realization, conditional on the
shared model parameters. Sequences use the same kernels, time step `dt`, and
readout (`C`, `d`, and `R` where applicable). A feature index must therefore
refer to the same observed variable across sequences; missing features are
represented by the mask. The model does not require sequences to share the
same realized trajectory.

## Masking

`valid_y` is a boolean mask broadcastable to `y`; `True` means observed.
`None` means all supplied entries are observed. Common batched mask shapes are:

- `(sequences, time, N)`: individual feature-time entries;
- `(sequences, time)` or `(sequences, time, 1)`: whole time steps, shared across features;
- `(sequences, 1, N)`: feature availability constant over time within each sequence.

For a single-sequence `(time, N)` input, `(time,)` is also a time-step mask. The
API expands these time-step masks and broadcasts them to `y.shape`. In an
ambiguous shape, a mask matching `y.shape[:-1]` is interpreted as a time-step
mask; use an explicit singleton axis to specify another interpretation. Missing
features may be filled with any finite value in `y`; they contribute neither
readout information nor readout updates. Every feature must be observed at least
once across training sequences and time steps. An already-fitted model can infer
from any subset.

### How masking works

Gaussian information increments sum the contributions of observed features
using diagonal noise variance; Poisson increments likewise sum only observed
features. The filter update is additive:

```
Z_post = Z_pred + J
z_post = z_pred + j
```

A fully masked time step contributes zero `j` and `J`. The smoothed posterior can
still use information from other time steps.

## Dimensions

The codebase uses three core dimensions.  The canonical symbols are
defined here and referenced throughout docstrings as parenthetical
annotations (e.g. `latent_dim (K)`).

| Symbol | Name | Meaning | Typical code variables |
|--------|------|---------|------------------------|
| `N` | observation dimension | number of features | `obs_dim`, `n_obs`, `y_dim` |
| `K` | latent dimension | number of GP kernels / latent factors | `latent_dim`, `n_components`, `n_factors` |
| `L` | state dimension | real-valued SDE state = `2 * sum(nple)` | `state_dim`, `L`, `d_z` |

Each kernel of order `p` contributes `nple = p + 1` complex state
dimensions and `2 * nple` real dimensions (after `real_repr`).  The
total state dimension is therefore `L = 2 * sum(kernel.nple)` over all
`K` kernels.

### Key matrix shapes

| Matrix | Shape | Space | Role |
|--------|-------|-------|------|
| `y` | `(sequences, T, N)` | observation | observations |
| `C` | `(N, K)` | latent | loading matrix |
| `M` | `(K, L)` | bridge | latent mask / selection matrix (owned by CVHM) |
| `Af`, `Qf`, etc. | `(L, L)` | state | SDE transition / process noise |
| `j`, `J` (CVI) | `(sequences, T, K)` / `(sequences, T, K, K)` | latent | pseudo-observation info |
| `j`, `J` (filter) | `(sequences, T, L)` / `(sequences, T, L, L)` | state | pseudo-observation info (after lift) |
| `z`, `Z` | `(sequences, T, L)` / `(sequences, T, L, L)` | state | information-form SDE posterior |
| `m`, `V` | `(sequences, T, K)` / `(sequences, T, K, K)` | latent | GP posterior mean / covariance |

The built-in readouts use the loading matrix `C`, whose entries retain
observation-specific magnitudes. In the SDE state space, the corresponding
observation matrix is `H = C @ M`.

## Latent Outputs

After fitting:

- `m`: posterior mean, `(sequences, time, latent_dim (K))`
- `V`: posterior covariance, `(sequences, time, latent_dim (K), latent_dim (K))`

## Variable-Length Sequences

JAX requires rectangular arrays, but sequences often have different lengths.  Use `pad_sequences` before fitting and `unpad_sequences`
after to handle this transparently — the filter never sees the padding
logic.

```python
from cvhmax import CVHM, HidaMatern, pad_sequences, unpad_sequences

# Per-sequence arrays of different lengths
y_list = [y_sequence_0, y_sequence_1, y_sequence_2]  # (300, N), (500, N), (250, N)

# Pad to rectangular arrays
y, valid_y, sequence_lengths = pad_sequences(y_list)
# y.shape == (3, 500, N),  valid_y.shape == (3, 500, N)

# Fit as usual — padded time steps have valid_y=0 so the filter skips them
model = CVHM(n_components=2, dt=0.01, kernels=[HidaMatern() for _ in range(2)])
model.fit(y, valid_y=valid_y)

# Strip padding from posterior
m_list = unpad_sequences(model.posterior[0], sequence_lengths)
V_list = unpad_sequences(model.posterior[1], sequence_lengths)
# m_list[0].shape == (300, K), m_list[1].shape == (500, K), ...

# Or unpad both at once as tuples
mv_list = unpad_sequences(model.posterior, sequence_lengths)
# mv_list[0] == (m_sequence_0, V_sequence_0)
```

### How it works

`pad_sequences` zero-pads shorter sequences along the time axis and sets
`valid_y = 0` for every feature in padded time steps. Because the information filter adds `j`
and `J` at each time step (see [Masking](#masking) above), padded time steps
contribute zero information and the filter falls through to a pure
prediction step.

Pre-existing missing values (`valid_y = 0`) within original sequences are
preserved — padding is right-sided (zeros appended after the last
original time step).

### API

| Function | Signature | Returns |
|----------|-----------|---------|
| `pad_sequences` | `(y_list, valid_y_list=None)` | `(y, valid_y, sequence_lengths)` |
| `unpad_sequences` | `(arrays, sequence_lengths)` | `list[Array]` or `list[tuple[Array, ...]]` |

`unpad_sequences` accepts either a single array or a tuple of arrays.
When given a tuple, each list element is a tuple of the unpadded
arrays for that sequence.
