# Data Model

This page defines the canonical shapes and masking conventions used by cvhmax.

## Observations

- `y`: observations shaped `(trials, time, features)`
- Single-trial inputs are allowed as `(time, features)` and will be expanded to `(1, time, features)`

## Masking

- `ymask`: binary mask aligned to the first two axes of `y`
- Shape is typically `(trials, time)` and must be broadcastable to `y`
- `1` indicates observed entries, `0` indicates missing/padded bins

### How masking works

A missing observation should contribute no information to the latent state
estimate. The information filter update is additive:

```
Z_post = Z_pred + J
z_post = z_pred + j
```

When `ymask[t] = 0`, both `j[t]` and `J[t]` are set to zero, so the
filter update at that time bin reduces to:

```
Z_post = Z_pred
z_post = z_pred
```

The posterior at a masked bin equals the prediction — no observation
information is incorporated. This is implemented in `bin_info_repr`
(`utils.py`), which applies `jnp.where(ymask, ·, 0)` to both `j` and
`J` after computing them. Since `bin_info_repr` operates on a single bin
where `ymask` is a scalar, the mask zeros out the entire vector/matrix
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

| Matrix | Shape | Role |
|--------|-------|------|
| `y` | `(trials, T, N)` | observations |
| `C` | `(N, K)` | loading matrix |
| `M` | `(K, L)` | latent mask / selection matrix |
| `H = C @ M` | `(N, L)` | effective observation matrix |
| `Af`, `Qf`, etc. | `(L, L)` | SDE transition / process noise |
| `z`, `Z` | `(trials, T, L)` / `(trials, T, L, L)` | information-form SDE state |
| `m`, `V` | `(trials, T, K)` / `(trials, T, K, K)` | GP posterior mean / covariance |

## Latent Outputs

After fitting:

- `m`: posterior mean, `(trials, time, latent_dim (K))`
- `V`: posterior covariance, `(trials, time, latent_dim (K), latent_dim (K))`

## Padding Unequal Lengths

For unequal trial lengths:

- pad `y` with zeros
- set padded bins to `0` in `ymask`
- the filters will skip those bins via masking
