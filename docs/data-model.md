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

## Latent Outputs

After fitting:

- `m`: posterior mean, `(trials, time, latent_dim)`
- `V`: posterior covariance, `(trials, time, latent_dim, latent_dim)`

## Padding Unequal Lengths

For unequal trial lengths:

- pad `y` with zeros
- set padded bins to `0` in `ymask`
- the filters will skip those bins via masking

## State-Space Dimensions

Internally, kernels expand latent dimensions into a real-valued state-space. You generally do not need to reason about this shape unless you are modifying kernel code.
