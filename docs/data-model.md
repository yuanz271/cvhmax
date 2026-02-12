# Data Model

This page defines the canonical shapes and masking conventions used by cvhmax.

## Observations

- `y`: observations shaped `(trials, time, features)`
- Single-trial inputs are allowed as `(time, features)` and will be expanded to `(1, time, features)`

## Masking

- `ymask`: binary mask aligned to the first two axes of `y`
- Shape is typically `(trials, time)` and must be broadcastable to `y`
- `1` indicates observed entries, `0` indicates missing/padded bins

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
