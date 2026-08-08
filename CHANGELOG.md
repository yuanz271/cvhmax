# Changelog

## Unreleased

## 0.2.0

### Breaking changes

- Removed the unvalidated symbolic kernel generator and its `kergen`
  dependency. Built-in Hida–Matérn state-space kernels now support orders 0,
  1, and 2.
- Removed the multi-device batch sharding path from `CVHM.fit`.

### Added

- Added optional CUDA 12 JAX installation through `cvhmax[cuda12]`.
- Added opt-in CUDA integration and CPU–GPU parity tests.
- Added `CVHM.get_config()`, `CVHM.from_config()`, `CVHM.save()`,
  `CVHM.load()`, and `CVHM.infer()`.
- Added a single-archive, data-only model format using JSON metadata and
  Equinox parameter serialization; posterior and latent caches are excluded.

### Fixed

- Replaced deprecated `optax.tree_utils.tree_l2_norm` with
  `optax.tree_utils.tree_norm`.
