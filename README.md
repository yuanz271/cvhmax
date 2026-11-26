# cvhmax

Variational latent-state inference with Hida-Matern kernels.

-----

**Table of Contents**

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Data model](#data-model)
- [Citation](#citation)
- [License](#license)

## Overview

`cvhmax` implements conjugate variational inference for latent Gaussian processes whose dynamics are described by Hida–Matérn kernels. It couples information-form Kalman filtering with CVI-EM updates so you can recover smooth latent trajectories from high-dimensional observations with either Gaussian or Poisson likelihoods. The package is written in JAX and keeps the full inference loop jittable.

## Features

- CVI-EM loop with Gaussian and Poisson readouts via the `CVI` registry.
- Hida–Matérn kernels parameterised as linear Gaussian state-space models.
- Forward/backward information filtering (`bifilter`) for smoothing latent trajectories.

## Installation

Requires Python 3.12+ and JAX.

```console
pip install git+https://github.com/yuanz271/cvhmax
```

The default installation uses the CPU JAX backend. For an NVIDIA GPU with CUDA 12 support, install the optional `cuda12` extra:

```console
pip install "cvhmax[cuda12]"
```

With `uv`, use `uv sync --extra cuda12` for a checkout. The extra uses JAX's pip-managed CUDA 12 plugin and runtime libraries. See the [official JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for driver and platform requirements.

Run the opt-in GPU integration test with CUDA enforced (to prevent CPU fallback):

```console
JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda,cpu uv run pytest tests/test_gpu.py --run-gpu -q
```

## Quickstart

```python
import jax.numpy as jnp

from cvhmax.cvhm import CVHM
from cvhmax.hm import HidaMatern

# Observations shaped (sequences, time, features)
y = jnp.asarray(...)            # substitute your data
valid_y = jnp.ones_like(y, dtype=bool)

n_latents = 2
dt = 1.0
kernels = [
    HidaMatern(sigma=1.0, rho=50.0, omega=0.0, order=0)
    for _ in range(n_latents)
]

model = CVHM(
    n_components=n_latents,
    dt=dt,
    kernels=kernels,
    observation="Poisson",  # or "Gaussian"
    max_iter=5,
    tol=0.05,
    min_iter=2,
    convergence_patience=2,
)
model.fit(y, valid_y=valid_y, random_state=0)
m, V = model.posterior  # latents: (sequences, time, n_latents) and covariances
```

Use `model.fit_transform(...)` when you only need the posterior means. The
readout uses its fitted loading matrix directly; the fixed GP kernel amplitudes
set the latent prior scale. `valid_y` can mask individual features and broadcast
to the shape of `y`; time-step masks shaped `(sequences, time)` are also supported.
Gaussian observation noise is diagonal. The inner CVI loop runs 20 iterations by
default; `cvi_iter` adjusts the accuracy–compute tradeoff for Poisson. For the
built-in Gaussian readout, `cvi_iter=0` gives the same posterior. For smoother
latent trajectories, use higher-order kernels (e.g. `order=2` for Matern-5/2).

`loading` chooses how the loadings are fitted. The default `"ridge"` holds a
fixed precision. `"ard"` fits one precision per latent for Gaussian or Poisson
observations; larger values in `model.params.alpha` indicate more strongly
shrunk loading columns. Poisson ARD uses a Laplace approximation and may be less
reliable when counts are sparse, for example
`CVHM(..., observation="Poisson", loading="ard")`. `"fixed"` holds the
initialized readout fixed. `loading_config` overrides the kind's defaults, such
as `{"alpha": 0.5}` or `{"iterations": 20}`. See `docs/algorithms.md` for ARD's
assumptions.

The Van der Pol demo compares Matérn orders 0, 1, and 2 under a frozen
readout, holding the data, readout, and other kernel hyperparameters fixed:

```console
JAX_ENABLE_X64=1 python examples/demo_vdp.py
```

A separate demo shows ARD. It simulates two Van der Pol latents, observes them
through a Gaussian readout, and fits five latents with `loading="ridge"` and
`loading="ard"` on the same data. ARD separates the fitted precisions by more than
three orders of magnitude and drives the three surplus loadings to numerical
zero, where ridge only shrinks them. It prints the ranking and writes
`examples/demo_ard.pdf`:

```console
JAX_ENABLE_X64=1 python examples/demo_ard.py
```

## Numerical stability

The recommended inference path is `JAX_ENABLE_X64=1` with CVHM correlation
scaling. `HidaMatern(s=1e-5)` is the conservative default: it represents a
small instantaneous state-space covariance component added at zero lag and
also regularizes the stationary solve. Positive-lag cross-covariances remain
raw. Dynamics use Cholesky-based stationary solves with a bounded numerical
fallback; derived process noise is symmetrized rather than eigenvalue-clipped.

For static-shape JAX transformations, use `make_Ks(order)` and close over the
integer state order. The built-in state-space implementation supports orders
0, 1, and 2; users can supply custom kernel objects through `CVHM(kernels=...)`
when they need other kernel families or state dimensions.

## Data model

- **Observations (`y`)** -- `(sequences, time, obs_dim)` or `(time, obs_dim)`.
- **Mask (`valid_y`)** -- binary, `1` = observed, `0` = missing/padded.
- **Posterior mean (`m`)** -- `(sequences, time, latent_dim)`.
- **Posterior covariance (`V`)** -- `(sequences, time, latent_dim, latent_dim)`.

Sequences have independent latent trajectories but share the kernels, time step,
and readout. Feature indices must identify the same variables across sequences.
JAX requires rectangular arrays for batched operations -- use `pad_sequences` /
`unpad_sequences` for variable-length sequences. See `docs/data-model.md` for details.

## Citation

This package is a JAX reimplementation of the methods described in:

> Dowling, M., Zhao, Y., & Park, I. M. (2023). Linear Time GPs for Inferring Latent Trajectories from Neural Spike Trains. *Proceedings of the 40th International Conference on Machine Learning (ICML)*. \[[OpenReview](https://openreview.net/forum?id=1hWB5XEUMa)\] \[[arXiv](https://arxiv.org/abs/2306.01802)\]

```bibtex
@InProceedings{Dowling2023c,
  author    = {Matthew Dowling and Yuan Zhao and Il Memming Park},
  booktitle = {International Conference on Machine Learning (ICML)},
  title     = {Linear time {GP}s for inferring latent trajectories from neural spike trains},
  year      = {2023},
  url       = {https://openreview.net/forum?id=1hWB5XEUMa},
}
```

## License

`cvhmax` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
