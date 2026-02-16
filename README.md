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
- Whittle-style spectral hyperparameter refinement through `hp.whittle`.
- Progress-aware training loops with `rich` instrumentation.

## Installation

Requires Python 3.12+ and JAX.

```console
pip install git+https://github.com/yuanz271/cvhmax
```

JAX wheels are platform-specific. For GPU/TPU support, follow the [official installation guide](https://jax.readthedocs.io/en/latest/installation.html) before installing `cvhmax`.

## Quickstart

```python
import jax.numpy as jnp

from cvhmax.cvhm import CVHM
from cvhmax.hm import HidaMatern

# Observations shaped (trials, time, features)
y = jnp.asarray(...)            # substitute your data
valid_y = jnp.ones_like(y[..., 0], dtype=jnp.uint8)

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
)
model.fit(y, valid_y=valid_y, random_state=0)
m, V = model.posterior  # latents: (trials, time, n_latents) and covariances
```

Use `model.fit_transform(...)` when you only need the posterior means. For smoother latent trajectories, use higher-order kernels (e.g. `order=2` for Matern-5/2).

## Data model

- **Observations (`y`)** -- `(trials, time, obs_dim)` or `(time, obs_dim)`.
- **Mask (`valid_y`)** -- binary, `1` = observed, `0` = missing/padded.
- **Posterior mean (`m`)** -- `(trials, time, latent_dim)`.
- **Posterior covariance (`V`)** -- `(trials, time, latent_dim, latent_dim)`.

JAX requires rectangular arrays for batched operations -- use `pad_trials` / `unpad_trials` for variable-length trials. See `docs/data-model.md` for details.

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
