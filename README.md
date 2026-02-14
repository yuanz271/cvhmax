# cvhmax

[![PyPI - Version](https://img.shields.io/pypi/v/cvhmax.svg)](https://pypi.org/project/cvhmax)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cvhmax.svg)](https://pypi.org/project/cvhmax)

Variational latent-state inference with Hida–Matérn kernels.

-----

**Table of Contents**

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Data model](#data-model)
- [Project layout](#project-layout)
- [Development](#development)
- [Testing](#testing)
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

Install the published package from PyPI:

```console
pip install cvhmax
```

or pull the bleeding edge directly from GitHub:

```console
pip install git+https://github.com/yuanz271/cvhmax
```

JAX wheels are platform specific. If you need GPU/TPU support, follow the [official installation guide](https://jax.readthedocs.io/en/latest/installation.html) before installing `cvhmax`.

## Quickstart

```python
import jax.numpy as jnp

from cvhmax.cvhm import CVHM
from cvhmax.hm import HidaMatern

# Observations shaped (trials, time, features)
y = jnp.asarray(...)            # substitute your data
ymask = jnp.ones_like(y[..., 0], dtype=jnp.uint8)

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
model.fit(y, ymask=ymask, random_state=0)
m, V = model.posterior  # latents: (trials, time, n_latents) and covariances
```

Use `model.fit_transform(...)` when you only need the posterior means. The `cvi` module exposes the underlying `Gaussian` and `Poisson` readouts if you need customised update rules.

## Data model

- **Observations (`y`)** – array shaped `(trial, time, obs_dim (N))` or `(time, obs_dim (N))`. Single-trial data are automatically expanded to match the expected rank.
- **Mask (`ymask`)** – binary array broadcastable over `y`. `1` marks observed entries, `0` marks missing/padded bins.
- **Posterior mean (`m`)** – returned via `model.posterior[0]`, shaped `(trial, time, latent_dim (K))`.
- **Posterior covariance (`V`)** – returned via `model.posterior[1]`, shaped `(trial, time, latent_dim (K), latent_dim (K))`.

Pad unequal trial lengths with zeros in `y`, mark them as missing in `ymask`, and the filters will skip them automatically.

## Project layout

- `src/cvhmax/cvhm.py` – high-level CVHM wrapper orchestrating CVI-EM.
- `src/cvhmax/cvi.py` – conjugate variational inference base class plus Gaussian/Poisson implementations.
- `src/cvhmax/filtering.py` – forward/backward information filters used for smoothing.
- `src/cvhmax/hm.py` / `src/cvhmax/hp.py` – Hida–Matérn kernels and Whittle spectral fitting utilities.
- `src/cvhmax/utils.py` – shared linear algebra helpers, optimisation utilities, and rich progress bars.
- `tests/` – regression tests mirroring the module layout (`test_cvhm.py`, `test_hp.py`, ...).

## Development

Create a development environment with [uv](https://github.com/astral-sh/uv):

```console
uv sync --group dev
```

If `uv` is unavailable, fall back to editable installs:

```console
python -m pip install -e . -r requirements-dev.lock
```

Helpful flags while debugging:

- `export JAX_ENABLE_X64=1` to keep parity with the test suite.
- `export XLA_FLAGS=--xla_force_host_platform_device_count=1` when running on CPU-only hosts.

## Testing

Run the full test matrix:

```console
pytest
```

Use module-level invocations for faster iteration, e.g. `pytest tests/test_hp.py`. Some tests generate diagnostic artefacts such as `cvhm.pdf`; remove them if you do not need the plots.

## Documentation

Local documentation lives in `docs/README.md`.

## License

`cvhmax` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
