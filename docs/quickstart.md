# Quickstart

This guide shows the minimum workflow to fit a CVHM model and retrieve the posterior.

## Gaussian Likelihood Example

```python
import jax.numpy as jnp

from cvhmax.cvhm import CVHM
from cvhmax.hm import HidaMatern

# Observations shaped (trials, time, features)
y = jnp.asarray(...)  # your data
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
    observation="Gaussian",
    max_iter=5,
)
model.fit(y, ymask=ymask, random_state=0)

m, V = model.posterior
```

## Poisson Likelihood Example

```python
import jax.numpy as jnp

from cvhmax.cvhm import CVHM
from cvhmax.hm import HidaMatern

y = jnp.asarray(...)
ymask = jnp.ones_like(y[..., 0], dtype=jnp.uint8)

model = CVHM(
    n_components=2,
    dt=1.0,
    kernels=[HidaMatern(order=0) for _ in range(2)],
    observation="Poisson",
    max_iter=5,
)
model.fit(y, ymask=ymask, random_state=0)

m, V = model.posterior
```

## Higher-Order Kernels

By default the examples above use `order=0` (Matern-1/2). The kernel
generator subpackage enables arbitrary smoothness orders:

```python
kernels = [
    HidaMatern(sigma=1.0, rho=50.0, omega=0.0, order=2)  # Matern-5/2
    for _ in range(n_latents)
]
```

Higher orders produce smoother latent trajectories. See
`kernel-generator.md` for the full usage guide.

## What You Get Back

- `m`: posterior means shaped `(trials, time, latent_dim)`
- `V`: posterior covariances shaped `(trials, time, latent_dim, latent_dim)`

Next: `data-model.md` for full shape and mask conventions.
