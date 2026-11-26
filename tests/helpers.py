"""Shared scaffolding for the test suite. Not part of the package."""

import numpy as np
from jax import numpy as jnp


def sample_gp_latents(kernel, n_time, dt, n_latents, rng, n_sequences=1, jitter=None):
    """Draw independent latent trajectories from a kernel's prior.

    Every latent of every sequence is an independent sample from the zero-mean
    Gaussian process whose covariance is ``kernel.kernel(|t - t'|)`` on a grid
    with spacing ``dt``.

    Parameters
    ----------
    kernel : HidaMatern
        Prior covariance, shared by all latents.
    n_time : int
        Number of time steps.
    dt : float
        Spacing of the time grid.
    n_latents : int
        Latents per sequence.
    rng : numpy.random.Generator
        Source of randomness.
    n_sequences : int, default=1
        Number of independent sequences.
    jitter : float, optional
        Diagonal added before the Cholesky factorisation. Defaults to
        ``kernel.s``, the kernel's own instantaneous component.

    Returns
    -------
    ndarray, shape (n_sequences, n_time, n_latents)
    """
    lags = np.abs(np.arange(n_time)[:, None] - np.arange(n_time)[None, :]) * dt
    jitter = kernel.s if jitter is None else jitter
    prior = np.asarray(kernel.kernel(jnp.asarray(lags))) + jitter * np.eye(n_time)
    chol = np.linalg.cholesky(prior)
    return np.stack(
        [chol @ rng.standard_normal((n_time, n_latents)) for _ in range(n_sequences)]
    )
