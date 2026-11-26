"""Seeded synthetic data for demonstrations and tests.

These generators take an explicit ``numpy.random.Generator`` so a seed fully
determines the data. They are not part of inference and are not exported from
the package root.
"""

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


def simulate_vdp(n_sequences, n_time, dt, rng, mu=1.0, noise_std=0.05):
    """Integrate the Van der Pol oscillator by Euler-Maruyama.

    Each step adds Gaussian state noise of standard deviation
    ``noise_std * sqrt(dt)`` to both coordinates. Trajectories start near
    ``(2, 0)`` with a small random offset.

    Parameters
    ----------
    n_sequences : int
        Number of independent trajectories.
    n_time : int
        Time steps per trajectory.
    dt : float
        Sampling interval.
    rng : numpy.random.Generator
        Source of randomness.
    mu : float, default=1.0
        Nonlinearity of the oscillator.
    noise_std : float, default=0.05
        State noise intensity.

    Returns
    -------
    ndarray, shape (n_sequences, n_time, 2)
        Position and velocity.
    """
    sqrt_dt = np.sqrt(dt)
    trajectories = np.empty((n_sequences, n_time, 2))
    for sequence in range(n_sequences):
        x, v = 2.0 + rng.normal(0, 0.1), rng.normal(0, 0.1)
        for t in range(n_time):
            trajectories[sequence, t] = (x, v)
            drift = np.array([v, mu * (1 - x**2) * v - x])
            x, v = np.array([x, v]) + drift * dt + noise_std * sqrt_dt * rng.standard_normal(2)
    return trajectories


def standardize(x):
    """Standardise to zero mean and unit variance over sequences and time.

    Parameters
    ----------
    x : ndarray, shape (n_sequences, n_time, D)

    Returns
    -------
    x_std : ndarray, same shape as ``x``
    mean : ndarray, shape (D,)
    std : ndarray, shape (D,)
    """
    flat = x.reshape(-1, x.shape[-1])
    mean, std = flat.mean(axis=0), flat.std(axis=0)
    return (x - mean) / std, mean, std
