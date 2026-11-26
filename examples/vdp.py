"""Van der Pol latent trajectories shared by the demos."""

import numpy as np


def vdp_rhs(t, state, mu):
    """Right-hand side of the Van der Pol ODE."""
    x, v = state
    dxdt = v
    dvdt = mu * (1 - x**2) * v - x
    return [dxdt, dvdt]


def simulate_vdp(n_sequences, T, dt, mu=1.0, noise_std=0.05, rng=None):
    """Integrate the Van der Pol SDE via Euler-Maruyama.

    At each time step, Gaussian state noise with standard deviation
    ``noise_std * sqrt(dt)`` is added to both state dimensions.

    Parameters
    ----------
    n_sequences : int
        Number of independent trajectories.
    T : int
        Number of time steps per sequence.
    dt : float
        Sampling interval.
    mu : float
        Nonlinearity parameter.
    noise_std : float
        Diffusion coefficient (state noise intensity).
    rng : numpy.random.Generator or None
        Random number generator.

    Returns
    -------
    x : ndarray, shape (n_sequences, T, 2)
        Latent trajectories (position, velocity).
    """
    if rng is None:
        rng = np.random.default_rng()

    sqrt_dt = np.sqrt(dt)
    trajectories = np.empty((n_sequences, T, 2))

    for sequence in range(n_sequences):
        state = np.array([2.0 + rng.normal(0, 0.1), 0.0 + rng.normal(0, 0.1)])
        for t in range(T):
            trajectories[sequence, t] = state
            drift = np.array(vdp_rhs(None, state, mu))
            state = state + drift * dt + noise_std * sqrt_dt * rng.standard_normal(2)

    return trajectories


def standardize(x):
    """Zero-mean, unit-variance standardisation over sequences and time.

    Parameters
    ----------
    x : ndarray, shape (n_sequences, T, D)

    Returns
    -------
    x_std : ndarray, same shape
    mean : ndarray, shape (D,)
    std : ndarray, shape (D,)
    """
    mean = x.reshape(-1, x.shape[-1]).mean(axis=0)
    std = x.reshape(-1, x.shape[-1]).std(axis=0)
    return (x - mean) / std, mean, std
