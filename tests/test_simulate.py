"""Seeded synthetic-data generators."""

import numpy as np
import numpy.testing as npt
from jax import numpy as jnp

from cvhmax.hm import HidaMatern
from cvhmax.simulate import sample_gp_latents, simulate_vdp, standardize


def test_gp_latents_follow_the_kernel_on_the_time_grid():
    """Across many sequences the sample covariance is the kernel at lags scaled by dt."""
    kernel = HidaMatern(sigma=1.0, rho=1.0, omega=0.0, order=1, s=1e-5)
    n_time, dt = 4, 0.5
    latents = sample_gp_latents(
        kernel, n_time, dt, 2, np.random.default_rng(0), n_sequences=4000
    )
    assert latents.shape == (4000, n_time, 2)

    lags = np.abs(np.arange(n_time)[:, None] - np.arange(n_time)[None, :]) * dt
    expected = np.asarray(kernel.kernel(jnp.asarray(lags))) + kernel.s * np.eye(n_time)
    # The standard error at 4000 sequences is about 0.03; a wrong dt scaling is off by 0.3.
    npt.assert_allclose(np.cov(latents[:, :, 0].T), expected, atol=0.1)


def test_gp_latents_are_independent_across_latents_and_jitter_adds_variance():
    kernel = HidaMatern(sigma=1.0, rho=1.0, omega=0.0, order=1, s=1e-5)
    latents = sample_gp_latents(
        kernel, 3, 0.5, 2, np.random.default_rng(1), n_sequences=4000, jitter=1.0
    )
    assert abs(np.corrcoef(latents[:, 0, 0], latents[:, 0, 1])[0, 1]) < 0.1
    # The kernel's own variance is 1, so unit jitter doubles the marginal variance.
    npt.assert_allclose(latents[:, 0, 0].var(), 2.0, rtol=0.1)


def test_gp_latents_are_reproducible_from_the_seed():
    kernel = HidaMatern(order=0)
    first = sample_gp_latents(kernel, 5, 1.0, 2, np.random.default_rng(3), n_sequences=2)
    again = sample_gp_latents(kernel, 5, 1.0, 2, np.random.default_rng(3), n_sequences=2)
    npt.assert_array_equal(first, again)


def test_vdp_without_noise_is_the_euler_step_of_the_oscillator():
    """With ``noise_std=0`` each state is exactly the previous plus ``dt * drift``."""
    dt, mu = 0.02, 1.5
    path = simulate_vdp(3, 50, dt, np.random.default_rng(0), mu=mu, noise_std=0.0)
    x, v = path[:, :-1, 0], path[:, :-1, 1]
    drift = np.stack([v, mu * (1 - x**2) * v - x], axis=-1)
    npt.assert_allclose(path[:, 1:], path[:, :-1] + dt * drift, atol=1e-12)


def test_vdp_state_noise_scales_with_the_square_root_of_dt():
    """The residual of one Euler step has standard deviation ``noise_std * sqrt(dt)``."""
    dt, noise_std = 0.02, 0.2
    path = simulate_vdp(1, 8000, dt, np.random.default_rng(1), noise_std=noise_std)[0]
    x, v = path[:-1, 0], path[:-1, 1]
    drift = np.stack([v, (1 - x**2) * v - x], axis=-1)
    residual = path[1:] - path[:-1] - dt * drift
    npt.assert_allclose(residual.std(axis=0), noise_std * np.sqrt(dt), rtol=0.1)


def test_standardize_returns_the_statistics_that_invert_it():
    x = np.random.default_rng(2).normal(loc=3.0, scale=5.0, size=(4, 30, 2))
    standardized, mean, std = standardize(x)
    flat = standardized.reshape(-1, 2)
    npt.assert_allclose(flat.mean(axis=0), 0.0, atol=1e-12)
    npt.assert_allclose(flat.std(axis=0), 1.0, atol=1e-12)
    npt.assert_allclose(standardized * std + mean, x, atol=1e-12)
