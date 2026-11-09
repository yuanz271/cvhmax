"""CUDA-backed JAX integration tests."""

import jax
import numpy as np
import pytest
from cvhmax.cvhm import CVHM
from cvhmax.hm import HidaMatern


def _gpu_available():
    try:
        return bool(jax.devices("gpu"))
    except RuntimeError:
        return False


def _cpu_available():
    try:
        return bool(jax.devices("cpu"))
    except RuntimeError:
        return False


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not _gpu_available(), reason="CUDA JAX backend unavailable"),
    pytest.mark.skipif(not _cpu_available(), reason="CPU JAX backend unavailable"),
]


def _fit_on_device(y, device, mask):
    """Fit the small integration model with all arrays on ``device``."""
    with jax.default_device(device):
        model = CVHM(
            n_components=1,
            dt=1.0,
            kernels=[HidaMatern(sigma=1.0, rho=2.0, omega=0.0, order=0)],
            observation="Gaussian",
            max_iter=2,
            cvi_iter=1,
        )
        model.fit(jax.device_put(y, device), jax.device_put(mask, device), random_state=0)
    return model.posterior


@pytest.mark.parametrize("partial", [False, True])
def test_cvhm_cpu_gpu_parity(partial):
    """A representative CVHM fit agrees between CPU and CUDA backends."""
    gpu = jax.devices("gpu")[0]
    cpu = jax.devices("cpu")[0]
    n_trials = 1
    n_time = 6
    n_features = 2
    rng = np.random.default_rng(0)
    y = rng.normal(size=(n_trials, n_time, n_features))
    mask = np.ones_like(y, dtype=bool)
    if partial:
        mask[0, ::2, 0] = False
        mask[0, 1::2, 1] = False

    cpu_posterior = _fit_on_device(y, cpu, mask)
    gpu_posterior = _fit_on_device(y, gpu, mask)
    cpu_means, cpu_covariances = cpu_posterior
    gpu_means, gpu_covariances = gpu_posterior

    assert all(device.platform == "cpu" for device in cpu_means.devices())
    assert all(device.platform == "cpu" for device in cpu_covariances.devices())
    assert all(device.platform == "gpu" for device in gpu_means.devices())
    assert all(device.platform == "gpu" for device in gpu_covariances.devices())
    assert cpu_means.shape == gpu_means.shape == (n_trials, n_time, 1)
    assert cpu_covariances.shape == gpu_covariances.shape == (
        n_trials,
        n_time,
        1,
        1,
    )
    np.testing.assert_allclose(
        np.asarray(cpu_means), np.asarray(gpu_means), rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(cpu_covariances),
        np.asarray(gpu_covariances),
        rtol=1e-5,
        atol=1e-6,
    )
    assert np.isfinite(np.asarray(gpu_means)).all()
    assert np.isfinite(np.asarray(gpu_covariances)).all()
