"""CUDA-backed JAX integration tests."""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from cvhmax.cvhm import CVHM
from cvhmax.hm import HidaMatern


def _gpu_available():
    try:
        return any(device.platform == "gpu" for device in jax.devices())
    except RuntimeError:
        return False


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not _gpu_available(), reason="CUDA JAX backend unavailable"),
]


def test_cvhm_gaussian_fit_on_cuda():
    """A representative CVHM fit compiles and executes on CUDA."""
    assert jax.default_backend() == "gpu"

    n_trials = 1
    n_time = 6
    n_features = 2
    rng = np.random.default_rng(0)
    y = jnp.asarray(rng.normal(size=(n_trials, n_time, n_features)))

    model = CVHM(
        n_components=1,
        dt=1.0,
        kernels=[HidaMatern(sigma=1.0, rho=2.0, omega=0.0, order=0)],
        observation="Gaussian",
        max_iter=1,
        cvi_iter=1,
    )
    model.fit(y, random_state=0)
    means, covariances = model.posterior

    assert means.shape == (n_trials, n_time, 1)
    assert covariances.shape == (n_trials, n_time, 1, 1)
    assert all(device.platform == "gpu" for device in means.devices())
    assert all(device.platform == "gpu" for device in covariances.devices())
    assert np.isfinite(np.asarray(means)).all()
    assert np.isfinite(np.asarray(covariances)).all()
