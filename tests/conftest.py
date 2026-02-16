"""Shared fixtures, markers, and sys.path hooks for cvhmax tests."""

import sys
from pathlib import Path

import numpy as np
import pytest

import jax
from jax import config

config.update("jax_enable_x64", True)

from cvhmax.hm import HidaMatern, sample_matern  # noqa: E402


# ---------------------------------------------------------------------------
# Reference code access: add hida_matern_gp_lvms/ to sys.path when available
# ---------------------------------------------------------------------------
_REF_ROOT = Path(__file__).resolve().parent.parent / "hida_matern_gp_lvms"

if _REF_ROOT.is_dir() and str(_REF_ROOT) not in sys.path:
    sys.path.insert(0, str(_REF_ROOT))


def _torch_and_ref_available():
    """Return True when both torch and the reference code are importable."""
    try:
        import torch  # noqa: F401
        from gp_kernels.stationary_hm_gp_kernels import (  # noqa: F401
            StationaryHidaMaternKernel,
        )

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------
def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run tests marked as slow (e.g. benchmarks).",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "parity: cross-implementation parity tests (require torch + ref code)",
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


# convenience skip decorator used in test_parity.py
requires_ref = pytest.mark.skipif(
    not _torch_and_ref_available(),
    reason="torch or hida_matern_gp_lvms reference code not available",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def dt():
    return 1.0


@pytest.fixture
def order0_kernel():
    return HidaMatern(sigma=1.0, rho=1.0, omega=0.0, order=0, s=0.0)


@pytest.fixture
def order0_kernel_osc():
    return HidaMatern(sigma=1.5, rho=2.0, omega=3.0, order=0, s=0.0)


@pytest.fixture
def linear_gaussian_data(rng, dt):
    """Synthetic linear-Gaussian data for 1 latent, 10 observations, T=200."""
    T = 200
    n_obs = 10
    sigma, rho = 1.0, 50.0

    x = sample_matern(T, dt, sigma, rho)  # (T,)
    x = np.asarray(x).reshape(T, 1)

    C = rng.standard_normal((n_obs, 1)) * 0.5
    d = rng.standard_normal(n_obs) * 0.1
    R_diag = np.full(n_obs, 0.25)
    R = np.diag(R_diag)

    noise = rng.standard_normal((T, n_obs)) * np.sqrt(R_diag)
    y = x @ C.T + d + noise
    y = y[np.newaxis, ...]  # (1, T, n_obs)

    return dict(y=y, x_true=x, C=C, d=d, R=R, T=T, n_obs=n_obs)
