"""Benchmark tests: regression guard for inference quality.

These tests reproduce the three demo scenarios from ``examples/demo_vdp.py``
and assert that R² stays above a hard floor.  They are marked ``slow``
(~40 s each) and skipped by default — run with::

    pytest tests/test_benchmark.py --run-slow

Baseline R² (recorded on the ``refactor/latent-space-cvi`` branch):

    Frozen readout:       0.972
    Estimated readout:    0.972
    Variable-length:      0.972

The hard floor is set to 0.96 to allow minor platform/JAX-version jitter.
"""

import numpy as np
import pytest

from jax import numpy as jnp

from cvhmax.cvhm import CVHM
from cvhmax.cvi import Params
from cvhmax.hm import HidaMatern
from cvhmax.utils import pad_trials, unpad_trials


# ---------------------------------------------------------------------------
# Shared constants & helpers
# ---------------------------------------------------------------------------

R2_FLOOR = 0.96

N_TRIALS = 10
T = 500
DT = 0.02
MU = 1.0
N_OBS = 50
N_LATENTS = 2
BASELINE_RATE = 5.0


def _vdp_rhs(state, mu):
    x, v = state
    return np.array([v, mu * (1 - x**2) * v - x])


def _simulate_vdp(rng, n_trials, T, dt, mu, noise_std=0.05):
    sqrt_dt = np.sqrt(dt)
    out = np.empty((n_trials, T, 2))
    for trial in range(n_trials):
        s = np.array([2.0 + rng.normal(0, 0.1), rng.normal(0, 0.1)])
        for t in range(T):
            out[trial, t] = s
            s = s + _vdp_rhs(s, mu) * dt + noise_std * sqrt_dt * rng.standard_normal(2)
    return out


def _standardize(x):
    flat = x.reshape(-1, x.shape[-1])
    mu, sigma = flat.mean(0), flat.std(0)
    return (x - mu) / sigma


def _r2(m, x_true):
    """Pooled R² after affine alignment."""
    K = m.shape[-1]
    mf = m.reshape(-1, K)
    xf = x_true.reshape(-1, K)
    A = np.hstack([mf, np.ones((mf.shape[0], 1))])
    W, _, _, _ = np.linalg.lstsq(A, xf, rcond=None)
    aligned = A @ W
    ss_res = np.sum((aligned - xf) ** 2)
    ss_tot = np.sum((xf - xf.mean(0)) ** 2)
    return 1.0 - ss_res / ss_tot


def _make_data(rng):
    """Simulate VdP + Poisson observations (deterministic given rng)."""
    x_raw = _simulate_vdp(rng, N_TRIALS, T, DT, MU)
    x_std = _standardize(x_raw)

    C_true = rng.standard_normal((N_OBS, N_LATENTS))
    C_true /= np.linalg.norm(C_true, axis=0, keepdims=True)
    d_true = np.full(N_OBS, np.log(BASELINE_RATE))

    log_rate = np.einsum("ntl,ol->nto", x_std, C_true) + d_true[None, None, :]
    y_np = rng.poisson(np.exp(log_rate))
    y = jnp.asarray(y_np, dtype=jnp.float64)

    return y, y_np, x_std, C_true, d_true


def _kernels():
    return [HidaMatern(sigma=1.0, rho=1.0, omega=0.0, order=1) for _ in range(N_LATENTS)]


# ---------------------------------------------------------------------------
# Frozen-readout Poisson subclass (same as demo)
# ---------------------------------------------------------------------------

from cvhmax.cvi import Poisson  # noqa: E402


class FrozenPoisson(Poisson):
    @classmethod
    def update_readout(cls, params, y, valid_y, m, V):
        return params, jnp.nan


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_benchmark_frozen_readout():
    rng = np.random.default_rng(2024)
    y, _, x_std, C_true, d_true = _make_data(rng)
    kernels = _kernels()

    true_params = Params(C=jnp.asarray(C_true), d=jnp.asarray(d_true), R=None)

    model = CVHM(
        n_components=N_LATENTS, dt=DT, kernels=kernels,
        observation="FrozenPoisson", max_iter=50, cvi_iter=5,
    )
    model.params = true_params
    model.fit(y, random_state=42)

    r2 = _r2(np.asarray(model.posterior[0]), x_std)
    assert r2 >= R2_FLOOR, f"Frozen readout R²={r2:.4f} < {R2_FLOOR}"


@pytest.mark.slow
def test_benchmark_estimated_readout():
    rng = np.random.default_rng(2024)
    y, _, x_std, _, _ = _make_data(rng)
    kernels = _kernels()

    model = CVHM(
        n_components=N_LATENTS, dt=DT, kernels=kernels,
        observation="Poisson", max_iter=50, cvi_iter=5,
    )
    model.fit(y, random_state=42)

    r2 = _r2(np.asarray(model.posterior[0]), x_std)
    assert r2 >= R2_FLOOR, f"Estimated readout R²={r2:.4f} < {R2_FLOOR}"


@pytest.mark.slow
def test_benchmark_variable_length():
    rng = np.random.default_rng(2024)
    y, y_np, x_std, _, _ = _make_data(rng)
    kernels = _kernels()

    trial_lengths_np = rng.integers(250, 501, size=N_TRIALS)
    y_list = [
        jnp.asarray(y_np[i, :tl], dtype=jnp.float64)
        for i, tl in enumerate(trial_lengths_np)
    ]
    x_list = [x_std[i, :tl] for i, tl in enumerate(trial_lengths_np)]

    y_padded, valid_y, trial_lengths = pad_trials(y_list)

    model = CVHM(
        n_components=N_LATENTS, dt=DT, kernels=kernels,
        observation="Poisson", max_iter=50, cvi_iter=5,
    )
    model.fit(y_padded, valid_y=valid_y, random_state=42)

    m_list = unpad_trials(model.posterior[0], trial_lengths)
    m_all = np.concatenate([np.asarray(mi) for mi in m_list])
    x_all = np.concatenate(x_list)

    r2 = _r2(m_all, x_all)
    assert r2 >= R2_FLOOR, f"Variable-length R²={r2:.4f} < {R2_FLOOR}"
