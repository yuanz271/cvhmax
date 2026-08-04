import json
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from jax import tree_util

from cvhmax import hm
from cvhmax.hm import HidaMatern, Ks, make_Ks, matern, spectral_density
from cvhmax.utils import conjtrans, real_repr


@pytest.mark.parametrize("order,expected_nple", [(0, 1), (1, 2), (2, 3)])
def test_Ks(order, expected_nple):
    """Ks returns a (nple, nple) stationary covariance matrix."""
    spec = {"sigma": 1.0, "rho": 1.0, "omega": 0.0, "order": order}
    K = hm.Ks(spec, 1.0)
    assert K.shape == (expected_nple, expected_nple)


def test_make_Ks_closes_over_static_order_under_jit():
    kernel = make_Ks(1)
    jit_kernel = jax.jit(kernel)
    params = {"sigma": 1.2, "rho": 0.7, "omega": 2.0}
    expected = Ks({**params, "order": 1}, 0.3)
    npt.assert_allclose(jit_kernel(params, 0.3), expected)


@pytest.mark.parametrize(
    "order,expected",
    [
        (0, lambda x: jnp.exp(-x)),
        (1, lambda x: (1 + jnp.sqrt(3) * x) * jnp.exp(-jnp.sqrt(3) * x)),
        (
            2,
            lambda x: (1 + jnp.sqrt(5) * x + 5 * x**2 / 3)
            * jnp.exp(-jnp.sqrt(5) * x),
        ),
    ],
)
def test_matern_half_integer_closed_forms(order, expected):
    x = jnp.array(0.7)
    npt.assert_allclose(matern(x, rho=1.0, order=order), expected(x))


def test_HidaMatern_kernel_matches_scalar_covariance():
    model = HidaMatern(sigma=1.5, rho=0.8, omega=2.0, order=2, s=1e-2)
    tau = jnp.array([-0.4, 0.0, 0.7])
    expected = 1.5**2 * matern(tau, rho=0.8, order=2) * jnp.cos(2.0 * tau)
    npt.assert_allclose(model.kernel(tau), expected)


def test_HidaMatern_kernel_is_jitter_free_at_zero():
    model = HidaMatern(sigma=1.5, rho=0.8, order=2, s=1e-2)
    npt.assert_allclose(model.kernel(0.0), 1.5**2)


def test_Ks_is_raw_and_HidaMatern_K_has_instantaneous_jitter():
    params = {"sigma": 1.5, "rho": 0.8, "omega": 2.0, "order": 2}
    model = HidaMatern(**params, s=1e-3)
    raw_zero = np.asarray(hm.Ks(params, 0.0))
    raw_lag = np.asarray(hm.Ks(params, 0.4))
    stabilized_zero = np.asarray(model.K(0.0))
    stabilized_lag = np.asarray(model.K(0.4))
    npt.assert_allclose(
        stabilized_zero, raw_zero + 1e-3 * np.eye(model.nple), atol=1e-12
    )
    npt.assert_allclose(stabilized_lag, raw_lag, atol=1e-12)


def test_dynamics_jitter_is_explicit_and_effective():
    params = {"sigma": 1.0, "rho": 1.0, "omega": 0.0, "order": 2}
    raw = np.asarray(hm.Qf(params, 1e-3, jitter=0.0))
    stabilized = np.asarray(hm.Qf(params, 1e-3, jitter=1e-5))
    assert np.max(np.abs(raw - stabilized)) > 1e-10
    npt.assert_allclose(
        stabilized,
        np.asarray(HidaMatern(**params, s=1e-5).Qf(1e-3)),
        atol=1e-12,
    )


def test_dynamics_use_raw_positive_lag_covariance():
    params = {"sigma": 1.0, "rho": 1.0, "omega": 0.0, "order": 0}
    tau = 0.4
    K0 = np.asarray(hm.Ks(params, 0.0)) + 1e-3 * np.eye(1)
    Kt = np.asarray(hm.Ks(params, tau))
    expected = K0 - Kt @ np.linalg.solve(K0, Kt.conj().T)
    actual = np.asarray(hm.Qf(params, tau, jitter=1e-3))
    npt.assert_allclose(actual, expected, atol=1e-12)


def test_zero_step_dynamics_are_identity_and_noiseless():
    params = {"sigma": 1.0, "rho": 1.0, "omega": 0.0, "order": 1}
    K0 = np.asarray(hm.Ks(params, 0.0)) + 1e-3 * np.eye(2)
    npt.assert_allclose(hm.Af(params, 0.0, jitter=1e-3), np.eye(2), atol=1e-12)
    npt.assert_allclose(hm.Ab(params, 0.0, jitter=1e-3), np.eye(2), atol=1e-12)
    npt.assert_allclose(hm.Qf(params, 0.0, jitter=1e-3), 0.0, atol=1e-12)
    npt.assert_allclose(hm.Qb(params, 0.0, jitter=1e-3), 0.0, atol=1e-12)
    npt.assert_allclose(hm.Ks(params, 0.0), K0 - 1e-3 * np.eye(2), atol=1e-12)


def test_HidaMatern_kernel_matches_generator_real_part():
    from cvhmax.kernel_generator import make_kernel

    model = HidaMatern(sigma=1.5, rho=0.8, omega=2.0, order=3, s=1e-2)
    tau = jnp.array([-0.4, 0.0, 0.7])
    generated = make_kernel(4).get_base_kernel(
        tau, jnp.array(model.sigma), jnp.array(model.rho), jnp.array(model.omega)
    )
    npt.assert_allclose(model.kernel(tau), generated)


def test_matern_rejects_invalid_parameters():
    with pytest.raises(ValueError, match="non-negative"):
        matern(0.0, rho=1.0, order=-1)
    with pytest.raises(ValueError, match="positive"):
        matern(0.0, rho=0.0, order=0)
    with pytest.raises(TypeError, match="static integer"):
        HidaMatern(order=1.5)
    with pytest.raises(ValueError, match="positive"):
        HidaMatern(rho=0.0)


def test_ssm_repr():
    dt = 1.0
    kernelparams = [
        [{"sigma": 1.0, "rho": 1.0, "omega": 0.0, "order": 1}],
        [
            {"sigma": 1.0, "rho": 1.0, "omega": 0.0, "order": 0},
            {"sigma": 1.0, "rho": 1.0, "omega": 1.0, "order": 1},
        ],
    ]
    Af, _, _, _ = hm.ssm_repr(kernelparams, dt)
    paramflat, _ = tree_util.tree_flatten(Af)
    assert len(paramflat) == 3


# ---------------------------------------------------------------------------
# New tests
# ---------------------------------------------------------------------------

_PARAM_GRID = [
    (1.0, 1.0, 0.0),
    (1.5, 2.0, 3.0),
    (0.5, 0.3, 10.0),
]


class TestKernelProperties:
    """Mathematical property tests for the HidaMatern kernel."""

    @pytest.mark.parametrize("sigma,rho,omega", _PARAM_GRID)
    def test_K0_diagonal_positive(self, sigma, rho, omega):
        """K(0) must have positive real diagonal entries."""
        k = HidaMatern(sigma=sigma, rho=rho, omega=omega, order=0, s=0.0)
        K0 = k.K(0.0)
        diag = jnp.diag(K0).real
        assert jnp.all(diag > 0), f"K(0) diagonal not positive: {diag}"

    @pytest.mark.parametrize("order", [0, 1])
    def test_nple_equals_order_plus_one(self, order):
        """nple property should return order + 1."""
        k = HidaMatern(order=order)
        assert k.nple == order + 1

    @pytest.mark.parametrize("dt_val", [0.1, 1.0, 5.0])
    @pytest.mark.parametrize("sigma,rho,omega", _PARAM_GRID)
    def test_Af_Qf_lyapunov_order0(self, dt_val, sigma, rho, omega):
        """Stationarity: A @ K(0) @ A^H + Q == K(0) for the forward model."""
        k = HidaMatern(sigma=sigma, rho=rho, omega=omega, order=0, s=0.0)
        K0 = k.K(0.0)
        A = k.Af(dt_val)
        Q = k.Qf(dt_val)
        reconstructed = A @ K0 @ conjtrans(A) + Q
        npt.assert_allclose(
            np.asarray(reconstructed),
            np.asarray(K0),
            atol=1e-6,
            rtol=5e-7,
            err_msg=f"Lyapunov (forward) failed for dt={dt_val}, sigma={sigma}, rho={rho}, omega={omega}",
        )

    @pytest.mark.parametrize("dt_val", [0.1, 1.0, 5.0])
    @pytest.mark.parametrize("sigma,rho,omega", _PARAM_GRID)
    def test_Ab_Qb_lyapunov_order0(self, dt_val, sigma, rho, omega):
        """Stationarity: Ab @ K(0) @ Ab^H + Qb == K(0) for the backward model."""
        k = HidaMatern(sigma=sigma, rho=rho, omega=omega, order=0, s=0.0)
        K0 = k.K(0.0)
        Ab = k.Ab(dt_val)
        Qb = k.Qb(dt_val)
        reconstructed = Ab @ K0 @ conjtrans(Ab) + Qb
        npt.assert_allclose(
            np.asarray(reconstructed),
            np.asarray(K0),
            atol=1e-6,
            rtol=5e-7,
            err_msg=f"Lyapunov (backward) failed for dt={dt_val}, sigma={sigma}, rho={rho}, omega={omega}",
        )

    @pytest.mark.parametrize("sigma,rho,omega", _PARAM_GRID)
    def test_Qf_positive_semidefinite_order0(self, sigma, rho, omega):
        """Qf(dt) in real form must be positive semidefinite."""
        k = HidaMatern(sigma=sigma, rho=rho, omega=omega, order=0, s=0.0)
        Q_complex = k.Qf(1.0)
        Q_real = real_repr(Q_complex)
        Q_real = 0.5 * (Q_real + Q_real.T)  # ensure exactly symmetric
        eigvals = jnp.linalg.eigvalsh(Q_real)
        assert jnp.all(eigvals >= -1e-10), f"Qf not PSD: eigenvalues = {eigvals}"

    def test_Af_identity_small_dt(self, order0_kernel):
        """For very small dt, Af should approach the identity matrix."""
        A_complex = order0_kernel.Af(1e-6)
        I_complex = jnp.eye(order0_kernel.nple, dtype=A_complex.dtype)
        npt.assert_allclose(np.asarray(A_complex), np.asarray(I_complex), atol=1e-4)

    def test_spectral_density_integral(self):
        """Integral of the PSD over all frequencies should approximate 2*sigma^2.

        cvhmax uses ``c * (s_pos + s_neg)`` without the ``0.5`` factor present
        in the reference implementation, so the integral is ``2 * sigma^2``
        rather than ``sigma^2``.
        """
        sigma, rho = 2.0, 1.0
        spec = {"sigma": sigma, "rho": rho, "omega": 0.0, "order": 0}
        freq = jnp.linspace(-50, 50, 100_000)
        psd = spectral_density(spec, freq)
        integral = float(jnp.trapezoid(psd, freq))
        npt.assert_allclose(integral, 2 * sigma**2, rtol=0.05)


def _pack_complex(array: np.ndarray) -> list:
    stacked = np.stack([array.real, array.imag], axis=-1)
    return stacked.tolist()


def _unpack_complex(payload: list) -> np.ndarray:
    stacked = np.asarray(payload)
    return stacked[..., 0] + 1j * stacked[..., 1]


def _run_kernel_script(enable_x64: bool, dtype: str) -> dict:
    script = """
import json
import os

import jax.numpy as jnp
import numpy as np

from cvhmax.hm import HidaMatern


def pack(arr):
    arr = np.asarray(arr)
    return np.stack([arr.real, arr.imag], axis=-1).tolist()


dtype = jnp.float32 if os.environ["KERNEL_DTYPE"] == "float32" else jnp.float64

sigma = jnp.asarray(1.0, dtype=dtype)
rho = jnp.asarray(1.0, dtype=dtype)
omega = jnp.asarray(0.0, dtype=dtype)
order = 2
s = jnp.asarray(1e-8, dtype=dtype)

tau = jnp.asarray(1e-3, dtype=dtype)

hm = HidaMatern(sigma=sigma, rho=rho, omega=omega, order=order, s=s)
K0 = hm.K(0.0)
A = hm.Af(tau)
Q = hm.Qf(tau)

payload = {"K0": pack(K0), "A": pack(A), "Q": pack(Q)}
print(json.dumps(payload))
"""
    env = os.environ.copy()
    env["JAX_ENABLE_X64"] = "1" if enable_x64 else "0"
    env["KERNEL_DTYPE"] = dtype
    result = subprocess.check_output([sys.executable, "-c", script], env=env)
    return json.loads(result)


def test_kernel_precision_parity_x64_toggle():
    payload_f32 = _run_kernel_script(enable_x64=False, dtype="float32")
    payload_f64 = _run_kernel_script(enable_x64=True, dtype="float32")

    for key in ("K0", "A", "Q"):
        arr_f32 = _unpack_complex(payload_f32[key])
        arr_f64 = _unpack_complex(payload_f64[key])
        # Float32 covariance subtraction at small lags is roundoff-limited;
        # the x32 path is checked for finite, bounded output rather than x64
        # ulp-level parity.
        npt.assert_allclose(arr_f32, arr_f64, rtol=5e-4, atol=1e-5)
        assert np.all(np.isfinite(arr_f32))


def test_kernel_precision_parity_inputs():
    if jnp.asarray(1.0).dtype != jnp.float64:
        pytest.skip("Requires x64 to compare float32 and float64 inputs")

    sigma32 = jnp.asarray(1.0, dtype=jnp.float32)
    rho32 = jnp.asarray(1.0, dtype=jnp.float32)
    omega32 = jnp.asarray(0.0, dtype=jnp.float32)
    s32 = jnp.asarray(1e-8, dtype=jnp.float32)

    sigma64 = jnp.asarray(1.0, dtype=jnp.float64)
    rho64 = jnp.asarray(1.0, dtype=jnp.float64)
    omega64 = jnp.asarray(0.0, dtype=jnp.float64)
    s64 = jnp.asarray(1e-8, dtype=jnp.float64)

    tau32 = jnp.asarray(1e-3, dtype=jnp.float32)
    tau64 = jnp.asarray(1e-3, dtype=jnp.float64)

    hm32 = HidaMatern(sigma=sigma32, rho=rho32, omega=omega32, order=2, s=s32)
    hm64 = HidaMatern(sigma=sigma64, rho=rho64, omega=omega64, order=2, s=s64)

    K0_32 = hm32.K(0.0)
    K0_64 = hm64.K(0.0)
    A_32 = hm32.Af(tau32)
    A_64 = hm64.Af(tau64)
    Q_32 = hm32.Qf(tau32)
    Q_64 = hm64.Qf(tau64)

    npt.assert_allclose(K0_32, K0_64.astype(K0_32.dtype), rtol=5e-4, atol=2e-6)
    npt.assert_allclose(A_32, A_64.astype(A_32.dtype), rtol=5e-4, atol=2e-6)
    npt.assert_allclose(Q_32, Q_64.astype(Q_32.dtype), rtol=5e-4, atol=2e-6)
