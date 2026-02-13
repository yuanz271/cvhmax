from jax import tree_util
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

# import equinox as eqx
import pytest

from cvhmax import hm, utils
from cvhmax.hm import HidaMatern, Ks0, spectral_density
from cvhmax.utils import real_repr, conjtrans


def test_HidaMatern():
    kernel = hm.HidaMatern(1.0, 1.0, 0.0, 0)

    K0 = kernel.K(0.0)
    assert K0.shape == (1, 1)

    dt = 1.0

    kernel.Af(dt)
    kernel.Qf(dt)
    kernel.Ab(dt)
    kernel.Qb(dt)


# def test_composite():
#     # 2 latents
#     # L1: 1 kernel
#     # L2: 2 kernels
#     hyperparams = [[{'sigma': 1., 'rho': 1., 'omega': 0., 'order': 1}], [{'sigma': 1., 'rho': 1., 'omega': 0., 'order': 0}, {'sigma': 1., 'rho': 1., 'omega': 1., 'order': 1}]]
#     hyperspec = [[{'sigma': True, 'rho': True, 'omega': True, 'order': False}], [{'sigma': True, 'rho': True, 'omega': True, 'order': False}, {'sigma': True, 'rho': True, 'omega': True, 'order': False}]]
#     # print(tree_util.tree_structure(hyperparams))
#     hyperdef, hyperflat = tree_util.tree_flatten(hyperparams)
#     # print(hyperflat)
#     # https://docs.kidger.site/equinox/all-of-equinox/
#     # eqx.partition

#     params, static = eqx.partition(hyperparams, hyperspec)
#     # eqx.tree_pprint(params)
#     # eqx.tree_pprint(static)
#     paramflat, paramdef = tree_util.tree_flatten(params)
#     # print(paramdef)
#     # print(paramflat)


def test_Ks():
    kernelparam = {"sigma": 1.0, "rho": 1.0, "omega": 0.0, "order": 0}
    hm.Ks(kernelparam, 1.0)

    kernelparam = {"sigma": 1.0, "rho": 1.0, "omega": 0.0, "order": 1}
    hm.Ks(kernelparam, 1.0)

    kernelparam = {"sigma": 1.0, "rho": 1.0, "omega": 0.0, "order": 2}
    K = hm.Ks(kernelparam, 1.0)
    assert K.shape == (3, 3)  # order 2 -> nple = 3


def test_ssm_repr():
    dt = 1.0
    kernelparams = [
        [{"sigma": 1.0, "rho": 1.0, "omega": 0.0, "order": 1}],
        [
            {"sigma": 1.0, "rho": 1.0, "omega": 0.0, "order": 0},
            {"sigma": 1.0, "rho": 1.0, "omega": 1.0, "order": 1},
        ],
    ]
    Af, Qf, Ab, Qb = hm.ssm_repr(kernelparams, dt)
    # eqx.tree_pprint(Af)
    paramflat, paramdef = tree_util.tree_flatten(Af)
    assert len(paramflat) == 3


def test_mask():
    kernelparams = [
        [{"sigma": 1.0, "rho": 1.0, "omega": 0.0, "order": 1}],
        [
            {"sigma": 1.0, "rho": 1.0, "omega": 0.0, "order": 0},
            {"sigma": 1.0, "rho": 1.0, "omega": 1.0, "order": 1},
        ],
    ]
    M = utils.latent_mask(kernelparams)
    print(M)


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
            atol=1e-8,
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
            atol=1e-8,
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
