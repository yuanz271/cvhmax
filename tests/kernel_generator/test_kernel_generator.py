"""Tests for the kernel_generator subpackage.

Requires the ``kergen`` extra (sympy, sympy2jax).
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

pytest.importorskip("sympy", reason="kergen extra not installed")
pytest.importorskip("sympy2jax", reason="kergen extra not installed")

from cvhmax.kernel_generator import HidaMaternKernelGenerator, make_kernel  # noqa: E402
from cvhmax.kernel_generator.matern import matern_poly, hida_matern_kernel  # noqa: E402
from cvhmax.hm import HidaMatern, Ks0, Ks1, Ks  # noqa: E402
from cvhmax.utils import real_repr, conjtrans  # noqa: E402


# ---------------------------------------------------------------------------
# Reference code availability check
# ---------------------------------------------------------------------------
_REF_ROOT = Path(__file__).resolve().parent.parent / "hida_matern_gp_lvms"
if _REF_ROOT.is_dir() and str(_REF_ROOT) not in sys.path:
    sys.path.insert(0, str(_REF_ROOT))


def _torch_and_ref_available():
    try:
        import torch  # noqa: F401
        from hida_matern_kernel_generator.hm_kernel_generator import (  # noqa: F401
            HidaMaternKernelGenerator as _,
        )

        return True
    except ImportError:
        return False


requires_ref = pytest.mark.skipif(
    not _torch_and_ref_available(),
    reason="torch or hida_matern_gp_lvms reference code not available",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ORDERS = [1, 2, 3, 4, 5]

_PARAM_GRID = [
    (1.0, 1.0, 0.0),
    (1.5, 2.0, 3.0),
    (0.5, 0.3, 10.0),
]


def _matern_closed_form(order, tau, sigma, rho):
    r = jnp.abs(tau) / rho
    if order == 1:
        return sigma**2 * jnp.exp(-r)
    if order == 2:
        return sigma**2 * (1.0 + jnp.sqrt(3.0) * r) * jnp.exp(-jnp.sqrt(3.0) * r)
    if order == 3:
        return (
            sigma**2
            * (1.0 + jnp.sqrt(5.0) * r + (5.0 / 3.0) * r**2)
            * jnp.exp(-jnp.sqrt(5.0) * r)
        )
    raise ValueError(f"Unsupported order {order} for closed-form Matérn")


def _complex_kernel(order, tau, sigma, rho, omega):
    base = _matern_closed_form(order, tau, sigma, rho)
    return base * jnp.exp(1j * omega * tau)


def _finite_diff_first(f, tau, h):
    return (f(tau + h) - f(tau - h)) / (2.0 * h)


def _finite_diff_second(f, tau, h):
    return (f(tau + h) - 2.0 * f(tau) + f(tau - h)) / (h**2)


@pytest.fixture(params=_ORDERS)
def gen(request):
    """Kernel generator for each test order."""
    return make_kernel(request.param)


# ---------------------------------------------------------------------------
# matern.py tests
# ---------------------------------------------------------------------------


class TestMaternSymbolic:
    """Tests for the symbolic Matern construction."""

    def test_matern_poly_order0(self):
        """Matern-1/2 should reduce to exp(-tau/rho)."""
        import sympy as sym

        tau = sym.Symbol("tau", positive=True)
        rho = sym.Symbol("rho", positive=True)
        k = matern_poly(0, tau, rho)
        # At tau=0, k=1
        assert k.subs(tau, 0) == 1

    def test_matern_poly_order1(self):
        """Matern-3/2 at tau=0 should be 1."""
        import sympy as sym

        tau = sym.Symbol("tau", positive=True)
        rho = sym.Symbol("rho", positive=True)
        k = matern_poly(1, tau, rho)
        assert k.subs(tau, 0) == 1

    def test_matern_poly_negative_order(self):
        """Negative order p should raise ValueError."""
        import sympy as sym

        tau = sym.Symbol("tau", positive=True)
        rho = sym.Symbol("rho", positive=True)
        with pytest.raises(ValueError, match="p must be >= 0"):
            matern_poly(-1, tau, rho)

    def test_hida_matern_kernel_order0(self):
        """Order < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="order must be >= 1"):
            hida_matern_kernel(0)

    def test_hida_matern_kernel_returns_symbols(self):
        """Should return 4 symbols: tau, sigma, rho, omega."""
        _, symbols = hida_matern_kernel(2)
        assert len(symbols) == 4


# ---------------------------------------------------------------------------
# Generator construction tests
# ---------------------------------------------------------------------------


class TestGeneratorConstruction:
    """Tests for HidaMaternKernelGenerator instantiation."""

    def test_invalid_order(self):
        with pytest.raises(ValueError, match="order must be >= 1"):
            HidaMaternKernelGenerator(0)

    def test_order_attribute(self):
        gen = HidaMaternKernelGenerator(3)
        assert gen.order == 3

    def test_make_kernel_caching(self):
        """make_kernel should return the same object for the same order."""
        g1 = make_kernel(2)
        g2 = make_kernel(2)
        assert g1 is g2

    def test_arbitrary_high_order(self):
        """Should handle high orders without error."""
        gen = make_kernel(8)
        K = gen.create_K_hat(
            jnp.array(0.5), jnp.array(1.0), jnp.array(1.0), jnp.array(0.0)
        )
        assert K.shape == (8, 8)


# ---------------------------------------------------------------------------
# create_K_hat tests
# ---------------------------------------------------------------------------


class TestCreateKHat:
    """Tests for the K_hat matrix computation."""

    def test_shape(self, gen):
        """K_hat should be (M, M) complex."""
        K = gen.create_K_hat(
            jnp.array(0.5), jnp.array(1.0), jnp.array(1.0), jnp.array(0.0)
        )
        M = gen.order
        assert K.shape == (M, M)
        assert jnp.iscomplexobj(K)

    @pytest.mark.parametrize("sigma,rho,omega", _PARAM_GRID)
    def test_K0_hermitian(self, gen, sigma, rho, omega):
        """K(0) must be Hermitian."""
        K0 = gen.create_K_hat(
            jnp.array(0.0),
            jnp.array(sigma),
            jnp.array(rho),
            jnp.array(omega),
        )
        npt.assert_allclose(
            np.asarray(K0),
            np.asarray(conjtrans(K0)),
            atol=1e-12,
            err_msg=f"K(0) not Hermitian for order={gen.order}",
        )

    @pytest.mark.parametrize("sigma,rho,omega", _PARAM_GRID)
    def test_K0_positive_definite(self, gen, sigma, rho, omega):
        """K(0) in real representation should be positive definite."""
        K0 = gen.create_K_hat(
            jnp.array(0.0),
            jnp.array(sigma),
            jnp.array(rho),
            jnp.array(omega),
        )
        K0_real = real_repr(K0)
        K0_real = 0.5 * (K0_real + K0_real.T)
        eigvals = jnp.linalg.eigvalsh(K0_real)
        assert jnp.all(eigvals > -1e-10), (
            f"K(0) not PSD for order={gen.order}: eigenvalues = {eigvals}"
        )

    @pytest.mark.parametrize("sigma,rho,omega", _PARAM_GRID)
    def test_K0_diagonal_positive(self, gen, sigma, rho, omega):
        """K(0) diagonal entries must have positive real part."""
        K0 = gen.create_K_hat(
            jnp.array(0.0),
            jnp.array(sigma),
            jnp.array(rho),
            jnp.array(omega),
        )
        diag = jnp.diag(K0).real
        assert jnp.all(diag > 0), f"K(0) diagonal not positive: {diag}"


# ---------------------------------------------------------------------------
# Lyapunov / SSM property tests
# ---------------------------------------------------------------------------


class TestSSMProperties:
    """Stationarity and SSM property tests for generated kernels."""

    @pytest.mark.parametrize("dt_val", [0.1, 1.0, 5.0])
    @pytest.mark.parametrize("sigma,rho,omega", _PARAM_GRID)
    def test_forward_lyapunov(self, gen, dt_val, sigma, rho, omega):
        """A @ K(0) @ A^H + Q == K(0) for the forward model."""
        s = jnp.array(sigma)
        r = jnp.array(rho)
        o = jnp.array(omega)

        K0 = gen.create_K_hat(jnp.array(0.0), s, r, o)
        Kt = gen.create_K_hat(jnp.array(dt_val), s, r, o)

        A = conjtrans(jnp.linalg.solve(conjtrans(K0), conjtrans(Kt)))
        Q = K0 - Kt @ jnp.linalg.solve(K0, conjtrans(Kt))

        reconstructed = A @ K0 @ conjtrans(A) + Q
        npt.assert_allclose(
            np.asarray(reconstructed),
            np.asarray(K0),
            atol=1e-7,
            err_msg=(
                f"Lyapunov (forward) failed for order={gen.order}, "
                f"dt={dt_val}, sigma={sigma}, rho={rho}, omega={omega}"
            ),
        )

    @pytest.mark.parametrize("dt_val", [0.1, 1.0, 5.0])
    @pytest.mark.parametrize("sigma,rho,omega", _PARAM_GRID)
    def test_backward_lyapunov(self, gen, dt_val, sigma, rho, omega):
        """Ab @ K(0) @ Ab^H + Qb == K(0) for the backward model."""
        s = jnp.array(sigma)
        r = jnp.array(rho)
        o = jnp.array(omega)

        K0 = gen.create_K_hat(jnp.array(0.0), s, r, o)
        Kt = gen.create_K_hat(jnp.array(dt_val), s, r, o)

        Ab = conjtrans(jnp.linalg.solve(conjtrans(K0), Kt))
        Qb = K0 - conjtrans(Kt) @ jnp.linalg.solve(K0, Kt)

        reconstructed = Ab @ K0 @ conjtrans(Ab) + Qb
        npt.assert_allclose(
            np.asarray(reconstructed),
            np.asarray(K0),
            atol=1e-7,
            err_msg=(
                f"Lyapunov (backward) failed for order={gen.order}, "
                f"dt={dt_val}, sigma={sigma}, rho={rho}, omega={omega}"
            ),
        )

    @pytest.mark.parametrize("sigma,rho,omega", _PARAM_GRID)
    def test_Qf_positive_semidefinite(self, gen, sigma, rho, omega):
        """Qf(dt) in real form must be PSD."""
        s = jnp.array(sigma)
        r = jnp.array(rho)
        o = jnp.array(omega)

        K0 = gen.create_K_hat(jnp.array(0.0), s, r, o)
        Kt = gen.create_K_hat(jnp.array(1.0), s, r, o)

        Q = K0 - Kt @ jnp.linalg.solve(K0, conjtrans(Kt))
        Q_real = real_repr(Q)
        Q_real = 0.5 * (Q_real + Q_real.T)
        eigvals = jnp.linalg.eigvalsh(Q_real)
        assert jnp.all(eigvals >= -1e-8), (
            f"Qf not PSD for order={gen.order}: eigenvalues = {eigvals}"
        )


# ---------------------------------------------------------------------------
# Parity with existing hand-coded kernels
# ---------------------------------------------------------------------------


class TestParityHandCoded:
    """Verify generated kernels match existing Ks0 and Ks1."""

    @pytest.mark.parametrize("sigma,rho,omega", _PARAM_GRID)
    @pytest.mark.parametrize("tau_val", [0.0, 0.1, 0.5, 1.0, 5.0])
    def test_parity_order1_Ks0(self, sigma, rho, omega, tau_val):
        """Generated order-1 kernel must match Ks0."""
        gen = make_kernel(1)
        K_gen = gen.create_K_hat(
            jnp.array(tau_val),
            jnp.array(sigma),
            jnp.array(rho),
            jnp.array(omega),
        )
        K_ref = Ks0(tau_val, sigma, rho, omega)
        npt.assert_allclose(np.asarray(K_gen), np.asarray(K_ref), atol=1e-12)

    @pytest.mark.parametrize("sigma,rho,omega", _PARAM_GRID)
    @pytest.mark.parametrize("tau_val", [0.0, 0.1, 0.5, 1.0, 5.0])
    def test_parity_order2_Ks1(self, sigma, rho, omega, tau_val):
        """Generated order-2 kernel must match Ks1."""
        gen = make_kernel(2)
        K_gen = gen.create_K_hat(
            jnp.array(tau_val),
            jnp.array(sigma),
            jnp.array(rho),
            jnp.array(omega),
        )
        K_ref = Ks1(tau_val, sigma, rho, omega)
        npt.assert_allclose(np.asarray(K_gen), np.asarray(K_ref), atol=1e-10)


# ---------------------------------------------------------------------------
# Integration with hm.py
# ---------------------------------------------------------------------------


class TestHmIntegration:
    """Test that hm.py Ks() and HidaMatern dispatch to the generator.

    Convention: HidaMatern.order (and dict 'order') is the Matern smoothness
    index. The SSM state dimension is order + 1 (= nple). So order=2 means
    Matern-5/2 with a 3x3 state-space matrix.
    """

    def test_Ks_order2(self):
        """Ks() should accept order 2 (Matern-5/2, 3x3)."""
        kp = {"sigma": 1.0, "rho": 1.0, "omega": 0.0, "order": 2}
        K = Ks(kp, 1.0)
        assert K.shape == (3, 3)

    def test_Ks_order3(self):
        """Ks() should accept order 3 (Matern-7/2, 4x4)."""
        kp = {"sigma": 1.0, "rho": 1.0, "omega": 2.0, "order": 3}
        K = Ks(kp, 0.5)
        assert K.shape == (4, 4)

    def test_HidaMatern_K_order2(self):
        """HidaMatern.K() should work for order 2 (3x3)."""
        hm = HidaMatern(sigma=1.0, rho=1.0, omega=0.0, order=2, s=0.0)
        K = hm.K(1.0)
        assert K.shape == (3, 3)

    def test_HidaMatern_K_order3(self):
        """HidaMatern.K() should work for order 3 (4x4)."""
        hm = HidaMatern(sigma=1.5, rho=2.0, omega=3.0, order=3, s=0.0)
        K = hm.K(0.5)
        assert K.shape == (4, 4)

    def test_HidaMatern_Af_Qf_order2(self):
        """Full SSM chain should work for order 2 (3x3)."""
        hm = HidaMatern(sigma=1.0, rho=1.0, omega=0.0, order=2, s=1e-6)
        A = hm.Af(1.0)
        Q = hm.Qf(1.0)
        assert A.shape == (3, 3)
        assert Q.shape == (3, 3)

    def test_ssm_repr_order2(self):
        """ssm_repr should work with order-2 kernels (3x3)."""
        from cvhmax.hm import ssm_repr

        kernelparams = [
            [{"sigma": 1.0, "rho": 1.0, "omega": 0.0, "order": 2}],
        ]
        Af, Qf, Ab, Qb = ssm_repr(kernelparams, 1.0)
        from jax import tree_util

        paramflat, _ = tree_util.tree_flatten(Af)
        assert len(paramflat) == 1
        assert paramflat[0].shape == (3, 3)

    def test_ssm_repr_mixed_orders(self):
        """ssm_repr should handle mixed hand-coded and generated orders."""
        from cvhmax.hm import ssm_repr

        kernelparams = [
            [{"sigma": 1.0, "rho": 1.0, "omega": 0.0, "order": 0}],
            [
                {"sigma": 1.0, "rho": 1.0, "omega": 0.0, "order": 1},
                {"sigma": 1.0, "rho": 1.0, "omega": 1.0, "order": 2},
            ],
        ]
        Af, Qf, Ab, Qb = ssm_repr(kernelparams, 1.0)
        from jax import tree_util

        paramflat, _ = tree_util.tree_flatten(Af)
        assert len(paramflat) == 3


# ---------------------------------------------------------------------------
# Moments tests
# ---------------------------------------------------------------------------


class TestMoments:
    """Tests for the spectral moments computation."""

    def test_moments_shape(self, gen):
        """Moments vector should have length 2M."""
        m = gen.get_moments(jnp.array(1.0), jnp.array(1.0), jnp.array(0.0))
        assert m.shape == (2 * gen.order,)

    def test_moments_odd_zero(self, gen):
        """Odd-indexed moments should be zero."""
        m = gen.get_moments(jnp.array(1.0), jnp.array(1.0), jnp.array(0.0))
        odd_moments = m[1::2]
        npt.assert_allclose(np.asarray(odd_moments), 0.0, atol=1e-12)

    def test_moments_real(self, gen):
        """Moments should be real-valued."""
        m = gen.get_moments(jnp.array(1.0), jnp.array(1.0), jnp.array(2.0))
        assert not jnp.iscomplexobj(m) or jnp.allclose(m.imag, 0.0, atol=1e-12)

    def test_moments_zeroth_equals_sigma_sq(self, gen):
        """The zeroth moment should equal sigma^2."""
        sigma = 2.5
        m = gen.get_moments(jnp.array(sigma), jnp.array(1.0), jnp.array(0.0))
        npt.assert_allclose(float(m[0]), sigma**2, rtol=1e-10)

    @pytest.mark.parametrize("sigma,rho,omega", _PARAM_GRID)
    def test_moments_nonnegative(self, gen, sigma, rho, omega):
        """All moments should be non-negative."""
        m = gen.get_moments(jnp.array(sigma), jnp.array(rho), jnp.array(omega))
        assert jnp.all(m >= -1e-12), f"Negative moments: {m}"


# ---------------------------------------------------------------------------
# Base kernel tests
# ---------------------------------------------------------------------------


class TestBaseKernel:
    """Tests for the scalar base kernel."""

    def test_base_kernel_at_zero(self, gen):
        """Base kernel at tau=0 should equal sigma^2."""
        sigma = 1.5
        bk = gen.get_base_kernel(
            jnp.array(0.0), jnp.array(sigma), jnp.array(1.0), jnp.array(0.0)
        )
        npt.assert_allclose(float(bk), sigma**2, rtol=1e-10)

    def test_base_kernel_real(self, gen):
        """Base kernel output should be real."""
        bk = gen.get_base_kernel(
            jnp.array(0.5), jnp.array(1.0), jnp.array(1.0), jnp.array(0.0)
        )
        # Should be a real-valued scalar
        assert not jnp.iscomplexobj(bk)

    def test_base_kernel_decay(self, gen):
        """Base kernel should decay with increasing tau (no oscillation)."""
        sigma = jnp.array(1.0)
        rho = jnp.array(1.0)
        omega = jnp.array(0.0)
        bk0 = gen.get_base_kernel(jnp.array(0.0), sigma, rho, omega)
        bk1 = gen.get_base_kernel(jnp.array(1.0), sigma, rho, omega)
        bk5 = gen.get_base_kernel(jnp.array(5.0), sigma, rho, omega)
        assert float(bk0) >= float(bk1) >= float(bk5)


# ---------------------------------------------------------------------------
# Mathematical correctness (paper-derived)
# ---------------------------------------------------------------------------


class TestMathematicalCorrectness:
    """Tests derived from Hida-Matérn kernel definitions."""

    @pytest.mark.parametrize("order", [1, 2, 3])
    @pytest.mark.parametrize("tau_val", [0.1, 1.0])
    def test_closed_form_matern(self, order, tau_val):
        sigma = jnp.array(1.2)
        rho = jnp.array(0.7)
        omega = jnp.array(0.0)
        gen = make_kernel(order)
        expected = _matern_closed_form(order, tau_val, sigma, rho)
        actual = gen.get_base_kernel(
            jnp.array(tau_val), sigma, rho, omega
        )
        npt.assert_allclose(float(actual), float(expected), rtol=1e-10, atol=1e-12)

    def test_derivative_construction(self):
        """K_hat outer entries match kernel derivatives at tau>0."""
        order = 3
        gen = make_kernel(order)
        sigma = jnp.array(1.0)
        rho = jnp.array(1.0)
        omega = jnp.array(0.7)
        tau = 0.5
        h = 1e-4

        def k_fn(t):
            return _complex_kernel(order, t, sigma, rho, omega)

        derivs = [k_fn(tau), _finite_diff_first(k_fn, tau, h), _finite_diff_second(k_fn, tau, h)]
        K_hat = gen.create_K_hat(jnp.array(tau), sigma, rho, omega)

        for c in range(order):
            expected = (-1.0) ** c * derivs[c]
            actual = K_hat[0, c]
            npt.assert_allclose(
                np.asarray(actual),
                np.asarray(expected),
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"Derivative mismatch at c={c}",
            )

    @pytest.mark.parametrize("tau_val", [0.1, 1.0])
    def test_oscillation_phase_only_scalar(self, tau_val):
        """Oscillation should modulate phase, not magnitude, for k(tau)."""
        order = 2
        sigma = jnp.array(1.0)
        rho = jnp.array(1.0)
        gen = make_kernel(order)
        k0 = gen.create_K_hat(jnp.array(tau_val), sigma, rho, jnp.array(0.0))[0, 0]
        k1 = gen.create_K_hat(jnp.array(tau_val), sigma, rho, jnp.array(3.0))[0, 0]
        npt.assert_allclose(
            np.abs(np.asarray(k0)),
            np.abs(np.asarray(k1)),
            rtol=1e-10,
            atol=1e-12,
        )


# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------


class TestNumericalStability:
    """Stability checks for the generated kernels."""

    @pytest.mark.parametrize("order", [2, 3])
    def test_tau_zero_limit_consistency(self, order):
        gen = make_kernel(order)
        sigma = jnp.array(1.0)
        rho = jnp.array(1.0)
        omega = jnp.array(0.0)
        K0 = gen.create_K_hat(jnp.array(0.0), sigma, rho, omega)
        K_eps = gen.create_K_hat(jnp.array(1e-6), sigma, rho, omega)
        diff = jnp.linalg.norm(K0 - K_eps)
        denom = jnp.linalg.norm(K0)
        rel = diff / denom
        assert float(rel) < 1e-5

    @pytest.mark.parametrize("rho", [0.5, 2.0])
    @pytest.mark.parametrize("omega", [0.0, 2.0])
    def test_high_order_finite_outputs(self, rho, omega):
        gen = make_kernel(8)
        sigma = jnp.array(1.0)
        rho = jnp.array(rho)
        omega = jnp.array(omega)
        K_hat = gen.create_K_hat(jnp.array(0.1), sigma, rho, omega)
        moments = gen.get_moments(sigma, rho, omega)
        assert jnp.all(jnp.isfinite(K_hat)), "K_hat contains NaN/Inf"
        assert jnp.all(jnp.isfinite(moments)), "moments contain NaN/Inf"

    def test_conditioning_sanity(self):
        gen = make_kernel(3)
        sigma = jnp.array(1.0)
        rho = jnp.array(1.0)
        omega = jnp.array(0.0)
        K0 = gen.create_K_hat(jnp.array(0.0), sigma, rho, omega)
        K0_real = real_repr(K0)
        cond = jnp.linalg.cond(K0_real)
        assert jnp.isfinite(cond)
        assert float(cond) < 1e12


# ---------------------------------------------------------------------------
# JIT compatibility tests
# ---------------------------------------------------------------------------


class TestJITCompatibility:
    """Verify all methods work under jax.jit."""

    def test_jit_create_K_hat(self, gen):
        """create_K_hat should be JIT-compatible."""

        @jax.jit
        def f(tau, sigma, rho, omega):
            return gen.create_K_hat(tau, sigma, rho, omega)

        K = f(jnp.array(0.5), jnp.array(1.0), jnp.array(1.0), jnp.array(0.0))
        assert K.shape == (gen.order, gen.order)

    def test_jit_get_moments(self, gen):
        """get_moments should be JIT-compatible."""

        @jax.jit
        def f(sigma, rho, omega):
            return gen.get_moments(sigma, rho, omega)

        m = f(jnp.array(1.0), jnp.array(1.0), jnp.array(0.0))
        assert m.shape == (2 * gen.order,)

    def test_jit_get_base_kernel(self, gen):
        """get_base_kernel should be JIT-compatible."""

        @jax.jit
        def f(tau, sigma, rho, omega):
            return gen.get_base_kernel(tau, sigma, rho, omega)

        bk = f(jnp.array(0.5), jnp.array(1.0), jnp.array(1.0), jnp.array(0.0))
        assert bk.shape == ()


# ---------------------------------------------------------------------------
# Parity with PyTorch reference (conditional)
# ---------------------------------------------------------------------------


@requires_ref
class TestParityPyTorchReference:
    """Cross-implementation parity tests against the PyTorch reference.

    These tests are only run when torch and the reference code are available.
    """

    @pytest.mark.parametrize("order", [1, 2, 3])
    @pytest.mark.parametrize("tau_val", [0.0, 0.5, 1.0])
    def test_K_hat_parity(self, order, tau_val):
        """Generated K_hat should match PyTorch reference."""
        import torch
        import importlib

        # Load PyTorch reference module
        mod = importlib.import_module(
            f"hida_matern_kernel_generator.hm_ss_kernels."
            f"hida_M_{order}.hida_M_{order}_K_hat"
        )

        # PyTorch reference uses log-space + softplus
        log_sigma = torch.tensor(0.0, dtype=torch.float64)  # softplus(0) ~ 0.693
        log_ls = torch.tensor(0.0, dtype=torch.float64)
        log_b = torch.tensor(0.0, dtype=torch.float64)

        sigma_val = float(torch.nn.functional.softplus(log_sigma))
        rho_val = float(torch.nn.functional.softplus(log_ls))
        omega_val = float(torch.nn.functional.softplus(log_b))

        K_ref = mod.create_K_hat(
            tau_val, log_sigma, log_ls, log_b, dtype=torch.complex128
        )
        K_ref = K_ref.numpy()

        # JAX implementation uses raw positive params
        gen = make_kernel(order)
        K_jax = gen.create_K_hat(
            jnp.array(tau_val),
            jnp.array(sigma_val),
            jnp.array(rho_val),
            jnp.array(omega_val),
        )
        K_jax = np.asarray(K_jax)

        npt.assert_allclose(K_jax, K_ref, atol=1e-8)
