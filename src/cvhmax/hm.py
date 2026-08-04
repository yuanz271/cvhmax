"""
Hida-Matern Kernel
"""

# Only polynomial of integer order p for Matern kernels
# 1/2 -> p=0
# 3/2 -> p=1
# 5/2 -> p=2
# etc.
from dataclasses import dataclass
from functools import partial
from math import factorial
from operator import itemgetter
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from cvhmax.utils import EPS, gamma

from .utils import conjtrans


def _kernel_compute_dtype() -> jnp.dtype:
    """Return the dtype used for numerically sensitive kernel computations."""
    return jnp.float64


def _kernel_output_dtype(*values) -> jnp.dtype:
    """Infer the desired output dtype from the input values."""
    arrays = [jnp.asarray(v) for v in values if v is not None]
    if not arrays:
        return jnp.float32
    return jnp.result_type(*arrays)


def _kernel_complex_dtype(output_dtype: jnp.dtype) -> jnp.dtype:
    """Map a real-valued output dtype to the matching complex dtype."""
    return jnp.result_type(jnp.complex64, output_dtype)


def _cast_kernel_inputs(dtype: jnp.dtype, *values):
    """Cast kernel inputs to the desired compute dtype."""
    return [jnp.asarray(v, dtype=dtype) for v in values]


def _hermitian(x: jnp.ndarray) -> jnp.ndarray:
    """Return the Hermitian-symmetrized version of a matrix."""
    return 0.5 * (x + conjtrans(x))


def _stabilize_covariance(
    Q: jnp.ndarray, *, jitter: jnp.ndarray, output_dtype: jnp.dtype
) -> jnp.ndarray:
    """Hermitian-symmetrize and add jitter in float32/complex64 outputs."""
    Q = _hermitian(Q)
    if jnp.dtype(_kernel_complex_dtype(output_dtype)) == jnp.dtype(jnp.complex64):
        Q = Q + jnp.eye(Q.shape[-1], dtype=Q.dtype) * jitter
    return Q

# TODO: see sympy2jax, equinox
# NOTE: cos(x) = (exp(j * x) + exp(-j * x)) / 2 = Real[exp(j * x)]


def matern(tau: float, *, rho: float, order: int):
    """Evaluate a half-integer Matérn kernel with unit amplitude.

    ``order=p`` corresponds to Matérn smoothness ``nu=p+1/2``. The
    half-integer Matérn family has a closed form in ``d=abs(tau)``.

    Parameters
    ----------
    tau : float or Array
        Time lag at which the kernel is evaluated.
    rho : float
        Length scale of the kernel.
    order : int
        Non-negative integer smoothness index ``p``.

    Returns
    -------
    Array
        Matérn covariance with value one at zero lag.
    """
    if order < 0:
        raise ValueError(f"Matérn order must be non-negative, got {order}")

    d = jnp.abs(tau)
    x = d / rho
    prefactor = factorial(order) / factorial(2 * order)
    decay = jnp.exp(-jnp.sqrt(2 * order + 1) * x)
    polynomial = sum(
        factorial(order + i)
        / (factorial(i) * factorial(order - i))
        * (jnp.sqrt(8 * order + 4) * x) ** (order - i)
        for i in range(order + 1)
    )
    return prefactor * decay * polynomial


def hm(tau: float, *, sigma: float, rho: float, order: int, omega: float):
    """Return the Hida–Matérn covariance at lag ``tau``.

    Parameters
    ----------
    tau : float
        Time lag at which the covariance is evaluated.
    sigma : float
        Kernel amplitude.
    rho : float
        Length scale parameter.
    order : int
        Matérn smoothness order.
    omega : float
        Oscillation frequency in radians per unit time.

    Returns
    -------
    Array
        Real-valued covariance for the requested lag.
    """
    # cos(t) == cos(-t)
    return sigma**2 * jnp.cos(omega * tau) * matern(tau, rho=rho, order=order)


def _matern_polynomial_coefficients(order, rho):
    """Return ascending coefficients of the normalized half-integer kernel."""
    prefactor = factorial(order) / factorial(2 * order)
    scale = jnp.sqrt(8 * order + 4) / rho
    return jnp.array(
        [
            prefactor
            * factorial(order + order - power)
            / (factorial(order - power) * factorial(power))
            * scale**power
            for power in range(order + 1)
        ]
    )


def _polynomial_derivative_value(coefficients, derivative, tau):
    """Evaluate one derivative of a polynomial at ``tau``."""
    return sum(
        coefficient
        * factorial(power)
        / factorial(power - derivative)
        * tau ** (power - derivative)
        for power, coefficient in enumerate(coefficients)
        if power >= derivative
    )


def _state_covariance_from_polynomial(order, tau, sigma, rho, omega):
    """Assemble a state covariance from a Matérn polynomial and derivatives."""
    d = jnp.abs(tau)
    coefficients = _matern_polynomial_coefficients(order, rho)
    decay = jnp.sqrt(2 * order + 1) / rho
    lam = 1.0j * omega - decay
    exponential = jnp.exp(lam * d)

    derivatives = []
    for derivative in range(2 * order + 1):
        derivatives.append(
            exponential
            * sum(
                jnp.asarray(
                    factorial(derivative)
                    / (factorial(r) * factorial(derivative - r))
                )
                * lam ** (derivative - r)
                * _polynomial_derivative_value(coefficients, r, d)
                for r in range(min(derivative, order) + 1)
            )
        )

    return sigma**2 * jnp.array(
        [
            [(-1) ** column * derivatives[row + column] for column in range(order + 1)]
            for row in range(order + 1)
        ]
    )


# Built-in entries share the same polynomial/derivative assembler. A future
# optimized order can register a function with this signature at one place.
_BUILTIN_STATE_COVARIANCES = {
    order: partial(_state_covariance_from_polynomial, order)
    for order in (0, 1, 2)
}


class _Dynamics(NamedTuple):
    """Raw state-space dynamics derived from two covariance blocks."""

    forward: jnp.ndarray
    forward_noise: jnp.ndarray
    backward: jnp.ndarray
    backward_noise: jnp.ndarray


def _dynamics_from_covariances(K0, Kt, *, output_dtype):
    """Derive stabilized forward/backward dynamics from covariance blocks."""
    forward = conjtrans(jnp.linalg.solve(conjtrans(K0), conjtrans(Kt)))
    forward_noise = K0 - Kt @ jnp.linalg.solve(K0, conjtrans(Kt))
    backward = conjtrans(jnp.linalg.solve(conjtrans(K0), Kt))
    backward_noise = K0 - conjtrans(Kt) @ jnp.linalg.solve(K0, Kt)
    forward_noise = _stabilize_covariance(
        forward_noise,
        jitter=jnp.asarray(EPS, dtype=forward_noise.dtype),
        output_dtype=output_dtype,
    )
    backward_noise = _stabilize_covariance(
        backward_noise,
        jitter=jnp.asarray(EPS, dtype=backward_noise.dtype),
        output_dtype=output_dtype,
    )
    return _Dynamics(forward, forward_noise, backward, backward_noise)


@dataclass
class HidaMatern:
    """Hida-Matern kernel parameterised as a linear Gaussian SSM.

    Parameters
    ----------
    sigma : float, default=1.0
        Kernel amplitude.
    rho : float, default=1.0
        Length scale controlling temporal decay.
    omega : float, default=0.0
        Oscillation frequency in radians per unit time.
    order : int, default=0
        Smoothness order of the Matérn kernel.
    s : float, default=1e-5
        Jitter added to the stationary covariance for numerical stability.

    Notes
    -----
    Orders 0, 1, and 2 use closed-form covariance blocks for the
    Matérn-1/2, Matérn-3/2, and Matérn-5/2 kernels. Higher orders use the
    optional symbolic kernel generator.
    """

    sigma: float = 1.0
    rho: float = 1.0
    omega: float = 0.0
    order: int = 0
    s: float = 1e-5

    def cov(self, tau=0.0):
        raise NotImplementedError

    def kernel(self, tau=0.0):
        """Evaluate the scalar, jitter-free Hida–Matérn covariance.

        The returned real-valued covariance is
        ``sigma**2 * matern(tau, rho=rho, order=order) * cos(omega*tau)``.
        Unlike :meth:`K`, this method returns only the function-level kernel;
        it does not include derivative-state entries or numerical jitter.
        ``tau`` may be a scalar or an array.
        """
        return hm(
            tau,
            sigma=self.sigma,
            rho=self.rho,
            order=self.order,
            omega=self.omega,
        )

    def K(self, tau=0.0, *, compute_dtype: jnp.dtype | None = None, output_dtype: jnp.dtype | None = None):
        """Return the state-space covariance block at lag ``tau``.

        Parameters
        ----------
        tau : float, default=0.0
            Time lag at which the block is evaluated.
        compute_dtype : jnp.dtype | None, default=None
            Internal dtype used for numerically sensitive computations.
        output_dtype : jnp.dtype | None, default=None
            Output dtype returned to callers.

        Returns
        -------
        Array
            Complex state covariance for the requested lag.
        """
        compute_dtype = compute_dtype or _kernel_compute_dtype()
        output_dtype = output_dtype or _kernel_output_dtype(
            tau, self.sigma, self.rho, self.omega, self.s
        )
        tau_c, sigma_c, rho_c, omega_c, s_c = _cast_kernel_inputs(
            compute_dtype, tau, self.sigma, self.rho, self.omega, self.s
        )

        params = {
            "sigma": sigma_c,
            "rho": rho_c,
            "omega": omega_c,
            "order": self.order,
        }
        K = Ks(
            params,
            tau_c,
            compute_dtype=compute_dtype,
            output_dtype=compute_dtype,
        )
        K = K + jnp.eye(self.nple, dtype=K.dtype) * s_c
        return K.astype(_kernel_complex_dtype(output_dtype))

    @property
    def nple(self) -> int:
        return self.order + 1

    def state_scale(self):
        """Return the diagonal correlation scaling for the stabilized state.

        The scaling is ``D_ii = 1 / sqrt(real(K_ii(0)))``. The configured
        jitter is included because the same stabilized stationary covariance
        is used by the state-space solves. Applying ``D`` on both sides
        normalizes each state to unit marginal variance.
        """
        K0 = self.K(0.0)
        return 1.0 / jnp.sqrt(jnp.real(jnp.diag(K0)))

    def Af(self, tau):
        """Forward dynamics transition.

        Parameters
        ----------
        tau : float
            Time step for the state transition.

        Returns
        -------
        Array
            Real-valued transition matrix.
        """
        params = self._parameters()
        return Af(params, tau, jitter=self.s)

    def Qf(self, tau):
        """Forward dynamics state noise covariance.

        Parameters
        ----------
        tau : float
            Time step for the state transition.

        Returns
        -------
        Array
            Real-valued process noise covariance.
        """
        params = self._parameters()
        return Qf(params, tau, jitter=self.s)

    def Ab(self, tau):
        """Backward dynamics transition.

        Parameters
        ----------
        tau : float
            Time step for the state transition.

        Returns
        -------
        Array
            Real-valued transition matrix for the reverse-time model.
        """
        params = self._parameters()
        return Ab(params, tau, jitter=self.s)

    def Qb(self, tau):
        """Backward dynamics state noise covariance.

        Parameters
        ----------
        tau : float
            Time step for the state transition.

        Returns
        -------
        Array
            Real-valued process noise covariance for the reverse-time model.
        """
        params = self._parameters()
        return Qb(params, tau, jitter=self.s)

    def _parameters(self):
        """Return the JAX-pytree kernel parameter mapping."""
        return {
            "sigma": self.sigma,
            "rho": self.rho,
            "omega": self.omega,
            "order": self.order,
        }

    def spectral(self):
        raise NotImplementedError


# TODO: composite kernel: linear combination
# TODO: kernel parameters as pytree: List[composite kernel per latent dimension]; composite kernel: List[HM kernel]
# TODO: +: (k1, k2) -> k
# # 2 latents
# # L1: 1 kernel
# # L2: 2 kernels
# hyperparams = [[{'sigma': 1., 'rho': 1., 'omega': 0., 'order': 1}], [{'sigma': 1., 'rho': 1., 'omega': 0., 'order': 0}, {'sigma': 1., 'rho': 1., 'omega': 1., 'order': 1}]]
# hyperspec = [[{'sigma': True, 'rho': True, 'omega': True, 'order': False}], [{'sigma': True, 'rho': True, 'omega': True, 'order': False}, {'sigma': True, 'rho': True, 'omega': True, 'order': False}]]
# # print(tree_util.tree_structure(hyperparams))
# hyperdef, hyperflat = tree_util.tree_flatten(hyperparams)
# # print(hyperflat)
# # https://docs.kidger.site/equinox/all-of-equinox/
# # eqx.partition

# params, static = eqx.partition(hyperparams, hyperspec)
# eqx.tree_pprint(params)
# eqx.tree_pprint(static)
# paramdef, paramflat = tree_util.tree_flatten(params)


def Ks(kernelparam, tau, *, compute_dtype: jnp.dtype | None = None, output_dtype: jnp.dtype | None = None):
    """Look up the complex-valued HM state covariance block.

    Parameters
    ----------
    kernelparam : dict
        Kernel hyperparameters containing `sigma`, `rho`, `omega`, and `order`.
    tau : float
        Time lag at which the block is evaluated.
    compute_dtype : jnp.dtype | None, default=None
        Internal dtype used for numerically sensitive computations.
    output_dtype : jnp.dtype | None, default=None
        Output dtype returned to callers.

    Returns
    -------
    Array
        Complex state covariance block.
    """
    sigma, rho, omega, order = itemgetter("sigma", "rho", "omega", "order")(kernelparam)
    compute_dtype = compute_dtype or _kernel_compute_dtype()
    output_dtype = output_dtype or _kernel_output_dtype(tau, sigma, rho, omega)
    tau_c, sigma_c, rho_c, omega_c = _cast_kernel_inputs(
        compute_dtype, tau, sigma, rho, omega
    )

    builtin = _BUILTIN_STATE_COVARIANCES.get(order)
    if builtin is not None:
        K = builtin(tau_c, sigma_c, rho_c, omega_c)
    else:
        try:
            from .kernel_generator import make_kernel
        except ImportError:
            raise ImportError(
                "Orders >= 3 require the kergen extra. "
                "Install with:  pip install cvhmax[kergen]"
            ) from None

        # Generator order M = order + 1 (SSM state dimension)
        K = make_kernel(order + 1).create_K_hat(
            tau_c,
            sigma_c,
            rho_c,
            omega_c,
        )

    return K.astype(_kernel_complex_dtype(output_dtype))


def _stabilized_state_covariance(kernelparam, tau, *, jitter, compute_dtype):
    """Return a state covariance with explicit numerical jitter."""
    K = Ks(kernelparam, tau, compute_dtype=compute_dtype, output_dtype=compute_dtype)
    return K + jnp.eye(K.shape[-1], dtype=K.dtype) * jnp.asarray(
        jitter, dtype=K.real.dtype
    )


def _dynamics(kernelparam, tau, *, jitter=0.0):
    """Return dynamics from explicitly stabilized covariance blocks."""
    sigma, rho, omega, _ = itemgetter("sigma", "rho", "omega", "order")(kernelparam)
    compute_dtype = _kernel_compute_dtype()
    output_dtype = _kernel_output_dtype(tau, sigma, rho, omega, jitter)
    Kt = _stabilized_state_covariance(
        kernelparam, tau, jitter=jitter, compute_dtype=compute_dtype
    )
    K0 = _stabilized_state_covariance(
        kernelparam, 0.0, jitter=jitter, compute_dtype=compute_dtype
    )
    return _dynamics_from_covariances(K0, Kt, output_dtype=output_dtype)


def Af(kernelparam, tau, *, jitter=0.0):
    """Forward dynamics transition for a kernel dictionary.

    Parameters
    ----------
    kernelparam : dict
        Kernel hyperparameters containing `sigma`, `rho`, `omega`, and `order`.
    tau : float
        Time step for the state transition.

    Returns
    -------
    Array
        Real-valued transition matrix.
    """
    sigma, rho, omega, _ = itemgetter("sigma", "rho", "omega", "order")(kernelparam)
    output_dtype = _kernel_output_dtype(tau, sigma, rho, omega)
    return _dynamics(kernelparam, tau, jitter=jitter).forward.astype(
        _kernel_complex_dtype(output_dtype)
    )


def Qf(kernelparam, tau, *, jitter=0.0):
    """Forward dynamics state noise covariance.

    Parameters
    ----------
    kernelparam : dict
        Kernel hyperparameters containing `sigma`, `rho`, `omega`, and `order`.
    tau : float
        Time step for the state transition.

    Returns
    -------
    Array
        Real-valued process noise covariance.
    """
    sigma, rho, omega, _ = itemgetter("sigma", "rho", "omega", "order")(kernelparam)
    output_dtype = _kernel_output_dtype(tau, sigma, rho, omega)
    return _dynamics(kernelparam, tau, jitter=jitter).forward_noise.astype(
        _kernel_complex_dtype(output_dtype)
    )


def Ab(kernelparam, tau, *, jitter=0.0):
    """Backward dynamics transition.

    Parameters
    ----------
    kernelparam : dict
        Kernel hyperparameters containing `sigma`, `rho`, `omega`, and `order`.
    tau : float
        Time step for the state transition.

    Returns
    -------
    Array
        Real-valued transition matrix for the reverse-time model.
    """
    sigma, rho, omega, _ = itemgetter("sigma", "rho", "omega", "order")(kernelparam)
    output_dtype = _kernel_output_dtype(tau, sigma, rho, omega)
    return _dynamics(kernelparam, tau, jitter=jitter).backward.astype(
        _kernel_complex_dtype(output_dtype)
    )


def Qb(kernelparam, tau, *, jitter=0.0):
    """
    Backward dynamics state noise covariance
    """
    sigma, rho, omega, _ = itemgetter("sigma", "rho", "omega", "order")(kernelparam)
    output_dtype = _kernel_output_dtype(tau, sigma, rho, omega)
    return _dynamics(kernelparam, tau, jitter=jitter).backward_noise.astype(
        _kernel_complex_dtype(output_dtype)
    )


def ssm_repr(kernelparams, tau):
    """
    Transform kernel specification into SSM parameters
    param kernelparams: List[latent]
        latent: List[spec]
        spec: Dict
    Example:
        2 latents
        Lat 1: 1 HM kernel
        Lat 2: 2 HM kernels
        kernelparams = [[{'sigma': 1., 'rho': 1., 'omega': 0., 'order': 1}], [{'sigma': 1., 'rho': 1., 'omega': 0., 'order': 0}, {'sigma': 1., 'rho': 1., 'omega': 1., 'order': 1}]]
    """
    # The big K matrix is a block matrix of all the primitive kernels
    tree_map = partial(jax.tree.map, is_leaf=lambda x: isinstance(x, dict))
    Afm = tree_map(partial(Af, tau=tau), kernelparams)
    Qfm = tree_map(partial(Qf, tau=tau), kernelparams)
    Abm = tree_map(partial(Ab, tau=tau), kernelparams)
    Qbm = tree_map(partial(Qb, tau=tau), kernelparams)

    return Afm, Qfm, Abm, Qbm


def spectral_density(kernel_spec: dict, freq):
    """
    HM power spectral density
    param kernel_spec: kernel specification
    param freq: frequencies that are calculated at
    """
    sigma, rho, omega, p = itemgetter("sigma", "rho", "omega", "order")(kernel_spec)

    # spectral density on R^1
    f_b = omega / (2 * jnp.pi)  # 2*pi*f = omega
    nu = p + 0.5
    num_c = sigma**2 * 2 * jnp.sqrt(jnp.pi) * gamma(nu + 0.5) * (2 * nu) ** nu
    den_c = gamma(nu) * rho ** (2 * nu)
    c = num_c / den_c

    s_pos_f = (2 * nu / rho**2 + 4 * jnp.pi**2 * (freq - f_b) ** 2) ** (-(nu + 0.5))
    s_pos_f_neg = (2 * nu / rho**2 + 4 * jnp.pi**2 * (-freq - f_b) ** 2) ** (
        -(nu + 0.5)
    )
    s = c * (s_pos_f + s_pos_f_neg)

    return s


def sample_matern(n, dt, sigma, rho, jitter=1e-6):
    """Sample a trajectory from a Matérn-1/2 Gaussian process.

    Parameters
    ----------
    n : int
        Number of time points to sample.
    dt : float
        Time step between samples.
    sigma : float
        Kernel amplitude.
    rho : float
        Length scale parameter.
    jitter : float, default=1e-6
        Diagonal regularization for numerical stability.

    Returns
    -------
    ndarray
        Sampled trajectory of length `n`.
    """
    t = np.arange(n) * dt
    D = np.abs(t[None, :] - t[:, None])
    K = sigma**2 * np.exp(-D / rho) + jitter * np.eye(n)
    L = np.linalg.cholesky(K)
    z = np.random.randn(n)
    x = L @ z
    return x
