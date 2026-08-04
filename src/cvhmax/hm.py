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
import jax.scipy as jsp
import numpy as np

from cvhmax.utils import gamma

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


def _stabilize_covariance(Q: jnp.ndarray) -> jnp.ndarray:
    """Return the Hermitian part of a derived covariance block.

    Numerical regularization belongs on the stationary covariance before
    solves and conditional-covariance subtraction. Adding a correction after
    deriving ``Q`` would hide cancellation and break the covariance identity.
    """
    return _hermitian(Q)

# TODO: see sympy2jax, equinox
# NOTE: cos(x) = (exp(j * x) + exp(-j * x)) / 2 = Real[exp(j * x)]


def _validate_order(order) -> int:
    """Validate static state-space order metadata."""
    if not isinstance(order, (int, np.integer)) or isinstance(order, bool):
        raise TypeError(f"Matérn order must be a static integer, got {order!r}")
    order = int(order)
    if order < 0:
        raise ValueError(f"Matérn order must be non-negative, got {order}")
    return order


def _validate_scalar_parameters(sigma, rho, omega, s=0.0):
    """Validate scalar kernel parameters at the object/API boundary."""
    values = {"sigma": sigma, "rho": rho, "omega": omega, "s": s}
    for name, value in values.items():
        array = np.asarray(value)
        if array.ndim != 0 or not np.isfinite(array):
            raise ValueError(f"{name} must be a finite scalar, got {value!r}")
    if float(np.asarray(rho)) <= 0:
        raise ValueError(f"rho must be positive, got {rho!r}")
    if float(np.asarray(s)) < 0:
        raise ValueError(f"s must be non-negative, got {s!r}")


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
    order = _validate_order(order)
    try:
        rho_array = np.asarray(rho)
    except (TypeError, ValueError):
        rho_array = None
    if rho_array is not None and (
        rho_array.ndim != 0
        or not np.isfinite(rho_array)
        or float(rho_array) <= 0
    ):
        raise ValueError(f"rho must be positive and finite, got {rho!r}")
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


def _adaptive_cholesky(K0, *, jitter):
    """Factor a stationary covariance, escalating only when necessary.

    The candidate ladder is fixed so this helper remains compatible with JAX
    transformations. The first candidate is the requested covariance; later
    candidates add machine-scale multiples of the stationary diagonal scale.
    A failed ladder returns the final factor, whose non-finite entries make
    the failure visible to downstream numerical checks rather than clipping a
    derived process-noise matrix.
    """
    K0 = _hermitian(K0)
    dtype = K0.real.dtype
    diagonal_scale = jnp.maximum(jnp.max(jnp.abs(jnp.real(jnp.diag(K0)))), 1.0)
    base = jnp.asarray(jitter, dtype=dtype)
    machine = jnp.finfo(dtype).eps * diagonal_scale
    candidates = jnp.concatenate(
        [
            base[None],
            base + machine * (10.0 ** jnp.arange(5, dtype=dtype)),
        ]
    )
    eye = jnp.eye(K0.shape[-1], dtype=K0.dtype)
    factors = jnp.stack(
        [jnp.linalg.cholesky(K0 + candidate * eye) for candidate in candidates]
    )
    valid = jnp.all(jnp.isfinite(factors), axis=(1, 2))
    selected = jnp.argmax(valid)
    # If every candidate failed, use the last factor so NaNs propagate and the
    # caller can diagnose the invalid covariance; never eigenvalue-clip here.
    selected = jnp.where(jnp.any(valid), selected, len(candidates) - 1)
    return factors[selected], K0 + candidates[selected] * eye


def _dynamics_from_covariances(K0, Kt, *, jitter=0.0):
    """Derive dynamics using a Cholesky solve and conditional covariance."""
    K0 = _hermitian(K0)
    chol, K0 = _adaptive_cholesky(K0, jitter=jitter)
    solve_K0 = lambda rhs: jsp.linalg.cho_solve((chol, True), rhs)
    forward = conjtrans(solve_K0(conjtrans(Kt)))
    forward_noise = K0 - Kt @ solve_K0(conjtrans(Kt))
    backward = conjtrans(solve_K0(Kt))
    backward_noise = K0 - conjtrans(Kt) @ solve_K0(Kt)
    return _Dynamics(
        forward,
        _stabilize_covariance(forward_noise),
        backward,
        _stabilize_covariance(backward_noise),
    )


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

    def __post_init__(self):
        self.order = _validate_order(self.order)
        _validate_scalar_parameters(self.sigma, self.rho, self.omega, self.s)

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
            Complex state covariance for the requested scalar lag.

        Notes
        -----
        ``s`` is an instantaneous state-space component: it is added to the
        stationary block at zero lag and is absent from positive-lag
        cross-covariances.
        """
        compute_dtype = compute_dtype or _kernel_compute_dtype()
        output_dtype = output_dtype or _kernel_output_dtype(
            tau, self.sigma, self.rho, self.omega, self.s
        )
        tau_c, sigma_c, rho_c, omega_c, s_c = _cast_kernel_inputs(
            compute_dtype, tau, self.sigma, self.rho, self.omega, self.s
        )
        if tau_c.ndim != 0:
            raise ValueError(f"state covariance tau must be scalar, got shape {tau_c.shape}")

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
        zero_lag = jnp.equal(tau_c, 0.0)
        K = K + jnp.eye(self.nple, dtype=K.dtype) * jnp.where(zero_lag, s_c, 0.0)
        return K.astype(_kernel_complex_dtype(output_dtype))

    @property
    def nple(self) -> int:
        return self.order + 1

    def state_scale(self):
        """Return diagonal correlation scaling for the stabilized state.

        The scaling is ``D_ii = 1 / sqrt(real(K_ii(0)))``. The configured
        instantaneous component is included because the same stationary
        covariance is used by the state-space solves.
        """
        K0 = self.K(0.0)
        diagonal = jnp.real(jnp.diag(K0))
        if not bool(jnp.all(jnp.isfinite(diagonal) & (diagonal > 0))):
            raise ValueError("stationary state covariance has invalid diagonal")
        return 1.0 / jnp.sqrt(diagonal)

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
    order = _validate_order(order)
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


def make_Ks(order: int):
    """Create a JAX-compatible raw kernel function with static ``order``.

    ``order`` determines the returned matrix shape and is therefore closed
    over when tracing. The returned function accepts a mapping containing only
    numerical ``sigma``, ``rho``, and ``omega`` leaves.
    """
    order = _validate_order(order)

    def kernel(kernelparam, tau, *, compute_dtype=None, output_dtype=None):
        params = {
            "sigma": kernelparam["sigma"],
            "rho": kernelparam["rho"],
            "omega": kernelparam["omega"],
            "order": order,
        }
        return Ks(
            params,
            tau,
            compute_dtype=compute_dtype,
            output_dtype=output_dtype,
        )

    return kernel


def _stabilized_state_covariance(kernelparam, tau, *, jitter, compute_dtype):
    """Return raw covariance plus instantaneous jitter at zero lag."""
    K = Ks(kernelparam, tau, compute_dtype=compute_dtype, output_dtype=compute_dtype)
    zero_lag = jnp.equal(jnp.asarray(tau), 0.0)
    return K + jnp.eye(K.shape[-1], dtype=K.dtype) * jnp.where(
        zero_lag, jnp.asarray(jitter, dtype=K.real.dtype), 0.0
    )


def _raw_dynamics_covariances(kernelparam, tau, *, jitter, compute_dtype):
    """Build the stationary and positive-lag blocks for state dynamics."""
    K0 = _stabilized_state_covariance(
        kernelparam, 0.0, jitter=jitter, compute_dtype=compute_dtype
    )
    Kt_raw = Ks(
        kernelparam, tau, compute_dtype=compute_dtype, output_dtype=compute_dtype
    )
    # A zero-step transition is the identity with zero process noise. For
    # positive lags, retain the raw cross-covariance so an instantaneous
    # component has no temporal cross-covariance.
    zero_lag = jnp.equal(jnp.asarray(tau), 0.0)
    Kt = jnp.where(zero_lag, K0, Kt_raw)
    return K0, Kt


def _dynamics(kernelparam, tau, *, jitter=0.0):
    """Return dynamics from a jittered stationary and raw lag block."""
    compute_dtype = _kernel_compute_dtype()
    K0, Kt = _raw_dynamics_covariances(
        kernelparam, tau, jitter=jitter, compute_dtype=compute_dtype
    )
    return _dynamics_from_covariances(K0, Kt)


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
