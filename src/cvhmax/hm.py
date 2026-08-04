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
from typing import Dict

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


def Ks0(tau, sigma, rho, omega):
    """State-space kernel matrix for the 1/2-order HM kernel.

    Parameters
    ----------
    tau : float
        Time lag at which the kernel block is evaluated.
    sigma : float
        Kernel amplitude.
    rho : float
        Length scale parameter.
    omega : float
        Oscillation frequency in radians per unit time.

    Returns
    -------
    Array
        Complex state covariance block for the specified lag.
    """
    # Not confused with the kernel matrix
    d = jnp.abs(tau)
    return sigma**2 * jnp.array([[jnp.exp(d * (1.0j * omega - 1 / rho))]])


def Ks1(tau, sigma, rho, omega):
    """State-space kernel matrix for the 3/2-order HM kernel.

    Parameters
    ----------
    tau : float
        Time lag at which the kernel block is evaluated.
    sigma : float
        Kernel amplitude.
    rho : float
        Length scale parameter.
    omega : float
        Oscillation frequency in radians per unit time.

    Returns
    -------
    Array
        Complex state covariance block for the specified lag.
    """
    # Not confused with the kernel matrix
    d = jnp.abs(tau)
    sqrt3 = jnp.sqrt(3)

    return sigma**2 * jnp.array(
        [
            [
                (rho + sqrt3 * d)
                * jnp.exp(d * (1.0j * omega * rho - sqrt3) / rho)
                / rho,
                -(1.0j * omega * rho**2 + sqrt3 * 1.0j * omega * rho * d - 3 * d)
                * jnp.exp(d * (1.0j * omega - sqrt3 / rho))
                / rho**2,
            ],
            [
                (1.0j * omega * rho**2 + sqrt3 * 1.0j * omega * rho * d - 3 * d)
                * jnp.exp(d * (1.0j * omega - sqrt3 / rho))
                / rho**2,
                -(
                    -(omega**2) * rho**3
                    - sqrt3 * omega**2 * rho**2 * d
                    - 6.0j * omega * rho * d
                    - 3 * rho
                    + 3 * sqrt3 * d
                )
                * jnp.exp(d * (1.0j * omega - sqrt3 / rho))
                / rho**3,
            ],
        ]
    )


def Ks2(tau, sigma, rho, omega):
    """State-space kernel matrix for the 5/2-order HM kernel.

    This is the closed-form three-state covariance block for the
    Matérn-5/2 kernel modulated by ``exp(1j * omega * tau)``.  The
    derivatives are evaluated on the positive-lag branch, with ``abs(tau)``
    matching the convention used by :func:`Ks0` and :func:`Ks1`.
    """
    d = jnp.abs(tau)
    a = jnp.sqrt(5.0) / rho
    b = 5.0 / (3.0 * rho**2)
    lam = 1.0j * omega - a
    polynomial = 1.0 + a * d + b * d**2
    exponential = jnp.exp(lam * d)

    # k^(n)(d) / sigma^2 for n = 0, ..., 4, where
    # k(d) = sigma^2 * (1 + a*d + b*d^2) * exp((i*omega-a)*d).
    k0 = exponential * polynomial
    k1 = exponential * (lam * polynomial + a + 2.0 * b * d)
    k2 = exponential * (
        lam**2 * polynomial + 2.0 * lam * (a + 2.0 * b * d) + 2.0 * b
    )
    k3 = exponential * (
        lam**3 * polynomial
        + 3.0 * lam**2 * (a + 2.0 * b * d)
        + 6.0 * lam * b
    )
    k4 = exponential * (
        lam**4 * polynomial
        + 4.0 * lam**3 * (a + 2.0 * b * d)
        + 12.0 * lam**2 * b
    )

    return sigma**2 * jnp.array(
        [
            [k0, -k1, k2],
            [k1, -k2, k3],
            [k2, -k3, k4],
        ]
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

        # TODO: confusing with covariance matrix
        # somehow not decorable by cache or cached_property
        match self.order:
            case 0:
                K = Ks0(tau_c, sigma_c, rho_c, omega_c)
                K = K + jnp.eye(self.nple, dtype=K.dtype) * s_c
            case 1:
                K = Ks1(tau_c, sigma_c, rho_c, omega_c)
                K = K + jnp.eye(self.nple, dtype=K.dtype) * s_c
            case 2:
                K = Ks2(tau_c, sigma_c, rho_c, omega_c)
                K = K + jnp.eye(self.nple, dtype=K.dtype) * s_c
            case _:
                try:
                    from .kernel_generator import make_kernel
                except ImportError:
                    raise ImportError(
                        "Orders >= 3 require the kergen extra. "
                        "Install with:  pip install cvhmax[kergen]"
                    ) from None

                # Generator order M = self.order + 1 (SSM state dimension)
                gen = make_kernel(self.nple)
                K = gen.create_K_hat(
                    tau_c,
                    sigma_c,
                    rho_c,
                    omega_c,
                )
                K = K + jnp.eye(self.nple, dtype=K.dtype) * s_c

        return K.astype(_kernel_complex_dtype(output_dtype))

    @property
    def nple(self) -> int:
        return self.order + 1

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
        compute_dtype = _kernel_compute_dtype()
        output_dtype = _kernel_output_dtype(tau, self.sigma, self.rho, self.omega, self.s)

        Kt = self.K(tau, compute_dtype=compute_dtype, output_dtype=compute_dtype)
        K0 = self.K(0.0, compute_dtype=compute_dtype, output_dtype=compute_dtype)
        A = conjtrans(jnp.linalg.solve(conjtrans(K0), conjtrans(Kt)))  # K(t)K(0)^-1
        return A.astype(_kernel_complex_dtype(output_dtype))

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
        compute_dtype = _kernel_compute_dtype()
        output_dtype = _kernel_output_dtype(tau, self.sigma, self.rho, self.omega, self.s)

        Kt = self.K(tau, compute_dtype=compute_dtype, output_dtype=compute_dtype)
        K0 = self.K(0.0, compute_dtype=compute_dtype, output_dtype=compute_dtype)
        Q = K0 - Kt @ jnp.linalg.solve(K0, conjtrans(Kt))  # K(0) - K(t) K(0)^-1 K(t)'
        jitter = jnp.maximum(
            jnp.asarray(self.s, dtype=Q.dtype), jnp.asarray(EPS, dtype=Q.dtype)
        )
        Q = _stabilize_covariance(Q, jitter=jitter, output_dtype=output_dtype)
        return Q.astype(_kernel_complex_dtype(output_dtype))

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
        compute_dtype = _kernel_compute_dtype()
        output_dtype = _kernel_output_dtype(tau, self.sigma, self.rho, self.omega, self.s)

        Kt = self.K(tau, compute_dtype=compute_dtype, output_dtype=compute_dtype)
        K0 = self.K(0.0, compute_dtype=compute_dtype, output_dtype=compute_dtype)
        A = conjtrans(jnp.linalg.solve(conjtrans(K0), Kt))  # K(t)'K(0)^-1
        return A.astype(_kernel_complex_dtype(output_dtype))

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
        compute_dtype = _kernel_compute_dtype()
        output_dtype = _kernel_output_dtype(tau, self.sigma, self.rho, self.omega, self.s)

        Kt = self.K(tau, compute_dtype=compute_dtype, output_dtype=compute_dtype)
        K0 = self.K(0.0, compute_dtype=compute_dtype, output_dtype=compute_dtype)
        Q = K0 - conjtrans(Kt) @ jnp.linalg.solve(K0, Kt)  # K(0) - K(t)' K(0)^-1 K(t)
        jitter = jnp.maximum(
            jnp.asarray(self.s, dtype=Q.dtype), jnp.asarray(EPS, dtype=Q.dtype)
        )
        Q = _stabilize_covariance(Q, jitter=jitter, output_dtype=output_dtype)
        return Q.astype(_kernel_complex_dtype(output_dtype))

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

    if order == 0:
        K = Ks0(tau_c, sigma_c, rho_c, omega_c)
    elif order == 1:
        K = Ks1(tau_c, sigma_c, rho_c, omega_c)
    elif order == 2:
        K = Ks2(tau_c, sigma_c, rho_c, omega_c)
    else:
        try:
            from .kernel_generator import make_kernel
        except ImportError:
            raise ImportError(
                "Orders >= 3 require the kergen extra. "
                "Install with:  pip install cvhmax[kergen]"
            ) from None

        # Generator order M = order + 1 (SSM state dimension)
        gen = make_kernel(order + 1)
        K = gen.create_K_hat(
            tau_c,
            sigma_c,
            rho_c,
            omega_c,
        )

    return K.astype(_kernel_complex_dtype(output_dtype))


def Af(kernelparam, tau):
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
    compute_dtype = _kernel_compute_dtype()
    output_dtype = _kernel_output_dtype(tau, sigma, rho, omega)

    Kt = Ks(kernelparam, tau, compute_dtype=compute_dtype, output_dtype=compute_dtype)
    K0 = Ks(kernelparam, 0.0, compute_dtype=compute_dtype, output_dtype=compute_dtype)
    A = conjtrans(jnp.linalg.solve(conjtrans(K0), conjtrans(Kt)))  # K(t)K(0)^-1
    return A.astype(_kernel_complex_dtype(output_dtype))


def Qf(kernelparam, tau):
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
    compute_dtype = _kernel_compute_dtype()
    output_dtype = _kernel_output_dtype(tau, sigma, rho, omega)

    Kt = Ks(kernelparam, tau, compute_dtype=compute_dtype, output_dtype=compute_dtype)
    K0 = Ks(kernelparam, 0.0, compute_dtype=compute_dtype, output_dtype=compute_dtype)
    Q = K0 - Kt @ jnp.linalg.solve(K0, conjtrans(Kt))  # K(0) - K(t) K(0)^-1 K(t)'
    jitter = jnp.asarray(EPS, dtype=Q.dtype)
    Q = _stabilize_covariance(Q, jitter=jitter, output_dtype=output_dtype)
    return Q.astype(_kernel_complex_dtype(output_dtype))


def Ab(kernelparam, tau):
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
    compute_dtype = _kernel_compute_dtype()
    output_dtype = _kernel_output_dtype(tau, sigma, rho, omega)

    Kt = Ks(kernelparam, tau, compute_dtype=compute_dtype, output_dtype=compute_dtype)
    K0 = Ks(kernelparam, 0.0, compute_dtype=compute_dtype, output_dtype=compute_dtype)
    A = conjtrans(jnp.linalg.solve(conjtrans(K0), Kt))  # K(t)'K(0)^-1
    return A.astype(_kernel_complex_dtype(output_dtype))


def Qb(kernelparam, tau):
    """
    Backward dynamics state noise covariance
    """
    sigma, rho, omega, _ = itemgetter("sigma", "rho", "omega", "order")(kernelparam)
    compute_dtype = _kernel_compute_dtype()
    output_dtype = _kernel_output_dtype(tau, sigma, rho, omega)

    Kt = Ks(kernelparam, tau, compute_dtype=compute_dtype, output_dtype=compute_dtype)
    K0 = Ks(kernelparam, 0.0, compute_dtype=compute_dtype, output_dtype=compute_dtype)
    Q = K0 - conjtrans(Kt) @ jnp.linalg.solve(K0, Kt)  # K(0) - K(t)' K(0)^-1 K(t)
    jitter = jnp.asarray(EPS, dtype=Q.dtype)
    Q = _stabilize_covariance(Q, jitter=jitter, output_dtype=output_dtype)
    return Q.astype(_kernel_complex_dtype(output_dtype))


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


def spectral_density(kernel_spec: Dict, freq):
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
