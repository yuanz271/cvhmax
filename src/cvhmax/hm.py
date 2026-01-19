"""
Hida-Matern Kernel
"""

# Only polynomial of integer order p for Matern kernels
# 1/2 -> p=0
# 3/2 -> p=1
# 5/2 -> p=2
# etc.
from operator import itemgetter
from dataclasses import dataclass
from functools import cached_property, partial
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np

from cvhmax.utils import gamma

from .utils import conjtrans

# TODO: see sympy2jax, equinox
# NOTE: cos(x) = (exp(j * x) + exp(-j * x)) / 2 = Real[exp(j * x)]


def matern(tau: float, *, rho: float, order: int):
    """Evaluate the Matérn kernel with the requested smoothness.

    Parameters
    ----------
    tau : float
        Time lag at which the kernel is evaluated.
    rho : float
        Length scale of the kernel.
    order : int
        Smoothness order corresponding to the Matérn parameter.

    Raises
    ------
    NotImplementedError
        Raised while the general Matérn evaluator is pending implementation.
    """
    raise NotImplementedError("General Matérn kernel evaluation is not implemented.")


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
        Complex-valued covariance for the requested lag.
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


@dataclass
class HidaMatern:
    """Hida–Matérn kernel parameterised as a linear Gaussian SSM.

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
    """

    sigma: float = 1.0
    rho: float = 1.0
    omega: float = 0.0
    order: int = 0
    s: float = 1e-5

    def cov(self, tau=0.0):
        raise NotImplementedError

    def K(self, tau=0.0):
        """Return the state-space covariance block at lag ``tau``.

        Parameters
        ----------
        tau : float, default=0.0
            Time lag at which the block is evaluated.

        Returns
        -------
        Array
            Complex state covariance for the requested lag.

        Raises
        ------
        NotImplementedError
            Raised when the requested order lacks a closed-form implementation.
        """
        # TODO: confusing with covariance matrix
        # somehow not decorable by cache or cached_property
        if self.order == 0:
            K = Ks0(tau, self.sigma, self.rho, self.omega) + jnp.eye(self.nple) * self.s
        else:
            raise NotImplementedError

        return K

    @cached_property
    def nple(self):
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
        Kt = self.K(tau)
        K0 = self.K()
        A = conjtrans(jnp.linalg.solve(conjtrans(K0), conjtrans(Kt)))  # K(t)K(0)^-1
        return A

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
        Kt = self.K(tau)
        K0 = self.K()
        Q = K0 - Kt @ jnp.linalg.solve(K0, conjtrans(Kt))  # K(0) - K(t) K(0)^-1 K(t)'
        return Q

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
        Kt = self.K(tau)
        K0 = self.K()
        A = conjtrans(jnp.linalg.solve(conjtrans(K0), Kt))  # K(t)'K(0)^-1
        return A

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
        Kt = self.K(tau)
        K0 = self.K()
        Q = K0 - conjtrans(Kt) @ jnp.linalg.solve(K0, Kt)  # K(0) - K(t)' K(0)^-1 K(t)
        return Q

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


def Ks(kernelparam, tau):
    """Look up the complex-valued HM state covariance block.

    Parameters
    ----------
    kernelparam : dict
        Kernel hyperparameters containing `sigma`, `rho`, `omega`, and `order`.
    tau : float
        Time lag at which the block is evaluated.

    Returns
    -------
    Array
        Complex state covariance block.

    Raises
    ------
    NotImplementedError
        Raised for unsupported Matérn orders.
    """
    sigma, rho, omega, order = itemgetter("sigma", "rho", "omega", "order")(kernelparam)
    if order == 0:
        return Ks0(tau, sigma, rho, omega)
    elif order == 1:
        return Ks1(tau, sigma, rho, omega)
    else:
        raise NotImplementedError


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
    Kt = Ks(kernelparam, tau)
    K0 = Ks(kernelparam, 0.0)
    A = conjtrans(jnp.linalg.solve(conjtrans(K0), conjtrans(Kt)))  # K(t)K(0)^-1
    return A


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
    Kt = Ks(kernelparam, tau)
    K0 = Ks(kernelparam, 0.0)
    Q = K0 - Kt @ jnp.linalg.solve(K0, conjtrans(Kt))  # K(0) - K(t) K(0)^-1 K(t)'
    return Q


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
    Kt = Ks(kernelparam, tau)
    K0 = Ks(kernelparam, 0.0)
    A = conjtrans(jnp.linalg.solve(conjtrans(K0), Kt))  # K(t)'K(0)^-1
    return A


def Qb(kernelparam, tau):
    """
    Backward dynamics state noise covariance
    """
    Kt = Ks(kernelparam, tau)
    K0 = Ks(kernelparam, 0.0)
    Q = K0 - conjtrans(Kt) @ jnp.linalg.solve(K0, Kt)  # K(0) - K(t)' K(0)^-1 K(t)
    return Q


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
