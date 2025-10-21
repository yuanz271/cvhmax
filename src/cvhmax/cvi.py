from abc import abstractmethod
from collections.abc import Callable
from functools import partial
from typing import ClassVar, override
from sklearn.decomposition import FactorAnalysis

from jax import Array, lax, numpy as jnp, vmap
from jax.numpy.linalg import inv, solve, multi_dot
from jax.scipy.linalg import cho_factor, cho_solve
from equinox import Module

from cvhmax.utils import ridge_estimate

from .utils import filter_array, trial_info_repr, norm_loading, lbfgs_solve


TAU = 1e-6
MAX_LOGRATE = 7.0


def fa_init(ys, n_components, random_state):
    """Run a Factor Analysis warm start for CVI parameters.

    Parameters
    ----------
    ys : Array
        Observations flattened to `(time * trials, features)`.
    n_components : int
        Number of latent factors to extract.
    random_state : int
        Seed forwarded to `sklearn` initialisation.

    Returns
    -------
    tuple[Array, Array, Array]
        Mean trajectories, loading matrix, and bias vector from FA.
    """
    fa = FactorAnalysis(n_components=n_components, random_state=random_state)
    fa.fit(ys)
    ms = jnp.array(fa.transform(ys))
    C = jnp.array(fa.components_.T)
    d = jnp.array(fa.mean_)

    return ms, C, d


class Params(Module):
    """Container of CVI readout parameters."""

    C: Array
    d: Array
    R: Array
    M: Array

    def loading(self) -> Array:
        """Column-normalised loading matrix.

        Returns
        -------
        Array
            Loading matrix with unit-norm columns.
        """
        return norm_loading(self.C)

    def lmask(self) -> Array:
        """Latent mask treated as a constant during differentiation.

        Returns
        -------
        Array
            Mask with gradients stopped.
        """
        return lax.stop_gradient(self.M)


class CVI:
    """Base class for conjugate variational inference readouts."""

    registry: ClassVar[dict] = dict()

    def __init_subclass__(cls, *args, **kwargs):
        """Register subclasses for lookup by likelihood name."""
        super().__init_subclass__(*args, **kwargs)
        CVI.registry[cls.__name__] = cls

    @classmethod
    def infer(
        cls,
        params: Params,
        j: Array,
        J: Array,
        y: Array,
        ymask: Array,
        z0: Array,
        Z0: Array,
        smooth_fun: Callable,
        smooth_args: tuple,
        cvi_iter: int,
        lr: float,
    ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
        """Run CVI smoothing iterations for the given pseudo-observations.

        Parameters
        ----------
        params : Params
            Current readout parameter state.
        j, J : Array
            Initial pseudo-observation natural parameters.
        y : Array
            Observation tensor.
        ymask : Array
            Observation mask aligned with `y`.
        z0, Z0 : Array
            Initial latent information parameters.
        smooth_fun : Callable
            Filtering routine returning smoothed information tuples.
        smooth_args : tuple
            Additional arguments forwarded to `smooth_fun`.
        cvi_iter : int
            Number of CVI iterations to perform.
        lr : float
            Pseudo-observation learning rate.

        Returns
        -------
        tuple[tuple[Array, Array], tuple[Array, Array]]
            Smoothed latent information along with the updated pseudo-observations.
        """
        smooth_batch = vmap(
            lambda jk, Jk, zk0, Zk0: smooth_fun(jk, Jk, zk0, Zk0, *smooth_args)
        )

        def step(i, carry) -> tuple[Array, Array]:
            j, J = carry
            z, Z = smooth_batch(j, J, z0, Z0)
            j, J = cls.update_pseudo(params, y, ymask, z, Z, j, J, lr)
            return j, J

        j, J = lax.fori_loop(0, cvi_iter, step, (j, J))
        z, Z = vmap(
            lambda jk, Jk, zk0, Zk0: smooth_fun(jk, Jk, zk0, Zk0, *smooth_args)
        )(j, J, z0, Z0)

        return (z, Z), (j, J)

    @classmethod
    @abstractmethod
    def update_readout(cls, *args, **kwargs) -> tuple[Params, float]:
        """Update readout parameters given latent statistics."""

    @classmethod
    @abstractmethod
    def update_pseudo(
        cls,
        params: Params,
        y: Array,
        ymask: Array,
        z: Array,
        Z: Array,
        j: Array,
        J: Array,
        lr: float,
    ) -> tuple[Array, Array]:
        """Produce new pseudo-observations conditioned on the latest latents."""

    @classmethod
    @abstractmethod
    def initialize_info(
        cls, params: Params, y: Array, ymask: Array, A: Array, Q: Array
    ) -> tuple[Array, Array]:
        """Initialise pseudo-observation natural parameters."""

    @classmethod
    @abstractmethod
    def initialize_params(cls, *args, **kwargs) -> Params:
        """Create the initial readout parameter state."""


class Gaussian(CVI):
    """Linear-Gaussian readout with closed-form updates."""

    @classmethod
    @override
    def infer(
        cls,
        params: Params,
        j: Array,
        J: Array,
        y: Array,
        ymask: Array,
        z0: Array,
        Z0: Array,
        smooth_fun: Callable,
        smooth_args: tuple,
        cvi_iter: int,
        lr: float,
    ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
        """Use a single CVI iteration thanks to conjugacy.

        Returns
        -------
        tuple[tuple[Array, Array], tuple[Array, Array]]
            Smoothed latents and pseudo-observations.
        """
        return CVI.infer(params, j, J, y, ymask, z0, Z0, smooth_fun, smooth_args, 1, lr)

    @classmethod
    @override
    def update_pseudo(
        cls,
        params: Params,
        y: Array,
        ymask: Array,
        z: Array,
        Z: Array,
        j: Array,
        J: Array,
        lr: float,
    ) -> tuple[Array, Array]:
        """Return unmodified pseudo-observations for Gaussian readouts.

        Returns
        -------
        tuple[Array, Array]
            Same pseudo-observation parameters `(j, J)`.
        """
        return j, J

    @classmethod
    @override
    def initialize_info(
        cls, params: Params, y: Array, ymask: Array, A: Array, Q: Array
    ) -> tuple[Array, Array]:
        """Compute Gaussian observation information from the readout.

        Parameters
        ----------
        params : Params
            Current readout parameter state.
        y : Array
            Observation tensor.
        ymask : Array
            Observation mask aligned with `y`.
        A : Array
            Forward transition matrix (unused, keeps API symmetry).
        Q : Array
            Forward process noise covariance (unused, keeps API symmetry).

        Returns
        -------
        tuple[Array, Array]
            Observation information vectors and matrices.
        """
        C = params.loading()
        d = params.d
        R: Array = params.R
        M = params.lmask()

        H = C @ M

        return vmap(partial(trial_info_repr, C=H, d=d, R=R))(y, ymask)

    @classmethod
    @override
    def update_readout(
        cls, params: Params, y: Array, ymask: Array, m: Array, P: Array
    ) -> tuple[Params, float]:
        """Perform a ridge regression update of the Gaussian readout.

        Parameters
        ----------
        params : Params
            Current readout parameter state.
        y : Array
            Observation tensor.
        ymask : Array
            Observation mask aligned with `y`.
        m : Array
            Posterior means.
        P : Array
            Posterior covariances.

        Returns
        -------
        tuple[Params, float]
            Updated parameters and the (unused) negative log-likelihood proxy.
        """
        y = filter_array(y, ymask)
        m = filter_array(m, ymask)
        C, d, R = ridge_estimate(y, m, P)
        params = Params(C=C, d=d, R=R, M=params.M)
        return params, jnp.nan

    @classmethod
    @override
    def initialize_params(
        cls,
        y: Array,
        ymask: Array,
        n_factors: int,
        lmask: Array,
        *,
        random_state: int,
    ) -> Params:
        """Initialise Gaussian readout via FA.

        Parameters
        ----------
        y : Array
            Observation tensor.
        ymask : Array
            Observation mask aligned with `y`.
        n_factors : int
            Number of latent factors.
        lmask : Array
            Latent mask mapping latent components to outputs.
        random_state : int
            Seed used for factor analysis.

        Returns
        -------
        Params
            Initial Gaussian readout parameters.
        """
        y = filter_array(y, ymask)
        _, C, d = fa_init(y, n_factors, random_state)

        return Params(C=C, d=d, R=jnp.zeros(y.shape[-1]), M=lmask)


def poisson_trial_nell(
    params: tuple[Array, Array],
    y: Array,
    m: Array,
    V: Array,
    gamma: float = 10.0,
) -> float:
    """Average negative expected log-likelihood across trials.

    Parameters
    ----------
    params : tuple[Array, Array]
        Loading matrix and bias vector `(C, d)`.
    y : Array
        Observed counts with shape `(trial, time, obs_dim)`.
    m : Array
        Posterior means with matching leading dimensions.
    V : Array
        Posterior covariances.
    gamma : float, default=10.0
        Strength of loading matrix L2 regularisation.

    Returns
    -------
    float
        Trial-averaged negative expected log-likelihood.
    """
    C, d = params

    def bin_nell(y_t, m_t, V_t):
        lin = C @ m_t + d
        quad = jnp.einsum(
            "ni,in->n", C, V_t @ C.T
        )  # (Y, Z) (T, Z, Z) (Z, Y) -> (T, Y, Y)
        eta = jnp.minimum(lin + 0.5 * quad, MAX_LOGRATE)
        lam = jnp.exp(eta)
        return jnp.sum(lam - eta * y_t, axis=-1)

    C_reg = gamma * jnp.linalg.norm(C) / y.shape[0]
    bin_nells = vmap(bin_nell)(y, m, V)

    return jnp.mean(bin_nells) + C_reg


def poisson_cvi_bin_stats(
    z: Array,
    Z: Array,
    y: Array,
    ymask: Array,
    H: Array,
    d: Array,
) -> tuple[Array, Array]:
    """Poisson CVI gradients for a single bin.

    Parameters
    ----------
    z : Array
        Information vector.
    Z : Array
        Information matrix.
    y : Array
        Observed counts for the bin.
    ymask : Array
        Boolean mask marking valid observations.
    H : Array
        Effective observation matrix.
    d : Array
        Bias term for the exponential link.

    Returns
    -------
    tuple[Array, Array]
        Gradients of the first and second natural parameters.

    Notes
    -----
    The natural parameters `(z, Z)` correspond to Gaussian statistics via
    `m = -0.5 * Z^{-1} z` and `V = -0.5 * Z^{-1}`.
    """
    U, s, V = jnp.linalg.svd(Z)
    Z = multi_dot((U, jnp.diag(s + TAU), U.T))

    Zcho = cho_factor(Z)
    m = -0.5 * cho_solve(Zcho, z)  # Vj

    lin = H @ m + d
    quad = jnp.einsum("nl, ln -> n", H, -0.5 * cho_solve(Zcho, H.mT))  # CVC'
    eta = jnp.minimum(lin + 0.5 * quad, MAX_LOGRATE)
    lam = jnp.exp(eta)

    grad_m = (y - lam) @ H
    grad_V = -0.5 * jnp.einsum("ni, n, nj -> ij", H, lam, H)

    K = -2 * grad_V
    k = grad_m - 2 * grad_V @ m

    k = jnp.where(jnp.expand_dims(ymask, -1), k, 0)
    K = jnp.where(jnp.expand_dims(ymask, (-2, -1)), K, 0)

    return k, K


class Poisson(CVI):
    """CVI readout for Poisson observations with exponential link."""

    @classmethod
    def initialize_info(
        cls,
        params: Params,
        y: Array,
        ymask: Array,
        A: Array,
        Q: Array,
    ) -> tuple[Array, Array]:
        """Initialise Poisson pseudo-observations by filtering forward.

        Parameters
        ----------
        params : Params
            Current readout parameter state.
        y : Array
            Observation tensor.
        ymask : Array
            Observation mask aligned with `y`.
        A : Array
            Forward transition matrix.
        Q : Array
            Forward process noise covariance.

        Returns
        -------
        tuple[Array, Array]
            Pseudo-observation vectors and matrices per time bin.
        """
        C = params.loading()
        M = params.lmask()
        H = C @ M
        d = params.d

        d_z = Q.shape[0]
        z0 = jnp.zeros(d_z)
        Z0 = P = inv(Q)

        A = A + 1e-3 * jnp.eye(d_z)  # ill-condition

        def forward(
            carry: tuple[Array, Array], ys: tuple[Array, Array]
        ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
            ztm1, Ztm1 = carry
            yt, ytmask = ys

            # predict
            M = solve(A.T, solve(A.T, Ztm1.T).T)
            G = solve((M + P).T, M.T).T
            eye = jnp.eye(G.shape[0])
            L = eye - G
            Zp = multi_dot((L, M, L.T)) + multi_dot((G, P, G.T))
            zp = L @ solve(A.T, ztm1)

            j, J = poisson_cvi_bin_stats(zp, Zp, yt, ytmask, H, d)

            zt = zp + j
            Zt = Zp + J
            Zt = 0.5 * (Zt + Zt.mT)

            return (zt, Zt), (j, J)

        _, (j, J) = lax.scan(forward, (z0, Z0), (y, ymask))

        return j, J

    @classmethod
    def update_readout(
        cls,
        params: Params,
        y: Array,
        ymask: Array,
        m: Array,
        V: Array,
    ) -> tuple[Params, float]:
        """Optimise Poisson readout parameters via LBFGS.

        Parameters
        ----------
        params : Params
            Current readout parameter state.
        y : Array
            Observation tensor.
        ymask : Array
            Observation mask aligned with `y`.
        m : Array
            Posterior means.
        V : Array
            Posterior covariances.

        Returns
        -------
        tuple[Params, float]
            Updated parameters and the negative expected log-likelihood.
        """
        C = params.loading()
        d = params.d
        R = params.R

        y = filter_array(y, ymask)
        m = filter_array(m, ymask)
        V = filter_array(V, ymask)

        C, d = lbfgs_solve((C, d), partial(poisson_trial_nell, y=y, m=m, V=V))  # type: ignore

        nell = poisson_trial_nell((C, d), y=y, m=m, V=V, gamma=0.0)  # type: ignore
        return Params(C=C, d=d, R=R, M=params.M), nell  # type: ignore

    @classmethod
    def update_pseudo(
        cls,
        params: Params,
        y: Array,
        ymask: Array,
        z: Array,
        Z: Array,
        j: Array,
        J: Array,
        lr: float,
    ) -> tuple[Array, Array]:
        """Update pseudo-observations using Poisson CVI gradients.

        Parameters
        ----------
        params : Params
            Current readout parameter state.
        y : Array
            Observation tensor.
        ymask : Array
            Observation mask aligned with `y`.
        z : Array
            Posterior information vectors.
        Z : Array
            Posterior information matrices.
        j : Array
            Current pseudo-observation vectors.
        J : Array
            Current pseudo-observation matrices.
        lr : float
            Learning rate for the convex combination.

        Returns
        -------
        tuple[Array, Array]
            Updated pseudo-observation vectors and matrices.
        """
        C = params.loading()
        M = params.lmask()
        H = C @ M
        d = params.d
        # print(f"{z.shape=} {Z.shape=} {y.shape=}, {ymask.shape=}")
        k, K = vmap(vmap(partial(poisson_cvi_bin_stats, H=H, d=d)))(
            z, Z, y, ymask
        )  # session

        j = (1 - lr) * j + lr * k
        J = (1 - lr) * J + lr * K

        return j, J

    @classmethod
    def initialize_params(
        cls,
        y: Array,
        ymask: Array,
        n_factors: int,
        lmask: Array,
        *,
        random_state: int,
    ) -> Params:
        """Initialise Poisson readout via FA followed by Whittle optimisation.

        Parameters
        ----------
        y : Array
            Observation tensor.
        ymask : Array
            Observation mask aligned with `y`.
        n_factors : int
            Number of latent factors.
        lmask : Array
            Latent mask mapping latent components to outputs.
        random_state : int
            Seed used for factor analysis.

        Returns
        -------
        Params
            Initial Poisson readout parameters.
        """
        y = filter_array(y, ymask)
        m, C, d = fa_init(y, n_factors, random_state)

        n_bins, n_obs = y.shape
        V = jnp.zeros((n_bins, n_factors, n_factors))  # dummpy variance

        C, d = lbfgs_solve((C, d), partial(poisson_trial_nell, y=y, m=m, V=V))  # type: ignore

        return Params(C=C, d=d, R=None, M=lmask)  # type: ignore
