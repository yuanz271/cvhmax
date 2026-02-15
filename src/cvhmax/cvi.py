from abc import abstractmethod
from collections.abc import Callable
from functools import partial
from typing import ClassVar, override

from sklearn.decomposition import FactorAnalysis

from jax import Array, lax, numpy as jnp, vmap
from jax.numpy.linalg import inv, solve, multi_dot
from jax.scipy.linalg import cho_factor, cho_solve
from equinox import Module

from .utils import (
    filter_array,
    trial_info_repr,
    norm_loading,
    lbfgs_solve,
    ridge_estimate,
)


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
    """Container of CVI readout parameters.

    Attributes
    ----------
    C : Array
        Loading matrix with shape `(obs_dim (N), latent_dim (K))`.
    d : Array
        Bias vector with shape `(obs_dim (N),)`.
    R : Array
        Observation covariance. Typically `(obs_dim (N), obs_dim (N))`; some
        initializers may use a vector of variances.
    M : Array
        Selection mask mapping latent components to state coordinates,
        shape `(latent_dim (K), state_dim (L))`.
    """

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
    """Base class for conjugate variational inference readouts.

    Subclasses register by name in `CVI.registry` for lookup via `likelihood`.
    """

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
        valid_y: Array,
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
            Initial pseudo-observation natural parameters with shapes
            `(trials, time, state_dim (L))` and `(trials, time, state_dim (L), state_dim (L))`.
        y : Array
            Observation tensor shaped `(trials, time, obs_dim)`.
        valid_y : Array
            Observation mask aligned with `y`, typically `(trials, time)`.
        z0, Z0 : Array
            Initial state information parameters shaped `(trials, state_dim (L))`
            and `(trials, state_dim (L), state_dim (L))`.
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
            j, J = cls.update_pseudo(params, y, valid_y, z, Z, j, J, lr)
            return j, J

        j, J = lax.fori_loop(0, cvi_iter, step, (j, J))
        z, Z = vmap(
            lambda jk, Jk, zk0, Zk0: smooth_fun(jk, Jk, zk0, Zk0, *smooth_args)
        )(j, J, z0, Z0)

        return (z, Z), (j, J)

    @classmethod
    @abstractmethod
    def update_readout(cls, *args, **kwargs) -> tuple[Params, float]:
        """Update readout parameters given latent statistics.

        Parameters
        ----------
        params : Params
            Current readout parameter state.
        y : Array
            Observations shaped `(trials, time, obs_dim (N))`.
        valid_y : Array
            Observation mask aligned with `y`.
        m : Array
            Posterior means shaped `(trials, time, latent_dim (K))`.
        V : Array
            Posterior covariances shaped `(trials, time, latent_dim (K), latent_dim (K))`.

        Returns
        -------
        tuple[Params, float]
            Updated parameter state and an objective value.
        """

    @classmethod
    @abstractmethod
    def update_pseudo(
        cls,
        params: Params,
        y: Array,
        valid_y: Array,
        z: Array,
        Z: Array,
        j: Array,
        J: Array,
        lr: float,
    ) -> tuple[Array, Array]:
        """Produce new pseudo-observations conditioned on the latest latents.

        Parameters
        ----------
        params : Params
            Current readout parameter state.
        y : Array
            Observations shaped `(trials, time, obs_dim (N))`.
        valid_y : Array
            Observation mask aligned with `y`.
        z : Array
            Posterior information vectors shaped `(trials, time, state_dim (L))`.
        Z : Array
            Posterior information matrices shaped `(trials, time, state_dim (L), state_dim (L))`.
        j : Array
            Pseudo-observation vectors shaped `(trials, time, state_dim (L))`.
        J : Array
            Pseudo-observation matrices shaped `(trials, time, state_dim (L), state_dim (L))`.
        lr : float
            Pseudo-observation learning rate.

        Returns
        -------
        tuple[Array, Array]
            Updated pseudo-observation parameters `(j, J)`.
        """

    @classmethod
    @abstractmethod
    def initialize_info(
        cls, params: Params, y: Array, valid_y: Array, A: Array, Q: Array
    ) -> tuple[Array, Array]:
        """Initialise pseudo-observation natural parameters.

        Parameters
        ----------
        params : Params
            Current readout parameter state.
        y : Array
            Observations shaped `(time, obs_dim (N))`.
        valid_y : Array
            Observation mask, shape `(time,)`.
        A : Array
            Forward transition matrix shaped `(state_dim (L), state_dim (L))`.
        Q : Array
            Forward process noise covariance shaped `(state_dim (L), state_dim (L))`.

        Returns
        -------
        tuple[Array, Array]
            Pseudo-observation vectors and matrices with shapes
            `(time, state_dim (L))` and `(time, state_dim (L), state_dim (L))`.
        """

    @classmethod
    @abstractmethod
    def initialize_params(
        cls,
        y: Array,
        valid_y: Array,
        n_factors: int,
        lmask: Array,
        *,
        random_state: int,
        params: Params | None = None,
    ) -> Params:
        """Create the initial readout parameter state.

        When ``params`` is provided, return it directly and skip
        initialisation.  This allows callers to supply pre-set
        parameters (e.g. the true readout in a simulation).

        Parameters
        ----------
        y : Array
            Observations shaped `(trials, time, obs_dim)`.
        valid_y : Array
            Observation mask aligned with `y`.
        n_factors : int
            Number of latent factors to initialize.
        lmask : Array
            Latent mask mapping components to state-space coordinates.
        random_state : int
            Seed for any stochastic initialisation.
        params : Params or None, optional
            If provided, returned as-is (skipping initialisation).

        Returns
        -------
        Params
            Initial readout parameter state.
        """


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
        valid_y: Array,
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
        return super().infer(
            params, j, J, y, valid_y, z0, Z0, smooth_fun, smooth_args, 1, lr
        )

    @classmethod
    @override
    def update_pseudo(
        cls,
        params: Params,
        y: Array,
        valid_y: Array,
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
        cls, params: Params, y: Array, valid_y: Array, A: Array, Q: Array
    ) -> tuple[Array, Array]:
        """Compute Gaussian observation information from the readout.

        Parameters
        ----------
        params : Params
            Current readout parameter state.
        y : Array
            Observation tensor shaped `(time, obs_dim)`.
        valid_y : Array
            Observation mask, shape `(time,)`.
        A : Array
            Forward transition matrix (unused, keeps API symmetry).
        Q : Array
            Forward process noise covariance (unused, keeps API symmetry).

        Returns
        -------
        tuple[Array, Array]
            Observation information vectors and matrices with shapes
            `(time, state_dim (L))` and `(time, state_dim (L), state_dim (L))`.
        """
        C = params.loading()
        d = params.d
        R: Array = params.R
        M = params.lmask()

        H = C @ M

        return trial_info_repr(y, valid_y, H, d, R)

    @classmethod
    @override
    def update_readout(
        cls, params: Params, y: Array, valid_y: Array, m: Array, P: Array
    ) -> tuple[Params, float]:
        """Perform a ridge regression update of the Gaussian readout.

        Parameters
        ----------
        params : Params
            Current readout parameter state.
        y : Array
            Observation tensor shaped `(trials, time, obs_dim (N))`.
        valid_y : Array
            Observation mask aligned with `y`.
        m : Array
            Posterior means shaped `(trials, time, latent_dim (K))`.
        P : Array
            Posterior covariances shaped `(trials, time, latent_dim (K), latent_dim (K))`.

        Returns
        -------
        tuple[Params, float]
            Updated parameters and the (unused) negative log-likelihood proxy.
        """
        y = y.reshape(-1, y.shape[-1])
        valid_y = valid_y.ravel()
        m = m.reshape(-1, m.shape[-1])
        C, d, R = ridge_estimate(y, valid_y, m, P)
        params = Params(C=C, d=d, R=R, M=params.M)
        return params, jnp.nan

    @classmethod
    @override
    def initialize_params(
        cls,
        y: Array,
        valid_y: Array,
        n_factors: int,
        lmask: Array,
        *,
        random_state: int,
        params: Params | None = None,
    ) -> Params:
        """Initialise Gaussian readout via FA.

        Parameters
        ----------
        y : Array
            Observation tensor shaped `(trials, time, obs_dim)`.
        valid_y : Array
            Observation mask aligned with `y`.
        n_factors : int
            Number of latent factors.
        lmask : Array
            Latent mask mapping latent components to outputs.
        random_state : int
            Seed used for factor analysis.
        params : Params or None, optional
            If provided, returned as-is (skipping initialisation).

        Returns
        -------
        Params
            Initial Gaussian readout parameters.
        """
        if params is not None:
            return params

        y = filter_array(y, valid_y)
        _, C, d = fa_init(y, n_factors, random_state)

        return Params(C=C, d=d, R=jnp.eye(y.shape[-1]), M=lmask)


def poisson_trial_nell(
    params: tuple[Array, Array],
    y: Array,
    valid_y: Array,
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
    valid_y : Array
        Observation mask aligned with `y`.
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
    n_valid_bins = jnp.sum(valid_y)

    def bin_nell(y_t, m_t, V_t):
        lin = C @ m_t + d
        quad = jnp.einsum(
            "ni,in->n", C, V_t @ C.T
        )  # (Y, Z) (T, Z, Z) (Z, Y) -> (T, Y, Y)
        eta = jnp.minimum(lin + 0.5 * quad, MAX_LOGRATE)
        lam = jnp.exp(eta)
        return jnp.sum(lam - eta * y_t, axis=-1)

    C_reg = gamma * jnp.linalg.norm(C) / n_valid_bins
    bin_nells = vmap(bin_nell)(y, m, V)
    bin_nells = jnp.where(valid_y, bin_nells, 0)

    return jnp.sum(bin_nells) / n_valid_bins + C_reg


def poisson_cvi_bin_stats(
    z: Array,
    Z: Array,
    y: Array,
    valid_y: Array,
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
    valid_y : Array
        Observation mask for one bin, shape ``()``. When zero, both ``k``
        and ``K`` are set to zero so the bin contributes no
        pseudo-observation gradient to the CVI update.
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
    The information parameters `(z, Z)` correspond to Gaussian statistics via
    `m = Z^{-1} z` and `V = Z^{-1}` where `Z = Σ⁻¹` and `z = Σ⁻¹ μ`.
    """
    U, s, V = jnp.linalg.svd(Z)
    Z = multi_dot((U, jnp.diag(s + TAU), U.T))

    Zcho = cho_factor(Z)
    m = cho_solve(Zcho, z)

    lin = H @ m + d
    quad = jnp.einsum("nl, ln -> n", H, cho_solve(Zcho, H.mT))  # CVC'
    eta = jnp.minimum(lin + 0.5 * quad, MAX_LOGRATE)
    lam = jnp.exp(eta)

    grad_m = (y - lam) @ H
    grad_V = -0.5 * jnp.einsum("ni, n, nj -> ij", H, lam, H)

    K = -2 * grad_V
    k = grad_m - 2 * grad_V @ m

    k = jnp.where(valid_y, k, 0)
    K = jnp.where(valid_y, K, 0)

    return k, K


class Poisson(CVI):
    """CVI readout for Poisson observations with exponential link."""

    @classmethod
    def initialize_info(
        cls,
        params: Params,
        y: Array,
        valid_y: Array,
        A: Array,
        Q: Array,
    ) -> tuple[Array, Array]:
        """Initialise Poisson pseudo-observations by filtering forward.

        Parameters
        ----------
        params : Params
            Current readout parameter state.
        y : Array
            Observation tensor shaped `(time, obs_dim (N))`.
        valid_y : Array
            Observation mask, shape `(time,)`.
        A : Array
            Forward transition matrix shaped `(state_dim (L), state_dim (L))`.
        Q : Array
            Forward process noise covariance shaped `(state_dim (L), state_dim (L))`.

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
            yt, valid_yt = ys

            # predict
            M = solve(A.T, solve(A.T, Ztm1.T).T)
            G = solve((M + P).T, M.T).T
            eye = jnp.eye(G.shape[0])
            L = eye - G
            Zp = multi_dot((L, M, L.T)) + multi_dot((G, P, G.T))
            zp = L @ solve(A.T, ztm1)

            j, J = poisson_cvi_bin_stats(zp, Zp, yt, valid_yt, H, d)

            zt = zp + j
            Zt = Zp + J
            Zt = 0.5 * (Zt + Zt.mT)

            return (zt, Zt), (j, J)

        _, (j, J) = lax.scan(forward, (z0, Z0), (y, valid_y))

        return j, J

    @classmethod
    def update_readout(
        cls,
        params: Params,
        y: Array,
        valid_y: Array,
        m: Array,
        V: Array,
    ) -> tuple[Params, float]:
        """Optimise Poisson readout parameters via LBFGS.

        Parameters
        ----------
        params : Params
            Current readout parameter state.
        y : Array
            Observation tensor shaped `(trials, time, obs_dim (N))`.
        valid_y : Array
            Observation mask aligned with `y`.
        m : Array
            Posterior means shaped `(trials, time, latent_dim (K))`.
        V : Array
            Posterior covariances shaped `(trials, time, latent_dim (K), latent_dim (K))`.

        Returns
        -------
        tuple[Params, float]
            Updated parameters and the negative expected log-likelihood.
        """
        C = params.loading()
        d = params.d
        R = params.R

        # y = filter_array(y, valid_y)
        # m = filter_array(m, valid_y)
        # V = filter_array(V, valid_y)

        y = y.reshape(-1, y.shape[-1])
        valid_y = valid_y.ravel()
        m = m.reshape(-1, m.shape[-1])
        V = V.reshape(-1, *V.shape[-2:])

        C, d = lbfgs_solve(
            (C, d), partial(poisson_trial_nell, y=y, valid_y=valid_y, m=m, V=V)
        )

        nell = poisson_trial_nell((C, d), y=y, valid_y=valid_y, m=m, V=V, gamma=0.0)
        return Params(C=C, d=d, R=R, M=params.M), nell

    @classmethod
    def update_pseudo(
        cls,
        params: Params,
        y: Array,
        valid_y: Array,
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
            Observation tensor shaped `(trials, time, obs_dim (N))`.
        valid_y : Array
            Observation mask aligned with `y`.
        z : Array
            Posterior information vectors shaped `(trials, time, state_dim (L))`.
        Z : Array
            Posterior information matrices shaped `(trials, time, state_dim (L), state_dim (L))`.
        j : Array
            Current pseudo-observation vectors shaped `(trials, time, state_dim (L))`.
        J : Array
            Current pseudo-observation matrices shaped `(trials, time, state_dim (L), state_dim (L))`.
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
        # print(f"{z.shape=} {Z.shape=} {y.shape=}, {valid_y.shape=}")
        k, K = vmap(vmap(partial(poisson_cvi_bin_stats, H=H, d=d)))(
            z, Z, y, valid_y
        )  # session

        j = (1 - lr) * j + lr * k
        J = (1 - lr) * J + lr * K

        return j, J

    @classmethod
    def initialize_params(
        cls,
        y: Array,
        valid_y: Array,
        n_factors: int,
        lmask: Array,
        *,
        random_state: int,
        params: Params | None = None,
    ) -> Params:
        """Initialise Poisson readout via FA followed by LBFGS optimisation.

        Parameters
        ----------
        y : Array
            Flattened observation tensor shaped `(trials * time, obs_dim)`.
        valid_y : Array
            Observation mask aligned with `y`.
        n_factors : int
            Number of latent factors.
        lmask : Array
            Latent mask mapping latent components to outputs.
        random_state : int
            Seed used for factor analysis.
        params : Params or None, optional
            If provided, returned as-is (skipping initialisation).

        Returns
        -------
        Params
            Initial Poisson readout parameters.
        """
        if params is not None:
            return params

        y = filter_array(y, valid_y)
        m, C, d = fa_init(y, n_factors, random_state)

        n_bins, n_obs = y.shape
        V = jnp.zeros((n_bins, n_factors, n_factors))  # dummy variance

        C, d = lbfgs_solve(
            (C, d),
            partial(poisson_trial_nell, y=y, valid_y=jnp.ones(y.shape[:1]), m=m, V=V),
        )

        return Params(C=C, d=d, R=None, M=lmask)
