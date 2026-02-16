from abc import abstractmethod
from functools import partial
from typing import ClassVar, override

from sklearn.decomposition import FactorAnalysis

from jax import Array, numpy as jnp, vmap
from jax.numpy.linalg import multi_dot
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

    This is a convenience container used by the built-in Gaussian and Poisson
    readouts.  Custom observation models may use any pytree-compatible
    structure — CVHM treats params as opaque.

    Attributes
    ----------
    C : Array
        Loading matrix with shape ``(obs_dim (N), latent_dim (K))``.
    d : Array
        Bias vector with shape ``(obs_dim (N),)``.
    R : Array | None
        Observation covariance. Typically ``(obs_dim (N), obs_dim (N))``.
        ``None`` for observation models that do not use it (e.g., Poisson).
    """

    C: Array
    d: Array
    R: Array | None

    def loading(self) -> Array:
        """Column-normalised loading matrix.

        Returns
        -------
        Array
            Loading matrix with unit-norm columns.
        """
        return norm_loading(self.C)


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
        params,
        y: Array,
        valid_y: Array,
        m: Array,
        V: Array,
        j: Array,
        J: Array,
        lr: float,
    ) -> tuple[Array, Array]:
        """Produce new pseudo-observations conditioned on the latest latent posterior.

        All arrays are in latent space.

        Parameters
        ----------
        params
            Current readout parameter state (opaque).
        y : Array
            Observations shaped ``(trials, time, obs_dim (N))``.
        valid_y : Array
            Observation mask aligned with ``y``.
        m : Array
            Posterior means shaped ``(trials, time, latent_dim (K))``.
        V : Array
            Posterior covariances shaped
            ``(trials, time, latent_dim (K), latent_dim (K))``.
        j : Array
            Pseudo-observation vectors shaped ``(trials, time, latent_dim (K))``.
        J : Array
            Pseudo-observation matrices shaped
            ``(trials, time, latent_dim (K), latent_dim (K))``.
        lr : float
            Pseudo-observation learning rate.

        Returns
        -------
        tuple[Array, Array]
            Updated pseudo-observation parameters ``(j, J)`` in latent space.
        """

    @classmethod
    @abstractmethod
    def initialize_info(
        cls, params, y: Array, valid_y: Array
    ) -> tuple[Array, Array]:
        """Initialise pseudo-observation natural parameters in latent space.

        Parameters
        ----------
        params
            Current readout parameter state (opaque).
        y : Array
            Observations shaped ``(time, obs_dim (N))``.
        valid_y : Array
            Observation mask, shape ``(time,)``.

        Returns
        -------
        tuple[Array, Array]
            Pseudo-observation vectors and matrices in latent space with
            shapes ``(time, latent_dim (K))`` and
            ``(time, latent_dim (K), latent_dim (K))``.
        """

    @classmethod
    @abstractmethod
    def initialize_params(
        cls,
        y: Array,
        valid_y: Array,
        n_factors: int,
        *,
        random_state: int,
        params=None,
    ):
        """Create the initial readout parameter state.

        When ``params`` is provided, return it directly and skip
        initialisation.  This allows callers to supply pre-set
        parameters (e.g. the true readout in a simulation).

        The returned params structure is opaque to CVHM — each CVI
        subclass may use any pytree-compatible container.

        Parameters
        ----------
        y : Array
            Observations shaped ``(trials, time, obs_dim)``.
        valid_y : Array
            Observation mask aligned with ``y``.
        n_factors : int
            Number of latent factors to initialize.
        random_state : int
            Seed for any stochastic initialisation.
        params : optional
            If provided, returned as-is (skipping initialisation).

        Returns
        -------
        params
            Initial readout parameter state.
        """


class Gaussian(CVI):
    """Linear-Gaussian readout with closed-form updates."""

    @classmethod
    @override
    def update_pseudo(
        cls,
        params: Params,
        y: Array,
        valid_y: Array,
        m: Array,
        V: Array,
        j: Array,
        J: Array,
        lr: float,
    ) -> tuple[Array, Array]:
        """Return unmodified pseudo-observations for Gaussian readouts.

        Gaussian readouts are conjugate so pseudo-observations are fully
        determined by ``initialize_info``.  All arrays are in latent space.

        Returns
        -------
        tuple[Array, Array]
            Same pseudo-observation parameters ``(j, J)``.
        """
        return j, J

    @classmethod
    @override
    def initialize_info(
        cls, params: Params, y: Array, valid_y: Array
    ) -> tuple[Array, Array]:
        """Compute Gaussian observation information in latent space.

        Parameters
        ----------
        params : Params
            Current readout parameter state.
        y : Array
            Observation tensor shaped ``(time, obs_dim)``.
        valid_y : Array
            Observation mask, shape ``(time,)``.

        Returns
        -------
        tuple[Array, Array]
            Observation information vectors and matrices in latent space
            with shapes ``(time, latent_dim (K))`` and
            ``(time, latent_dim (K), latent_dim (K))``.
        """
        C = params.loading()
        d = params.d
        R = params.R
        if R is None:
            raise ValueError("Gaussian readout requires a noise covariance R.")

        return trial_info_repr(y, valid_y, C, d, R)

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
        return Params(C=C, d=d, R=R), jnp.nan

    @classmethod
    @override
    def initialize_params(
        cls,
        y: Array,
        valid_y: Array,
        n_factors: int,
        *,
        random_state: int,
        params: Params | None = None,
    ) -> Params:
        """Initialise Gaussian readout via FA.

        Parameters
        ----------
        y : Array
            Observation tensor shaped ``(trials, time, obs_dim)``.
        valid_y : Array
            Observation mask aligned with ``y``.
        n_factors : int
            Number of latent factors.
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

        return Params(C=C, d=d, R=jnp.eye(y.shape[-1]))


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


def poisson_cvi_bin_stats_latent(
    m: Array,
    V: Array,
    y: Array,
    valid_y: Array,
    C: Array,
    d: Array,
) -> tuple[Array, Array]:
    """Poisson CVI gradients for a single bin in latent space.

    Unlike :func:`poisson_cvi_bin_stats` which operates in state space with
    information-form ``(z, Z)`` and ``H = C @ M``, this function works
    directly with latent-space moments ``(m, V)`` and loading ``C``.

    Parameters
    ----------
    m : Array
        Posterior mean in latent space, shape ``(latent_dim (K),)``.
    V : Array
        Posterior covariance in latent space,
        shape ``(latent_dim (K), latent_dim (K))``.
    y : Array
        Observed counts for the bin, shape ``(obs_dim (N),)``.
    valid_y : Array
        Observation mask for one bin, shape ``()``.
    C : Array
        Loading matrix, shape ``(obs_dim (N), latent_dim (K))``.
    d : Array
        Bias term, shape ``(obs_dim (N),)``.

    Returns
    -------
    tuple[Array, Array]
        Information vector ``k`` of shape ``(latent_dim (K),)`` and
        information matrix ``K`` of shape ``(latent_dim (K), latent_dim (K))``.
    """
    lin = C @ m + d
    quad = jnp.einsum("nk, kl, nl -> n", C, V, C)  # diag(C V C^T)
    eta = jnp.minimum(lin + 0.5 * quad, MAX_LOGRATE)
    lam = jnp.exp(eta)

    grad_m = (y - lam) @ C
    grad_V = -0.5 * jnp.einsum("ni, n, nj -> ij", C, lam, C)

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
    ) -> tuple[Array, Array]:
        """Initialise Poisson pseudo-observations in latent space.

        Computes per-bin CVI gradients from a zero-mean prior in latent space.
        Unlike the previous state-space implementation, this does not run a
        forward filter — each bin is initialised independently.  The CVI
        iterations in CVHM will refine the pseudo-observations.

        Parameters
        ----------
        params : Params
            Current readout parameter state.
        y : Array
            Observation tensor shaped ``(time, obs_dim (N))``.
        valid_y : Array
            Observation mask, shape ``(time,)``.

        Returns
        -------
        tuple[Array, Array]
            Pseudo-observation vectors and matrices in latent space with
            shapes ``(time, latent_dim (K))`` and
            ``(time, latent_dim (K), latent_dim (K))``.
        """
        C = params.loading()
        d = params.d
        K = C.shape[1]  # latent_dim

        m0 = jnp.zeros(K)
        V0 = jnp.eye(K)

        j, J = vmap(partial(poisson_cvi_bin_stats_latent, C=C, d=d))(
            jnp.broadcast_to(m0, (y.shape[0], K)),
            jnp.broadcast_to(V0, (y.shape[0], K, K)),
            y,
            valid_y,
        )
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

        y = y.reshape(-1, y.shape[-1])
        valid_y = valid_y.ravel()
        m = m.reshape(-1, m.shape[-1])
        V = V.reshape(-1, *V.shape[-2:])

        C, d = lbfgs_solve(
            (C, d), partial(poisson_trial_nell, y=y, valid_y=valid_y, m=m, V=V)
        )

        nell = poisson_trial_nell((C, d), y=y, valid_y=valid_y, m=m, V=V, gamma=0.0)
        return Params(C=C, d=d, R=None), nell

    @classmethod
    def update_pseudo(
        cls,
        params: Params,
        y: Array,
        valid_y: Array,
        m: Array,
        V: Array,
        j: Array,
        J: Array,
        lr: float,
    ) -> tuple[Array, Array]:
        """Update pseudo-observations using Poisson CVI gradients in latent space.

        Parameters
        ----------
        params : Params
            Current readout parameter state.
        y : Array
            Observation tensor shaped ``(trials, time, obs_dim (N))``.
        valid_y : Array
            Observation mask aligned with ``y``.
        m : Array
            Posterior means shaped ``(trials, time, latent_dim (K))``.
        V : Array
            Posterior covariances shaped
            ``(trials, time, latent_dim (K), latent_dim (K))``.
        j : Array
            Current pseudo-observation vectors shaped
            ``(trials, time, latent_dim (K))``.
        J : Array
            Current pseudo-observation matrices shaped
            ``(trials, time, latent_dim (K), latent_dim (K))``.
        lr : float
            Learning rate for the convex combination.

        Returns
        -------
        tuple[Array, Array]
            Updated pseudo-observation vectors and matrices in latent space.
        """
        C = params.loading()
        d = params.d
        k, K = vmap(vmap(partial(poisson_cvi_bin_stats_latent, C=C, d=d)))(
            m, V, y, valid_y
        )

        j = (1 - lr) * j + lr * k
        J = (1 - lr) * J + lr * K

        return j, J

    @classmethod
    def initialize_params(
        cls,
        y: Array,
        valid_y: Array,
        n_factors: int,
        *,
        random_state: int,
        params: Params | None = None,
    ) -> Params:
        """Initialise Poisson readout via FA followed by LBFGS optimisation.

        Parameters
        ----------
        y : Array
            Flattened observation tensor shaped ``(trials * time, obs_dim)``.
        valid_y : Array
            Observation mask aligned with ``y``.
        n_factors : int
            Number of latent factors.
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

        return Params(C=C, d=d, R=None)
