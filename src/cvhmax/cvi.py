from abc import abstractmethod
from functools import partial
from typing import ClassVar, override

from sklearn.decomposition import FactorAnalysis

from jax import Array, numpy as jnp, vmap
from jax.numpy.linalg import multi_dot
from jax.scipy.linalg import cho_factor, cho_solve
from equinox import Module

from .utils import (
    sequence_info_repr,
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
        Observations flattened to `(time * sequences, features)`.
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


def _initialization_data(y: Array, valid_y: Array) -> tuple[Array, Array, Array]:
    """Prepare observed time steps and mean-filled data for Factor Analysis."""
    y = y.reshape(-1, y.shape[-1])
    valid_y = valid_y.reshape(y.shape)
    active = jnp.any(valid_y, axis=1)
    y, valid_y = y[active], valid_y[active]
    observed = jnp.where(valid_y, y, 0.0)
    means = jnp.sum(observed, axis=0) / jnp.sum(valid_y, axis=0)
    return observed, valid_y, jnp.where(valid_y, y, means)


class Params(Module):
    """Container of CVI readout parameters.

    This container is used by the built-in Gaussian and Poisson readouts.
    CVHM's convergence metric compares the readout fields ``C``, ``d``, and
    ``R``.

    Attributes
    ----------
    C : Array
        Loading matrix with shape ``(obs_dim (N), latent_dim (K))``.
    d : Array
        Bias vector with shape ``(obs_dim (N),)``.
    R : Array
        Diagonal Gaussian observation covariance with shape ``(N, N)``.
        For Poisson readouts, the scalar sentinel ``jnp.asarray(0.0)``.
    """

    C: Array
    d: Array
    R: Array


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
            Observations shaped `(sequences, time, obs_dim (N))`.
        valid_y : Array
            Observation mask aligned with `y`.
        m : Array
            Posterior means shaped `(sequences, time, latent_dim (K))`.
        V : Array
            Posterior covariances shaped `(sequences, time, latent_dim (K), latent_dim (K))`.

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
            Observations shaped ``(sequences, time, obs_dim (N))``.
        valid_y : Array
            Observation mask aligned with ``y``.
        m : Array
            Posterior means shaped ``(sequences, time, latent_dim (K))``.
        V : Array
            Posterior covariances shaped
            ``(sequences, time, latent_dim (K), latent_dim (K))``.
        j : Array
            Pseudo-observation vectors shaped ``(sequences, time, latent_dim (K))``.
        J : Array
            Pseudo-observation matrices shaped
            ``(sequences, time, latent_dim (K), latent_dim (K))``.
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
            Observation mask, shape ``(time, obs_dim (N))``.

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

        CVHM's convergence metric compares the returned state's ``C``, ``d``,
        and ``R`` fields.

        Parameters
        ----------
        y : Array
            Observations shaped ``(sequences, time, obs_dim)``.
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
            Observation mask, shape ``(time, obs_dim)``.

        Returns
        -------
        tuple[Array, Array]
            Observation information vectors and matrices in latent space
            with shapes ``(time, latent_dim (K))`` and
            ``(time, latent_dim (K), latent_dim (K))``.
        """
        C = params.C
        d = params.d
        R = params.R
        return sequence_info_repr(y, valid_y, C, d, R)

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
            Observation tensor shaped `(sequences, time, obs_dim (N))`.
        valid_y : Array
            Observation mask aligned with `y`.
        m : Array
            Posterior means shaped `(sequences, time, latent_dim (K))`.
        P : Array
            Posterior covariances shaped `(sequences, time, latent_dim (K), latent_dim (K))`.

        Returns
        -------
        tuple[Params, float]
            Updated parameters and the (unused) negative log-likelihood proxy.
        """
        y = y.reshape(-1, y.shape[-1])
        valid_y = valid_y.reshape(y.shape)
        m = m.reshape(-1, m.shape[-1])
        P = P.reshape(-1, *P.shape[-2:])
        C, d, R = ridge_estimate(y, valid_y, m, P, params.R)
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
            Observation tensor shaped ``(sequences, time, obs_dim)``.
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

        _, _, filled = _initialization_data(y, valid_y)
        _, C, d = fa_init(filled, n_factors, random_state)
        return Params(C=C, d=d, R=jnp.eye(y.shape[-1]))


def poisson_readout_objective(
    params: tuple[Array, Array],
    y: Array,
    valid_y: Array,
    m: Array,
    V: Array,
    gamma: float = 10.0,
) -> float:
    """Average Poisson readout objective per observed time step.

    The count term uses the variance-adjusted log rate. This retained
    objective is not the exact expected Poisson log-likelihood.

    Parameters
    ----------
    params : tuple[Array, Array]
        Loading matrix and bias vector `(C, d)`.
    y : Array
        Counts with shape `(samples, obs_dim)`, flattened across sequences and time.
    valid_y : Array
        Observation mask aligned with `y`.
    m : Array
        Posterior means with shape `(samples, latent_dim)`.
    V : Array
        Posterior covariances with shape `(samples, latent_dim, latent_dim)`.
    gamma : float, default=10.0
        Strength of loading matrix L2 regularisation.

    Returns
    -------
    float
        Readout objective averaged over observed time steps.
    """
    C, d = params
    n_valid_steps = jnp.sum(jnp.any(valid_y, axis=-1))

    def step_objective(y_t, mask_t, m_t, V_t):
        lin = C @ m_t + d
        quad = jnp.einsum("ni,in->n", C, V_t @ C.T)
        eta = jnp.minimum(lin + 0.5 * quad, MAX_LOGRATE)
        observed_y = jnp.where(mask_t, y_t, 0.0)
        return jnp.sum(jnp.where(mask_t, jnp.exp(eta) - eta * observed_y, 0.0))

    step_objectives = vmap(step_objective)(y, valid_y, m, V)
    return (jnp.sum(step_objectives) + gamma * jnp.linalg.norm(C)) / n_valid_steps


def poisson_cvi_step_stats(
    z: Array,
    Z: Array,
    y: Array,
    valid_y: Array,
    H: Array,
    d: Array,
) -> tuple[Array, Array]:
    """Poisson CVI gradients for a single time step.

    Parameters
    ----------
    z : Array
        Information vector.
    Z : Array
        Information matrix.
    y : Array
        Observed counts for the time step.
    valid_y : Array
        Observation mask for one time step, shape ``(obs_dim,)``.
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

    observed_y = jnp.where(valid_y, y, 0.0)
    observed_lam = jnp.where(valid_y, lam, 0.0)
    grad_m = (observed_y - observed_lam) @ H
    K = jnp.einsum("ni,n,nj->ij", H, observed_lam, H)
    return grad_m + K @ m, K


def poisson_cvi_step_stats_latent(
    m: Array,
    V: Array,
    y: Array,
    valid_y: Array,
    C: Array,
    d: Array,
) -> tuple[Array, Array]:
    """Poisson CVI gradients for a single time step in latent space.

    Unlike :func:`poisson_cvi_step_stats` which operates in state space with
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
        Observed counts for the time step, shape ``(obs_dim (N),)``.
    valid_y : Array
        Observation mask for one time step, shape ``(obs_dim (N),)``.
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

    observed_y = jnp.where(valid_y, y, 0.0)
    observed_lam = jnp.where(valid_y, lam, 0.0)
    grad_m = (observed_y - observed_lam) @ C
    K = jnp.einsum("ni,n,nj->ij", C, observed_lam, C)
    return grad_m + K @ m, K


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

        Computes per-time-step CVI gradients from a zero-mean prior in latent space.
        Unlike the previous state-space implementation, this does not run a
        forward filter — each time step is initialised independently.  The CVI
        iterations in CVHM will refine the pseudo-observations.

        Parameters
        ----------
        params : Params
            Current readout parameter state.
        y : Array
            Observation tensor shaped ``(time, obs_dim (N))``.
        valid_y : Array
            Observation mask, shape ``(time, obs_dim (N))``.

        Returns
        -------
        tuple[Array, Array]
            Pseudo-observation vectors and matrices in latent space with
            shapes ``(time, latent_dim (K))`` and
            ``(time, latent_dim (K), latent_dim (K))``.
        """
        C = params.C
        d = params.d
        K = C.shape[1]  # latent_dim

        m0 = jnp.zeros(K)
        V0 = jnp.eye(K)

        j, J = vmap(partial(poisson_cvi_step_stats_latent, C=C, d=d))(
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
            Observation tensor shaped `(sequences, time, obs_dim (N))`.
        valid_y : Array
            Observation mask aligned with `y`.
        m : Array
            Posterior means shaped `(sequences, time, latent_dim (K))`.
        V : Array
            Posterior covariances shaped `(sequences, time, latent_dim (K), latent_dim (K))`.

        Returns
        -------
        tuple[Params, float]
            Updated parameters and the unregularized readout objective.
        """
        C = params.C
        d = params.d

        y = y.reshape(-1, y.shape[-1])
        valid_y = valid_y.reshape(y.shape)
        m = m.reshape(-1, m.shape[-1])
        V = V.reshape(-1, *V.shape[-2:])

        C, d = lbfgs_solve(
            (C, d), partial(poisson_readout_objective, y=y, valid_y=valid_y, m=m, V=V)
        )

        nell = poisson_readout_objective((C, d), y=y, valid_y=valid_y, m=m, V=V, gamma=0.0)
        return Params(C=C, d=d, R=jnp.asarray(0.0)), nell

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
            Observation tensor shaped ``(sequences, time, obs_dim (N))``.
        valid_y : Array
            Observation mask aligned with ``y``.
        m : Array
            Posterior means shaped ``(sequences, time, latent_dim (K))``.
        V : Array
            Posterior covariances shaped
            ``(sequences, time, latent_dim (K), latent_dim (K))``.
        j : Array
            Current pseudo-observation vectors shaped
            ``(sequences, time, latent_dim (K))``.
        J : Array
            Current pseudo-observation matrices shaped
            ``(sequences, time, latent_dim (K), latent_dim (K))``.
        lr : float
            Learning rate for the convex combination.

        Returns
        -------
        tuple[Array, Array]
            Updated pseudo-observation vectors and matrices in latent space.
        """
        C = params.C
        d = params.d
        k, K = vmap(vmap(partial(poisson_cvi_step_stats_latent, C=C, d=d)))(
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
            Flattened observation tensor shaped ``(sequences * time, obs_dim)``.
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

        y, valid_y, filled = _initialization_data(y, valid_y)
        m, C, _ = fa_init(filled, n_factors, random_state)
        counts = jnp.sum(valid_y, axis=0)
        mean_rate = (jnp.sum(y, axis=0) + 0.5) / counts
        # FA operates on counts; the Poisson readout operates on log rates.
        C = C / mean_rate[:, None]
        d = jnp.log(mean_rate)
        V = jnp.zeros((y.shape[0], n_factors, n_factors))
        C, d = lbfgs_solve(
            (C, d), partial(poisson_readout_objective, y=y, valid_y=valid_y, m=m, V=V)
        )

        return Params(C=C, d=d, R=jnp.asarray(0.0))
