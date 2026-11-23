from abc import abstractmethod
from functools import partial
from typing import ClassVar, override

from sklearn.decomposition import FactorAnalysis

import jax
import numpy as np
import scipy.optimize
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

# Lower bound on the ARD precision, as a log. The marginal evidence is flat
# (not minimised) as a precision vanishes, so optimising it needs a floor.
ARD_LOG_ALPHA_FLOOR = -12.0


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


def _gaussian_loading_moments(
    y: Array, valid_y: Array, m: Array, V: Array
) -> tuple[Array, Array, Array]:
    """Sufficient statistics of the expected Gaussian readout in ``(d, C)``.

    Given latent moments ``q(z) = N(m, V)``, the expected readout
    log-likelihood is quadratic in ``beta = [d, C]`` of the form
    ``-0.5 beta' Prec beta + beta' c``, up to terms independent of ``beta``.
    That quadratic form is what makes a Gaussian loading prior conjugate.

    Parameters
    ----------
    y : Array
        Observations shaped ``(time, obs_dim (N))``.
    valid_y : Array
        Observation mask aligned with ``y``.
    m : Array
        Latent posterior means shaped ``(time, latent_dim (K))``.
    V : Array
        Latent posterior covariances shaped ``(time, latent_dim (K), latent_dim (K))``.

    Returns
    -------
    tuple[Array, Array, Array]
        ``Prec`` shaped ``(N, K + 1, K + 1)``, ``c`` shaped ``(N, K + 1)``, and
        the observed count per feature shaped ``(N,)``. The bias index is 0 and
        is never regularised by the loading prior.
    """
    n_time = m.shape[0]
    design = jnp.concatenate((jnp.ones((n_time, 1), dtype=m.dtype), m), axis=1)
    outer = design[:, :, None] * design[:, None, :]
    outer = outer.at[:, 1:, 1:].add(V)
    observed = valid_y.astype(m.dtype)
    precision = jnp.einsum("tn,tij->nij", observed, outer)
    linear = jnp.einsum("tn,ti->ni", jnp.where(valid_y, y, 0.0), design)
    return precision, linear, jnp.sum(observed, axis=0)


def gaussian_ard_evidence(
    y: Array, valid_y: Array, m: Array, V: Array, log_alpha: Array, log_R: Array
) -> Array:
    """Marginal log-evidence of the Gaussian readout under ARD loading priors.

    Integrates the loadings out of the expected readout log-likelihood under
    ``C[:, k] ~ N(0, alpha_k^-1 I)``. The prior on the bias is flat, which
    contributes an additive constant that is identical across models and is
    therefore omitted.

    Parameters
    ----------
    y : Array
        Observations shaped ``(time, obs_dim (N))``.
    valid_y : Array
        Observation mask aligned with ``y``.
    m : Array
        Latent posterior means shaped ``(time, latent_dim (K))``.
    V : Array
        Latent posterior covariances shaped ``(time, latent_dim (K), latent_dim (K))``.
    log_alpha : Array
        Log ARD precisions, shape ``(latent_dim (K),)``.
    log_R : Array
        Log diagonal observation variances, shape ``(obs_dim (N),)``.

    Returns
    -------
    Array
        Scalar marginal log-evidence.

    Notes
    -----
    The likelihood used here is ``exp(E_q[log p(y | z, C, d)])``, which Jensen
    bounds below the latent-averaged likelihood ``E_q[p(y | z, C, d)]``. The
    returned value is therefore a lower bound on the evidence at the supplied
    latent moments, not the evidence under the latent prior.
    """
    precision, linear, counts = _gaussian_loading_moments(y, valid_y, m, V)
    n_obs = precision.shape[0]
    variance = jnp.exp(log_R)
    scaled = linear / variance[:, None]
    posterior = precision / variance[:, None, None] + jnp.diag(
        jnp.concatenate((jnp.zeros(1), jnp.exp(log_alpha)))
    )[None]
    _, logdet = jnp.linalg.slogdet(posterior)
    solved = jnp.linalg.solve(posterior, scaled[..., None])[..., 0]
    squared = jnp.sum(jnp.where(valid_y, y**2, 0.0), axis=0)
    # Per feature: the Gaussian normalisation of the observation term, plus the
    # proper-Gaussian normalisation of the precision prior, plus the
    # normalisation of the Gaussian integral over (d, C).
    normalisation = 0.5 * n_obs * (
        jnp.sum(log_alpha - jnp.log(2 * jnp.pi))
        + (log_alpha.shape[0] + 1) * jnp.log(2 * jnp.pi)
    )
    return (
        jnp.sum(
            -0.5 * counts * jnp.log(2 * jnp.pi * variance)
            - 0.5 * squared / variance
            - 0.5 * logdet
            + 0.5 * jnp.sum(scaled * solved, axis=-1)
        )
        + normalisation
    )


def gaussian_ard_estimate(
    y: Array,
    valid_y: Array,
    m: Array,
    V: Array,
    *,
    R: Array | None = None,
    alpha_init: Array | None = None,
    rounds: int = 4,
) -> tuple[Params, Array, float]:
    """Fit Gaussian loadings with per-latent ARD precisions.

    Maximises :func:`gaussian_ard_evidence` over the precisions, with the
    loadings integrated out rather than profiled. Precisions are reported as
    fitted: a large value marks a latent whose column has been shrunk towards
    zero, and the separation between latent precisions is what identifies an
    effective dimensionality. Latent moments are held fixed, so this is a
    conditional analysis.

    This replaces the ridge loading update rather than supplementing it: it
    owns the whole readout M-step, alternating the precision fit with the
    loading solve and the variance refresh. Combined with a separate ridge
    penalty it would double-regularise the same block.

    Parameters
    ----------
    y : Array
        Observations shaped ``(time, obs_dim (N))``.
    valid_y : Array
        Observation mask aligned with ``y``.
    m : Array
        Latent posterior means shaped ``(time, latent_dim (K))``.
    V : Array
        Latent posterior covariances shaped ``(time, latent_dim (K), latent_dim (K))``.
    R : Array, optional
        Initial diagonal observation covariance shaped ``(N, N)``. Defaults to
        the marginal variance of each feature about its masked mean.
    alpha_init : Array, optional
        Initial ARD precisions, shape ``(K,)``. Defaults to ones.
    rounds : int, default=4
        Number of loadings/variances/precisions refinements.

    Returns
    -------
    tuple[Params, Array, float]
        Fitted readout parameters, ARD precisions shaped ``(K,)``, and the
        marginal log-evidence at the optimum.

    Notes
    -----
    The observation variances are re-estimated from expected residuals after
    each precision update, so the two are jointly consistent at the returned
    point. They are never optimised against the surrogate evidence itself,
    which grows without bound as a variance vanishes.
    """
    n_latent = m.shape[1]
    counts = jnp.sum(valid_y.astype(m.dtype), axis=0)
    if R is None:
        mean = jnp.sum(jnp.where(valid_y, y, 0.0), axis=0) / counts
        variance = jnp.sum(jnp.where(valid_y, (y - mean) ** 2, 0.0), axis=0) / counts
    else:
        variance = jnp.diag(jnp.asarray(R))
    alpha = jnp.ones(n_latent) if alpha_init is None else jnp.asarray(alpha_init)

    def solve_loadings(alpha_: Array, variance_: Array) -> tuple[Array, Array]:
        precision, linear, _ = _gaussian_loading_moments(y, valid_y, m, V)
        posterior = precision / variance_[:, None, None] + jnp.diag(
            jnp.concatenate((jnp.zeros(1), alpha_))
        )[None]
        beta = jnp.linalg.solve(
            posterior, (linear / variance_[:, None])[..., None]
        )[..., 0]
        return beta[:, 1:], beta[:, 0]

    def maximise_evidence(variance_: Array, start: Array) -> Array:
        # An unconstrained variable with the precision floored: the evidence
        # flattens, rather than diverging, as a precision vanishes.
        def negative_evidence(unconstrained: Array) -> Array:
            log_alpha = ARD_LOG_ALPHA_FLOOR + jax.nn.softplus(unconstrained)
            return -gaussian_ard_evidence(
                y, valid_y, m, V, log_alpha, jnp.log(variance_)
            )

        value_and_grad = jax.jit(jax.value_and_grad(negative_evidence))

        def objective(unconstrained):
            value, grad = value_and_grad(jnp.asarray(unconstrained))
            return float(value), np.asarray(grad)

        initial = jnp.log(jnp.expm1(jnp.log(start) - ARD_LOG_ALPHA_FLOOR))
        result = scipy.optimize.minimize(
            objective,
            np.asarray(initial),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-10},
        )
        return jnp.exp(ARD_LOG_ALPHA_FLOOR + jax.nn.softplus(jnp.asarray(result.x)))

    for _ in range(rounds):
        alpha = maximise_evidence(variance, alpha)
        loadings, bias = solve_loadings(alpha, variance)
        residual = jnp.where(valid_y, y - (m @ loadings.T + bias), 0.0)
        uncertainty = jnp.einsum("nk,tkl,nl->tn", loadings, V, loadings)
        variance = jnp.sum(
            jnp.where(valid_y, residual**2 + uncertainty, 0.0), axis=0
        ) / counts

    loadings, bias = solve_loadings(alpha, variance)
    params = Params(C=loadings, d=bias, R=jnp.diag(variance))
    evidence = float(
        gaussian_ard_evidence(y, valid_y, m, V, jnp.log(alpha), jnp.log(variance))
    )
    return params, alpha, evidence


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
    gamma: float = 1.5,
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
    gamma : float, default=1.5
        Strength of the squared L2 penalty on the loading matrix. This default
        matches the effective strength of the previous norm penalty, whose
        gradient magnitude was `gamma` rather than `2 * gamma * ||C||`.

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
    return (jnp.sum(step_objectives) + gamma * jnp.sum(C**2)) / n_valid_steps


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
