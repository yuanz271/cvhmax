"""How the observation loadings and their precision are fitted.

An estimator owns the readout M-step: it turns a likelihood's readout data
term into fitted :class:`~cvhmax.cvi.Params`. It also carries the loading prior.
Ridge and ARD share one prior, ``C[:, k] ~ N(0, alpha_k^-1 I)`` with a flat
bias, and differ only in whether ``alpha`` is fitted -- so choosing an estimator
chooses a prior as well as an algorithm.

Two likelihood capabilities are used. A likelihood may expose a *conjugate*
loading representation, a quadratic form in ``[d, C]``; that is what lets an
estimator integrate the loadings out rather than profile them. Otherwise it
exposes a readout objective that takes the penalty diagonal and owns its own
averaging.
"""

from abc import abstractmethod
from typing import ClassVar

import jax
import numpy as np
import scipy.optimize
from jax import Array, numpy as jnp

from .cvi import Params
from .utils import ARD_LOG_ALPHA_FLOOR, LoadingTerms, lbfgs_solve

Penalty = Array


def _flattened(
    y: Array, valid_y: Array, m: Array, V: Array
) -> tuple[Array, Array, Array, Array]:
    """Collapse sequence and time into one axis.

    The readout is fitted per feature against every observed time step, so the
    leading axes carry no meaning beyond their product.
    """
    y = y.reshape(-1, y.shape[-1])
    valid_y = valid_y.reshape(y.shape)
    m = m.reshape(-1, m.shape[-1])
    V = V.reshape(-1, *V.shape[-2:])
    return y, valid_y, m, V


class Estimator:
    """Base class for loading estimators, registered by class name.

    Attributes
    ----------
    registry : dict
        Maps class names to estimator classes, mirroring `CVI.registry`.
    requires_conjugate_terms : bool
        Whether the estimator needs the likelihood's conjugate loading
        representation. Checked at the API boundary.
    """

    registry: ClassVar[dict[str, type["Estimator"]]] = {}
    requires_conjugate_terms: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Estimator.registry[cls.__name__] = cls

    def __init__(self, alpha: float | Array = 0.1):
        """Construct an estimator.

        Parameters
        ----------
        alpha : float or Array, default=0.1
            Loading precision: a scalar applied to every latent column, or one
            value per column. For `RidgeEstimator` it is held fixed; for
            `ARDEstimator` it is the starting point.
        """
        values = np.atleast_1d(np.asarray(alpha, dtype=float))
        self.alpha = jnp.asarray(values)
        # Resolved here rather than on demand: this is read from inside a traced
        # EM loop, where boolean conversion of a JAX array is not allowed even
        # when the array is concrete.
        self.isotropic = bool(np.all(values == values[0]))

    def penalty(self, n_latents: int) -> Penalty:
        """Return the penalty diagonal, zero at the bias index.

        Parameters
        ----------
        n_latents : int
            Number of latent columns.

        Returns
        -------
        Array
            Shape ``(n_latents + 1,)``, in log-density units.
        """
        alpha = jnp.broadcast_to(self.alpha, (n_latents,))
        return jnp.concatenate((jnp.zeros(1), alpha))

    def initialize_params(
        self, likelihood, y: Array, valid_y: Array, n_factors: int, *, random_state, params=None
    ) -> Params:
        """Produce the starting readout parameters.

        The data-driven start belongs to the likelihood; the estimator drives it
        and supplies the penalty convention, so initialization and refinement
        cannot drift apart.

        Parameters
        ----------
        likelihood : CVI
            Readout whose observation model supplies the starting point.
        y : Array
            Observations shaped ``(time, features)``.
        valid_y : Array
            Observation mask aligned with ``y``.
        n_factors : int
            Number of latent columns.
        random_state : int
            Seed for the likelihood's initialization.
        params : Params, optional
            If given, returned unchanged. This is how a fit is frozen at preset
            values.

        Returns
        -------
        Params
            Initial readout parameters.
        """
        if params is not None:
            return params
        return likelihood.initialize_params(
            y,
            valid_y,
            n_factors,
            random_state=random_state,
            penalty=self.penalty(n_factors),
        )

    @abstractmethod
    def estimate(
        self, likelihood, params: Params, y: Array, valid_y: Array, m: Array, V: Array
    ) -> tuple[Params, float]:
        """Fit the readout parameters.

        Parameters
        ----------
        likelihood : CVI
            Readout whose observation model supplies the data term.
        params : Params
            Current readout parameters.
        y : Array
            Observations shaped ``(time, features)``.
        valid_y : Array
            Observation mask aligned with ``y``.
        m : Array
            Latent posterior means shaped ``(time, latent_dim)``.
        V : Array
            Latent posterior covariances shaped ``(time, latent_dim, latent_dim)``.

        Returns
        -------
        tuple[Params, float]
            Updated parameters and a scalar objective for reporting.
        """


class RidgeEstimator(Estimator):
    """Fixed-precision loadings: ``alpha`` is a hyperparameter, not fitted.

    Works with any likelihood. It uses the closed form when the likelihood
    exposes conjugate loading terms and a generic gradient solve otherwise.
    """

    def estimate(
        self, likelihood, params: Params, y: Array, valid_y: Array, m: Array, V: Array
    ) -> tuple[Params, float]:
        y, valid_y, m, V = _flattened(y, valid_y, m, V)
        penalty = self.penalty(m.shape[-1])
        terms = likelihood.loading_terms(y, valid_y, m, V)
        if terms is not None:
            return _conjugate_solve(terms, penalty, params, y, valid_y, m, V), jnp.nan
        C, d = lbfgs_solve(
            (params.C, params.d),
            lambda readout: likelihood.readout_objective(
                readout, y, valid_y, m, V, penalty=penalty
            ),
        )
        return Params(C=C, d=d, R=params.R), jnp.nan


class FrozenEstimator(Estimator):
    """Hold the readout fixed while the latents are still inferred."""

    def estimate(
        self, likelihood, params: Params, y: Array, valid_y: Array, m: Array, V: Array
    ) -> tuple[Params, float]:
        return params, jnp.nan


def _conjugate_solve(
    terms: LoadingTerms,
    penalty: Penalty,
    params: Params,
    y: Array,
    valid_y: Array,
    m: Array,
    V: Array,
) -> Params:
    """Solve the penalised conjugate objective and refresh the variances.

    The penalty is applied in absolute precision units: solving each feature
    against its own noise variance brings an ``R_nn`` factor into the
    regulariser, which is a unit conversion rather than a change in strength.
    """
    variance = jnp.diag(params.R)
    posterior = terms.precision / variance[:, None, None] + jnp.diag(penalty)[None]
    beta = jnp.linalg.solve(
        posterior, (terms.linear / variance[:, None])[..., None]
    )[..., 0]
    C, d = beta[:, 1:], beta[:, 0]
    residual = jnp.where(valid_y, y - (m @ C.T + d), 0.0)
    uncertainty = jnp.einsum("nk,tkl,nl->tn", C, V, C)
    variance = jnp.sum(
        jnp.where(valid_y, residual**2 + uncertainty, 0.0), axis=0
    ) / terms.counts
    return Params(C=C, d=d, R=jnp.diag(variance))


class ARDEstimator(Estimator):
    """Per-column precision fitted against the evidence.

    Maximises the marginal evidence over the precisions and the observation
    variances jointly, with the loadings integrated out rather than profiled.
    Needs the likelihood's conjugate loading representation, because the
    evidence integral is only closed form when the data term is quadratic in
    the loadings.

    Fitted precisions are *not* zeros: an irrelevant column is shrunk towards
    zero and its precision plateaus at a large finite value, so exact selection
    remains the caller's decision.
    """

    requires_conjugate_terms = True

    def __init__(
        self,
        alpha: float | Array = 1.0,
        *,
        alpha_shape: float | None = None,
        alpha_rate: float | None = None,
    ):
        """Construct an ARD estimator.

        Parameters
        ----------
        alpha : float or Array, default=1.0
            Initial loading precision.
        alpha_shape : float, optional
            Shape of a Gamma hyperprior on the precisions. Both this and
            `alpha_rate` are needed to enable it. With a hyperprior the
            precisions are bounded at both ends by construction.
        alpha_rate : float, optional
            Rate of the Gamma hyperprior on the precisions.
        """
        super().__init__(alpha)
        self.alpha_shape = alpha_shape
        self.alpha_rate = alpha_rate
        # Fitted per-column precisions leave only permutation and sign.
        self.isotropic = False

    def estimate(
        self, likelihood, params: Params, y: Array, valid_y: Array, m: Array, V: Array
    ) -> tuple[Params, float]:
        y, valid_y, m, V = _flattened(y, valid_y, m, V)
        terms = likelihood.loading_terms(y, valid_y, m, V)
        n_latents = m.shape[-1]

        def negative_evidence(unconstrained: Array) -> Array:
            # An unconstrained variable with the precision floored: the
            # evidence flattens, rather than diverging, as a precision vanishes.
            log_alpha = ARD_LOG_ALPHA_FLOOR + jax.nn.softplus(
                unconstrained[:n_latents]
            )
            return -ard_evidence(
                terms,
                log_alpha,
                unconstrained[n_latents:],
                alpha_shape=self.alpha_shape,
                alpha_rate=self.alpha_rate,
            )

        value_and_grad = jax.jit(jax.value_and_grad(negative_evidence))

        def objective(unconstrained):
            value, grad = value_and_grad(jnp.asarray(unconstrained))
            return float(value), np.asarray(grad)

        variance = jnp.diag(params.R)
        initial = jnp.concatenate(
            (
                jnp.log(
                    jnp.expm1(
                        jnp.log(jnp.broadcast_to(self.alpha, (n_latents,)))
                        - ARD_LOG_ALPHA_FLOOR
                    )
                ),
                jnp.log(variance),
            )
        )
        result = scipy.optimize.minimize(
            objective,
            np.asarray(initial),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-10},
        )
        optimum = jnp.asarray(result.x)
        alpha = jnp.exp(
            ARD_LOG_ALPHA_FLOOR + jax.nn.softplus(optimum[:n_latents])
        )
        params = _conjugate_solve(
            terms,
            jnp.concatenate((jnp.zeros(1), alpha)),
            Params(C=params.C, d=params.d, R=jnp.diag(jnp.exp(optimum[n_latents:]))),
            y,
            valid_y,
            m,
            V,
        )
        return params, -float(result.fun)


def ard_evidence(
    terms: LoadingTerms,
    log_alpha: Array,
    log_variance: Array,
    *,
    alpha_shape: float | None = None,
    alpha_rate: float | None = None,
) -> Array:
    """Marginal log-evidence under per-column loading precisions.

    Parameters
    ----------
    terms : LoadingTerms
        Conjugate loading representation from the likelihood.
    log_alpha : Array
        Log loading precisions, shape ``(latent_dim (K),)``.
    log_variance : Array
        Log diagonal observation variances, shape ``(features (N),)``.
    alpha_shape : float, optional
        Shape of a Gamma hyperprior on the precisions. Both this and
        `alpha_rate` are needed to enable it.
    alpha_rate : float, optional
        Rate of the Gamma hyperprior on the precisions.

    Returns
    -------
    Array
        Scalar marginal log-evidence.

    Notes
    -----
    The likelihood used is ``exp(E_q[log p(y | z, C, d)])``, which Jensen bounds
    below the latent-averaged likelihood, so the value is a lower bound at the
    supplied latent moments. With ``V = 0`` the bound is tight and this is the
    exact marginal likelihood of an ARD linear regression on the latent means.
    """
    n_obs = terms.precision.shape[0]
    variance = jnp.exp(log_variance)
    scaled = terms.linear / variance[:, None]
    posterior = terms.precision / variance[:, None, None] + jnp.diag(
        jnp.concatenate((jnp.zeros(1), jnp.exp(log_alpha)))
    )[None]
    _, logdet = jnp.linalg.slogdet(posterior)
    solved = jnp.linalg.solve(posterior, scaled[..., None])[..., 0]
    # Per feature: the Gaussian normalisation of the observation term, the
    # proper-Gaussian normalisation of the precision prior, and the
    # normalisation of the Gaussian integral over (d, C).
    normalisation = 0.5 * n_obs * (
        jnp.sum(log_alpha - jnp.log(2 * jnp.pi))
        + (log_alpha.shape[0] + 1) * jnp.log(2 * jnp.pi)
    )
    evidence = (
        jnp.sum(
            -0.5 * terms.counts * jnp.log(2 * jnp.pi * variance)
            - 0.5 * terms.squared / variance
            - 0.5 * logdet
            + 0.5 * jnp.sum(scaled * solved, axis=-1)
        )
        + normalisation
    )
    if alpha_shape is not None and alpha_rate is not None:
        # Gamma(shape, rate) on the precisions, shared across features.
        evidence = evidence + (alpha_shape - 1.0) * jnp.sum(log_alpha)
        evidence = evidence - alpha_rate * jnp.sum(jnp.exp(log_alpha))
    return evidence
