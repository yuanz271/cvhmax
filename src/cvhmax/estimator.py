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
from jax import Array, numpy as jnp

from .cvi import Params
from .utils import LoadingTerms, lbfgs_solve

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
        self.alpha = jnp.asarray(np.atleast_1d(np.asarray(alpha, dtype=float)))

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
            Observations shaped ``(sequences, time, features)``.
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
        if params is None:
            params = likelihood.initialize_params(
                y,
                valid_y,
                n_factors,
                random_state=random_state,
                penalty=self.penalty(n_factors),
            )
        if params.alpha is None:
            # Carried through the EM loop, so it must be an array from the
            # start: a ``None`` placeholder would change the pytree structure
            # between the initial state and the first update.
            params = Params(
                C=params.C,
                d=params.d,
                R=params.R,
                alpha=jnp.broadcast_to(self.alpha, (n_factors,)),
            )
        return params

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
            Observations shaped ``(sequences, time, features)``.
        valid_y : Array
            Observation mask aligned with ``y``.
        m : Array
            Latent posterior means shaped ``(sequences, time, latent_dim)``.
        V : Array
            Latent posterior covariances shaped
            ``(sequences, time, latent_dim, latent_dim)``.

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
        return Params(C=C, d=d, R=params.R, alpha=params.alpha), jnp.nan


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
    return Params(C=C, d=d, R=jnp.diag(variance), alpha=params.alpha)


class ARDEstimator(Estimator):
    """Per-column precision fitted against the evidence.

    Maximises the marginal evidence over the precisions and the observation
    variances jointly, with the loadings integrated out rather than profiled.
    Needs the likelihood's conjugate loading representation, because the
    evidence integral is only closed form when the data term is quadratic in
    the loadings.

    Both stationary points have closed forms, so the fit is a fixed-point
    iteration rather than an optimisation: for each feature,

    ```
    gamma_nk = 1 - alpha_k * Sigma_n[k, k]
    alpha_k  = (sum_n gamma_nk + 2a - 2) / (sum_n C[n, k]^2 + 2b)
    R_n      = RSS_n / (N_n - sum_k gamma_nk)
    ```

    with `Sigma_n` the exact posterior covariance `(Prec_n / R_n + diag(0,
    alpha))^-1` and `RSS_n` the expected residual sum of squares including
    latent uncertainty. The observation variance needs no prior: its rule is
    the residual variance with the effective parameter count removed, and that
    denominator is bounded away from zero whenever a feature has more observed
    time steps than parameters.

    The loading precisions do need one. `gamma_k` and `||C[:, k]||^2` both
    vanish together for a column that is being pruned, so the bare rule is a
    `0 / 0` limit; measured over a grid of problems it diverges or runs to
    precisions of order `1e17` in a majority of cases. `(a, b)` is a weak Gamma
    prior on the loading precisions, defaulting to the reference
    implementation's settings, which removes the corner. It is not a modelling
    claim so much as a guard on the updater.

    An irrelevant column's loadings are driven to numerical zero, but the column
    is not removed from ``C`` and its precision plateaus at a large finite value
    set by the guard prior rather than by the data. Only the ranking of the
    precisions is meaningful, and any threshold is the caller's decision.
    """

    requires_conjugate_terms = True

    def __init__(
        self,
        alpha: float | Array = 1.0,
        *,
        alpha_shape: float = 1.0 + 1e-6,
        alpha_rate: float = 1e-6,
        iterations: int = 50,
    ):
        """Construct an ARD estimator.

        Parameters
        ----------
        alpha : float or Array, default=1.0
            Initial loading precision.
        alpha_shape : float, default=1.0 + 1e-6
            Shape of the Gamma prior on the loading precisions. In the
            reference ARD regression this is its `lambda_1 + 1`, and
            `alpha_rate` is `lambda_2`.
        alpha_rate : float, default=1e-6
            Rate of the Gamma prior on the loading precisions.
        iterations : int, default=50
            Fixed-point iterations.
        """
        super().__init__(alpha)
        self.alpha_shape = alpha_shape
        self.alpha_rate = alpha_rate
        self.iterations = iterations

    def estimate(
        self, likelihood, params: Params, y: Array, valid_y: Array, m: Array, V: Array
    ) -> tuple[Params, float]:
        y, valid_y, m, V = _flattened(y, valid_y, m, V)
        terms = likelihood.loading_terms(y, valid_y, m, V)
        n_latents = m.shape[-1]

        def solve(alpha_: Array, variance_: Array) -> tuple[Array, Array, Array]:
            """Posterior covariance and moments at the current hyperparameters."""
            precision = terms.precision / variance_[:, None, None] + jnp.diag(
                jnp.concatenate((jnp.zeros(1), alpha_))
            )[None]
            covariance = jnp.linalg.inv(precision)
            beta = covariance @ (terms.linear / variance_[:, None])[..., None]
            return covariance, beta[..., 0], jnp.diagonal(covariance, axis1=1, axis2=2)

        def step(carry, _):
            alpha_, variance_ = carry
            _, beta, diagonal = solve(alpha_, variance_)
            loadings, bias = beta[:, 1:], beta[:, 0]
            effective = 1.0 - jnp.concatenate((jnp.zeros(1), alpha_))[None, :] * diagonal
            alpha_next = (
                jnp.sum(effective[:, 1:], axis=0) + 2.0 * self.alpha_shape - 2.0
            ) / (jnp.sum(loadings**2, axis=0) + 2.0 * self.alpha_rate)
            residual = jnp.where(valid_y, y - (m @ loadings.T + bias), 0.0)
            uncertainty = jnp.einsum("nk,tkl,nl->tn", loadings, V, loadings)
            rss = jnp.sum(
                jnp.where(valid_y, residual**2 + uncertainty, 0.0), axis=0
            )
            variance_next = rss / (terms.counts - jnp.sum(effective, axis=-1))
            return (alpha_next, variance_next), None

        initial = (jnp.broadcast_to(self.alpha, (n_latents,)), jnp.diag(params.R))
        (alpha, variance), _ = jax.lax.scan(step, initial, None, length=self.iterations)

        _, beta, _ = solve(alpha, variance)
        evidence = ard_evidence(
            terms,
            jnp.log(alpha),
            jnp.log(variance),
            alpha_shape=self.alpha_shape,
            alpha_rate=self.alpha_rate,
        )
        return (
            Params(C=beta[:, 1:], d=beta[:, 0], R=jnp.diag(variance), alpha=alpha),
            evidence,
        )


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
        Shape of a Gamma prior on the precisions. Both this and
        `alpha_rate` are needed to enable it.
    alpha_rate : float, optional
        Rate of the Gamma prior on the precisions.

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
        # Gamma(shape, rate) on the loading precisions, shared across features.
        evidence = evidence + (alpha_shape - 1.0) * jnp.sum(log_alpha)
        evidence = evidence - alpha_rate * jnp.sum(jnp.exp(log_alpha))
    return evidence
