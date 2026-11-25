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

import math
from abc import abstractmethod
from typing import Any, ClassVar

import jax
import numpy as np
from jax import Array, numpy as jnp

from .cvi import CVI, Params, Poisson
from .utils import LoadingTerms, lbfgs_solve

Penalty = Array


def _portable_json(value: Any, name: str) -> Any:
    """Convert one estimator-configuration value to finite JSON primitives."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        scalar = float(value)
        if not math.isfinite(scalar):
            raise ValueError(f"{name} must be finite")
        return scalar
    if isinstance(value, (np.ndarray, jax.Array)):
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError(f"{name} must not contain object arrays")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite")
        return array.tolist()
    if isinstance(value, (list, tuple)):
        return [_portable_json(item, name) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError(f"{name} keys must be strings")
        return {key: _portable_json(item, f"{name}.{key}") for key, item in value.items()}
    raise ValueError(
        f"{name} has unsupported type {type(value).__name__}; "
        "use JSON-compatible scalars, lists, or arrays"
    )


def _portable_estimator_config(config: dict | None) -> dict | None:
    """Return estimator constructor keywords as finite JSON primitives.

    Numpy and JAX scalars become Python scalars and arrays become nested lists,
    so a per-column ``alpha`` keeps its logical shape while the result can be
    written to a JSON manifest. Unsupported or non-finite values raise
    ``ValueError``.

    Parameters
    ----------
    config : dict or None
        Estimator constructor keywords, as passed to ``CVHM``.

    Returns
    -------
    dict or None
        A JSON-compatible copy with the same logical values.
    """
    if config is None:
        return None
    if not isinstance(config, dict):
        raise ValueError("estimator_config must be a dict or None")
    return _portable_json(config, "estimator_config")


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
        Whether the default likelihood check requires conjugate loading terms.
        An estimator may additionally support a specific likelihood.
    fits_ard_precisions : bool
        Whether the estimator fits ARD loading precisions. Checked at the fit
        boundary for the residual degrees-of-freedom contract and for
        non-finite or non-positive precisions or evidence.
    """

    registry: ClassVar[dict[str, type["Estimator"]]] = {}
    requires_conjugate_terms: ClassVar[bool] = False
    fits_ard_precisions: ClassVar[bool] = False

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
            `ARDEstimator` it is the starting point. It must be finite and
            non-negative; ARD requires strictly positive values.
        """
        alpha = jnp.asarray(np.atleast_1d(np.asarray(alpha, dtype=float)))
        if not bool(jnp.all(jnp.isfinite(alpha) & (alpha >= 0))):
            raise ValueError("alpha must be finite and non-negative")
        self.alpha = alpha

    def _supports_likelihood(self, likelihood: type[CVI]) -> bool:
        """Check whether the likelihood provides the required loading terms."""
        return not self.requires_conjugate_terms or (
            likelihood.loading_terms.__func__ is not CVI.loading_terms.__func__
        )

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
            values. For `ARDEstimator`, ``params.alpha`` is the starting
            precision of the fixed-point iteration.

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
    Updated parameters report the configured precision; the generic objective
    reports its unregularized data term for training diagnostics.
    """

    def initialize_params(
        self, likelihood, y: Array, valid_y: Array, n_factors: int, *, random_state, params=None
    ) -> Params:
        """Initialize the readout with the configured fixed precision."""
        params = super().initialize_params(
            likelihood, y, valid_y, n_factors, random_state=random_state, params=params
        )
        return Params(C=params.C, d=params.d, R=params.R, alpha=self.penalty(n_factors)[1:])

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
        objective = likelihood.readout_objective(
            (C, d), y, valid_y, m, V, penalty=jnp.zeros_like(penalty)
        )
        return Params(C=C, d=d, R=params.R, alpha=penalty[1:]), objective


class FixedEstimator(Estimator):
    """Keep the initialized readout fixed while inferring the latents.

    Supplied ``C``, ``d``, and ``R`` are preserved; otherwise the observation
    model initializes them from the data before they are held fixed.
    """

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
    return Params(C=C, d=d, R=jnp.diag(variance), alpha=penalty[1:])


class ARDEstimator(Estimator):
    """Per-column loading precision fitted from conditional evidence.

    Gaussian loadings have a closed-form evidence integral and jointly fitted
    observation variances. Poisson loadings use a Laplace approximation to that
    integral, with the latent CVI E-step unchanged. Its precision update fixes
    the local curvature, so it is not the exact gradient of Laplace evidence.

    Both likelihoods use the shared-precision fixed-point update

    ```
    gamma_nk = 1 - alpha_k * Sigma_n[k, k]
    alpha_k  = (sum_n gamma_nk + 2a - 2) / (sum_n C[n, k]^2 + 2b)
    ```

    For Gaussian observations, `Sigma_n` is the exact conditional covariance
    `(Prec_n / R_n + diag(0, alpha))^-1`. The observation variance is updated
    from expected residuals as `R_n = RSS_n / (N_n - 1 - sum_k gamma_nk)`.
    Poisson instead uses the inverse Hessian at each feature's penalized MAP
    readout as `Sigma_n` and has no fitted observation variance.

    The loading precisions do need one. `gamma_k` and `||C[:, k]||^2` both
    vanish together for a column that is being pruned, so the bare rule is a
    `0 / 0` limit; measured over a grid of problems it diverges or runs to
    precisions of order `1e17` in a majority of cases. `(a, b)` is a weak Gamma
    prior on the loading precisions, defaulting to the reference
    implementation's settings, which removes the corner. It is not a modelling
    claim so much as a guard on the updater.

    An irrelevant column's loadings may shrink toward numerical zero, but the
    column is not removed from ``C``. The fitted precision ranking can indicate
    effective dimensionality; it does not guarantee the true rank, and any
    threshold is the caller's decision.

    Gaussian ARD requires each feature to have more than ``K + 1`` observed
    time steps to keep the variance denominator positive. Poisson ARD requires
    each feature's observed total to be positive and its mean below
    ``exp(MAX_LOGRATE)``: otherwise the flat-bias loading integral diverges under
    the capped readout objective. The fixed-point iteration warm-starts from
    ``params.alpha`` when available, or the configured initial precision.
    """

    requires_conjugate_terms = True
    fits_ard_precisions = True

    def _supports_likelihood(self, likelihood: type[CVI]) -> bool:
        return issubclass(likelihood, Poisson) or super()._supports_likelihood(likelihood)

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
            Initial loading precision. Must be finite and strictly positive.
        alpha_shape : float, default=1.0 + 1e-6
            Shape of the Gamma prior on the loading precisions. In the
            reference ARD regression this is its `lambda_1 + 1`, and
            `alpha_rate` is `lambda_2`. Must be finite and greater than one
            so the fixed-point precision update stays positive.
        alpha_rate : float, default=1e-6
            Rate of the Gamma prior on the loading precisions. Must be finite
            and strictly positive.
        iterations : int, default=50
            Fixed-point iterations per readout update. For Poisson, each
            iteration uses three damped Newton steps for the readout MAP.
            Must be at least 1.
        """
        super().__init__(alpha)
        if not bool(jnp.all(self.alpha > 0)):
            raise ValueError("ARD alpha must be strictly positive")
        alpha_shape = float(alpha_shape)
        if not math.isfinite(alpha_shape) or alpha_shape <= 1:
            raise ValueError(
                f"alpha_shape must be finite and greater than one, got {alpha_shape!r}"
            )
        alpha_rate = float(alpha_rate)
        if not math.isfinite(alpha_rate) or alpha_rate <= 0:
            raise ValueError(
                f"alpha_rate must be finite and strictly positive, got {alpha_rate!r}"
            )
        iterations = int(iterations)
        if iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {iterations!r}")
        self.alpha_shape = alpha_shape
        self.alpha_rate = alpha_rate
        self.iterations = iterations

    def estimate(
        self, likelihood, params: Params, y: Array, valid_y: Array, m: Array, V: Array
    ) -> tuple[Params, float]:
        y, valid_y, m, V = _flattened(y, valid_y, m, V)
        terms = likelihood.loading_terms(y, valid_y, m, V)
        if terms is None:
            return _poisson_laplace_ard(
                likelihood, params, y, valid_y, m, V,
                initial_alpha=self.alpha, alpha_shape=self.alpha_shape,
                alpha_rate=self.alpha_rate, iterations=self.iterations,
            )
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

        initial_alpha = params.alpha if params.alpha is not None else self.alpha
        initial = (
            jnp.broadcast_to(initial_alpha, (n_latents,)),
            jnp.diag(params.R),
        )
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


def _poisson_laplace_ard(
    likelihood: type[Poisson],
    params: Params,
    y: Array,
    valid_y: Array,
    m: Array,
    V: Array,
    *,
    initial_alpha: Array,
    alpha_shape: float,
    alpha_rate: float,
    iterations: int,
) -> tuple[Params, Array]:
    """Fit Poisson loadings and shared precisions using a local Laplace approximation.

    Each feature has its own bias and loading MAP, while the loading precisions
    are shared across features. The fixed-point update treats the MAP Hessian as
    locally constant; its full derivative through the MAP is not included.
    """
    n_latents = m.shape[-1]
    beta = jnp.concatenate((params.d[:, None], params.C), axis=1)
    alpha = params.alpha if params.alpha is not None else initial_alpha
    alpha = jnp.broadcast_to(alpha, (n_latents,))

    def objective(beta_n, alpha_, y_n, mask_n):
        penalty = jnp.concatenate((jnp.zeros(1), alpha_))
        # The readout objective averages over observed steps. Undo that
        # averaging before forming the per-feature posterior curvature.
        return jnp.sum(mask_n) * likelihood.readout_objective(
            (beta_n[None, 1:], beta_n[:1]),
            y_n[:, None], mask_n[:, None], m, V, penalty,
        )

    gradient = jax.grad(objective)
    hessian = jax.hessian(objective)

    def newton_step(beta_n, alpha_, y_n, mask_n):
        grad = gradient(beta_n, alpha_, y_n, mask_n)
        curvature = hessian(beta_n, alpha_, y_n, mask_n)
        direction = jnp.linalg.solve(curvature, grad)
        scales = 0.5 ** jnp.arange(15)
        candidates = beta_n[None, :] - scales[:, None] * direction[None, :]
        losses = jax.vmap(lambda candidate: objective(candidate, alpha_, y_n, mask_n))(
            candidates
        )
        best = jnp.argmin(losses)
        return jnp.where(
            losses[best] < objective(beta_n, alpha_, y_n, mask_n),
            candidates[best], beta_n,
        )

    newton = jax.vmap(newton_step, in_axes=(0, None, 1, 1))
    curvatures = jax.vmap(hessian, in_axes=(0, None, 1, 1))

    def update(carry, _):
        beta_, alpha_ = carry
        beta_ = jax.lax.fori_loop(
            0, 3, lambda _, value: newton(value, alpha_, y, valid_y), beta_
        )
        curvature = curvatures(beta_, alpha_, y, valid_y)
        covariance = jnp.linalg.solve(
            curvature, jnp.broadcast_to(jnp.eye(n_latents + 1), curvature.shape)
        )
        effective = 1.0 - alpha_[None, :] * jnp.diagonal(
            covariance, axis1=1, axis2=2
        )[:, 1:]
        alpha_ = (jnp.sum(effective, axis=0) + 2.0 * alpha_shape - 2.0) / (
            jnp.sum(beta_[:, 1:] ** 2, axis=0) + 2.0 * alpha_rate
        )
        return (beta_, alpha_), None

    (beta, alpha), _ = jax.lax.scan(update, (beta, alpha), None, length=iterations)
    # Refine the loadings at the final precision before evaluating the evidence.
    beta = jax.lax.fori_loop(
        0, 9, lambda _, value: newton(value, alpha, y, valid_y), beta
    )
    curvature = curvatures(beta, alpha, y, valid_y)
    chol = jnp.linalg.cholesky(curvature)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diagonal(chol, axis1=1, axis2=2)), axis=1)
    objective_values = jax.vmap(objective, in_axes=(0, None, 1, 1))(
        beta, alpha, y, valid_y
    )
    evidence = (
        -jnp.sum(objective_values)
        + 0.5 * y.shape[-1] * (jnp.sum(jnp.log(alpha)) + jnp.log(2.0 * jnp.pi))
        - 0.5 * jnp.sum(logdet)
        + (alpha_shape - 1.0) * jnp.sum(jnp.log(alpha))
        - alpha_rate * jnp.sum(alpha)
    )
    return Params(C=beta[:, 1:], d=beta[:, 0], R=params.R, alpha=alpha), evidence


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
