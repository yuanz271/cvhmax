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
from typing import ClassVar

import numpy as np
from jax import Array, numpy as jnp

from .cvi import CVI, Params, Poisson
from .readout import (
    Penalty,
    ard_fit_conjugate,
    ard_fit_laplace,
    flattened,
    penalty,
    ridge_fit,
)


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
        return penalty(self.alpha, n_latents)

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
        y, valid_y, m, V = flattened(y, valid_y, m, V)
        return ridge_fit(likelihood, params, y, valid_y, m, V, self.alpha)


class FixedEstimator(Estimator):
    """Keep the initialized readout fixed while inferring the latents.

    Supplied ``C``, ``d``, and ``R`` are preserved; otherwise the observation
    model initializes them from the data before they are held fixed.
    """

    def estimate(
        self, likelihood, params: Params, y: Array, valid_y: Array, m: Array, V: Array
    ) -> tuple[Params, float]:
        return params, jnp.nan


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
        y, valid_y, m, V = flattened(y, valid_y, m, V)
        terms = likelihood.loading_terms(y, valid_y, m, V)
        if terms is None:
            return ard_fit_laplace(
                likelihood, params, y, valid_y, m, V,
                initial_alpha=self.alpha, alpha_shape=self.alpha_shape,
                alpha_rate=self.alpha_rate, iterations=self.iterations,
            )
        return ard_fit_conjugate(
            terms, params, y, valid_y, m, V,
            alpha=self.alpha if params.alpha is None else params.alpha,
            alpha_shape=self.alpha_shape, alpha_rate=self.alpha_rate,
            iterations=self.iterations,
        )
