from abc import abstractmethod
from functools import partial
from typing import ClassVar, override

from sklearn.decomposition import FactorAnalysis

from jax import Array, grad, numpy as jnp, vmap
from .readout import (
    NONFINITE_EVIDENCE,
    NONFINITE_PRECISION,
    NONFINITE_VARIANCE,
    READOUT_OK,
    LoadingTerms,
    Params,
    ard_fit_conjugate,
    ard_fit_laplace,
    conjugate_solve,
    fitted_alpha,
    flattened,
    gradient_ridge_fit,
    penalty,
)
from .utils import lbfgs_solve, sequence_info_repr


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


class CVI:
    """Base class for conjugate variational inference readouts.

    Subclasses register by name in `CVI.registry` for lookup via `observation`.
    A readout supplies the latent E-step and its own readout M-step, so the
    loading kind it accepts and the way it is fitted live together here.

    The readout hooks (``readout_objective``, ``solve_ridge``, ``solve_ard``)
    receive the posterior and data flattened over sequences and time by
    :meth:`solve_readout`: ``y`` and ``valid_y`` are ``(samples, features)``,
    ``m`` is ``(samples, K)`` and ``V`` is ``(samples, K, K)``.
    """

    #: Loading kinds this readout can fit. ``"ridge"`` and ``"fixed"`` are
    #: generic; a readout adds ``"ard"`` only when it implements the evidence.
    supported_loadings: ClassVar[frozenset[str]] = frozenset({"ridge", "fixed"})

    registry: ClassVar[dict] = dict()

    def __init_subclass__(cls, *args, **kwargs):
        """Register subclasses for lookup by likelihood name."""
        super().__init_subclass__(*args, **kwargs)
        CVI.registry[cls.__name__] = cls

    @classmethod
    def readout_objective(
        cls, params, y: Array, valid_y: Array, m: Array, V: Array, penalty: Array
    ):
        """Scalar readout objective including the loading penalty.

        Used by the generic ridge fit and by readouts that fit ARD through a
        Laplace approximation. Average over **observed time steps**, not over entries. ARD undoes this
        averaging to recover a per-feature curvature, so a different convention
        silently rescales its Hessian and evidence. The penalty the readout
        receives is in the same units as its data term, with one entry per
        coefficient of ``[d, C]`` and index 0 the bias.

        Parameters
        ----------
        params : tuple[Array, Array]
            Loading matrix and bias vector `(C, d)`.
        y : Array
            Observations shaped `(samples, obs_dim (N))`.
        valid_y : Array
            Observation mask aligned with `y`.
        m : Array
            Posterior means shaped `(samples, latent_dim (K))`.
        V : Array
            Posterior covariances shaped `(samples, latent_dim (K), latent_dim (K))`.
        penalty : Array
            Diagonal loading penalty shaped `(latent_dim (K) + 1,)`.

        Returns
        -------
        Array
            Scalar objective.
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement readout_objective for the generic ridge fit"
        )

    @classmethod
    def solve_readout(
        cls, loading: str, settings: dict, params: Params, y: Array, valid_y: Array,
        m: Array, V: Array,
    ) -> tuple[Params, float | Array]:
        """Fit the readout at the given loading kind.

        Flattens the posterior over sequences and time once, then dispatches:
        ``"fixed"`` keeps ``params``, ``"ridge"`` calls :meth:`solve_ridge`, and
        ``"ard"`` calls :meth:`solve_ard`.

        The second return value is a display diagnostic whose meaning depends on
        the kind and readout: NaN when none exists, the unpenalised negative
        expected log-likelihood (lower is better) for a gradient ridge fit, and
        the conditional evidence (higher is better) for ARD. Only
        :meth:`readout_failure` decides whether a fit is valid.
        """
        if loading == "fixed":
            return params, jnp.nan
        y, valid_y, m, V = flattened(y, valid_y, m, V)
        if loading == "ridge":
            return cls.solve_ridge(settings, params, y, valid_y, m, V)
        return cls.solve_ard(settings, params, y, valid_y, m, V)

    @classmethod
    def solve_ridge(
        cls, settings: dict, params: Params, y: Array, valid_y: Array, m: Array, V: Array
    ) -> tuple[Params, float | Array]:
        """Ridge readout M-step; L-BFGS on :meth:`readout_objective` by default.

        A readout with a closed form overrides this, as :class:`Gaussian` does.
        """
        return gradient_ridge_fit(
            cls.readout_objective, params, y, valid_y, m, V, settings["alpha"]
        )

    @classmethod
    def solve_ard(
        cls, settings: dict, params: Params, y: Array, valid_y: Array, m: Array, V: Array
    ) -> tuple[Params, float | Array]:
        """ARD readout M-step; overridden by readouts that declare ``"ard"``."""
        raise NotImplementedError(
            f"{cls.__name__} declares ARD but does not implement solve_ard"
        )

    @classmethod
    def check_ard_fit(cls, y: Array, valid_y: Array, n_components: int) -> None:
        """Constraint ARD imposes on the data; none by default."""

    @classmethod
    def readout_failure(
        cls, loading: str, params: Params, score: float | Array
    ) -> Array:
        """Failure code for a freshly fitted readout; ``READOUT_OK`` when healthy.

        Called inside the traced EM loop, so it must be array-valued. The base
        checks what every ARD fit shares, the precisions and the evidence; a
        readout with fitted noise adds its own checks on top.
        """
        if loading != "ard":
            return jnp.asarray(READOUT_OK)
        alpha = fitted_alpha(params)
        bad_precision = jnp.any(~jnp.isfinite(alpha) | (alpha <= 0))
        return jnp.where(
            bad_precision,
            NONFINITE_PRECISION,
            jnp.where(~jnp.isfinite(score), NONFINITE_EVIDENCE, READOUT_OK),
        )

    @classmethod
    def check_params(cls, params: Params, n_features: int) -> None:
        """Validate caller-supplied readout parameters; none by default."""

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
        penalty: Array | None = None,
        params=None,
    ):
        """Create the initial readout parameter state.

        The data-driven start belongs to the observation model; the loading
        calls this and supplies the loading penalty. When ``params`` is
        provided, return it directly and skip initialisation. This allows
        callers to supply pre-set parameters (e.g. the true readout in a
        simulation).

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
        penalty : Array, optional
            Diagonal loading penalty shaped ``(latent_dim (K) + 1,)``, with index
            0 the bias, for readouts whose start is refined against their
            objective.
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

    supported_loadings = frozenset({"ridge", "ard", "fixed"})

    @classmethod
    def loading_terms(
        cls, y: Array, valid_y: Array, m: Array, V: Array
    ) -> LoadingTerms:
        """Conjugate representation of the Gaussian readout; see :class:`LoadingTerms`.

        Arguments are flattened over sequences and time. The bias is index 0.
        """
        design = jnp.concatenate((jnp.ones((m.shape[0], 1), dtype=m.dtype), m), axis=1)
        outer = design[:, :, None] * design[:, None, :]
        outer = outer.at[:, 1:, 1:].add(V)
        observed = valid_y.astype(m.dtype)
        return LoadingTerms(
            precision=jnp.einsum("tn,tij->nij", observed, outer),
            linear=jnp.einsum("tn,ti->ni", jnp.where(valid_y, y, 0.0), design),
            counts=jnp.sum(observed, axis=0),
            squared=jnp.sum(jnp.where(valid_y, y**2, 0.0), axis=0),
        )

    @classmethod
    @override
    def solve_ridge(
        cls, settings: dict, params: Params, y: Array, valid_y: Array, m: Array, V: Array
    ) -> tuple[Params, float | Array]:
        """Closed-form ridge solve; the loadings are conjugate to the Gaussian."""
        terms = cls.loading_terms(y, valid_y, m, V)
        penalty_diagonal = penalty(settings["alpha"], m.shape[-1])
        return conjugate_solve(terms, penalty_diagonal, params, y, valid_y, m, V), jnp.nan

    @classmethod
    @override
    def solve_ard(
        cls, settings: dict, params: Params, y: Array, valid_y: Array, m: Array, V: Array
    ) -> tuple[Params, float | Array]:
        """Fit Gaussian loadings against the exact conditional evidence."""
        terms = cls.loading_terms(y, valid_y, m, V)
        return ard_fit_conjugate(
            terms, params, y, valid_y, m, V,
            alpha=fitted_alpha(params),
            alpha_shape=settings["alpha_shape"],
            alpha_rate=settings["alpha_rate"],
            iterations=settings["iterations"],
        )

    @classmethod
    @override
    def readout_failure(
        cls, loading: str, params: Params, score: float | Array
    ) -> Array:
        """Add the check on the fitted observation variances."""
        variance = jnp.diag(params.R)
        bad_variance = jnp.any(~jnp.isfinite(variance) | (variance <= 0))
        return jnp.where(
            bad_variance,
            NONFINITE_VARIANCE,
            super().readout_failure(loading, params, score),
        )

    @classmethod
    @override
    def check_params(cls, params: Params, n_features: int) -> None:
        """Validate a caller-supplied diagonal observation covariance."""
        R = jnp.asarray(params.R)
        if R.shape != (n_features, n_features):
            raise ValueError("Gaussian R must have shape (features, features)")
        diagonal = jnp.diag(R)
        if not bool(jnp.all(jnp.isfinite(diagonal) & (diagonal > 0))) or not bool(
            jnp.all(R == jnp.diag(diagonal))
        ):
            raise ValueError("Gaussian R must be a positive diagonal covariance")

    @classmethod
    @override
    def check_ard_fit(cls, y: Array, valid_y: Array, n_components: int) -> None:
        """ARD needs more observations than readout parameters per feature."""
        counts = jnp.sum(valid_y, axis=(0, 1))
        minimum = n_components + 1
        if not bool(jnp.all(counts > minimum)):
            raise ValueError(
                "ARD requires every feature to be observed at more than "
                f"n_components + 1 = {minimum} time steps; "
                f"minimum observed count is {int(jnp.min(counts))}"
            )

    @classmethod
    @override
    def initialize_params(
        cls,
        y: Array,
        valid_y: Array,
        n_factors: int,
        *,
        random_state: int,
        penalty: Array | None = None,
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
        penalty : Array, optional
            Loading penalty supplied by the fit. Unused here: the
            Gaussian start is data-driven and is refined by the selected
            loading kind.
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


def _capped_exp(x: Array) -> Array:
    """``exp(x)`` up to ``MAX_LOGRATE``, then its tangent line there.

    Limits exponential growth without flattening the rate. A flat clip would
    leave ``-y * lin`` free to fall without limit once an optimiser step crosses
    the cap. The tangent is convex and matches the value and first derivative
    of ``exp`` at the cap. It does not bound the rate or ensure a bounded
    objective when the observed mean count is at least ``exp(MAX_LOGRATE)``.
    """
    cap = jnp.exp(MAX_LOGRATE)
    below = x <= MAX_LOGRATE
    # The inner where keeps exp finite on the unused branch; jnp.minimum would do
    # the same but splits the gradient between its arguments at x == MAX_LOGRATE.
    return jnp.where(
        below,
        jnp.exp(jnp.where(below, x, 0.0)),
        cap * (1.0 + x - MAX_LOGRATE),
    )


def poisson_readout_objective(
    params: tuple[Array, Array],
    y: Array,
    valid_y: Array,
    m: Array,
    V: Array,
    penalty: Array,
) -> float | Array:
    """Average Poisson readout objective per observed time step.

    With ``eta = C m + d`` and ``q = diag(C V C^T)``, the Gaussian expectation
    of ``y * log(rate) - rate`` is ``y * eta - exp(eta + q / 2)``; only the rate
    carries the variance correction. Moving it into the ``y`` term as well
    adds ``y q / 2``, which rewards inflating ``C`` and makes ``C`` drift along
    the latent-scale gauge without converging. The expected log-likelihood is
    exact while the cap is inactive; above it the rate follows a tangent line,
    see :func:`_capped_exp`.

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
    penalty : Array
        Diagonal loading precision shaped ``(latent_dim (K) + 1,)``, with
        index 0 the bias. The Gaussian prior contributes ``0.5 * alpha * C²``.

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
        rate = _capped_exp(lin + 0.5 * quad)
        observed_y = jnp.where(mask_t, y_t, 0.0)
        return jnp.sum(jnp.where(mask_t, rate - lin * observed_y, 0.0))

    step_objectives = vmap(step_objective)(y, valid_y, m, V)
    # One precision per latent column, broadcast across features.
    total = jnp.sum(step_objectives) + 0.5 * jnp.sum(penalty[1:][None, :] * C**2)
    return total / n_valid_steps


def poisson_cvi_step_stats_latent(
    m: Array,
    V: Array,
    y: Array,
    valid_y: Array,
    C: Array,
    d: Array,
) -> tuple[Array, Array]:
    """Poisson CVI gradients for a single time step in latent space.

    Works directly with latent-space moments ``(m, V)`` and loading ``C``.

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
    # The local expansion needs the rate's derivative, which above the cap stays
    # at the tangent slope while the rate itself keeps growing. Taking it from
    # ``_capped_exp`` keeps the E-step and the readout objective from drifting
    # apart; only this derivative (never the rate) enters the gradient and the
    # information matrix.
    rate_deriv = vmap(grad(_capped_exp))(lin + 0.5 * quad)

    observed_y = jnp.where(valid_y, y, 0.0)
    observed_deriv = jnp.where(valid_y, rate_deriv, 0.0)
    grad_m = (observed_y - observed_deriv) @ C
    K = jnp.einsum("ni,n,nj->ij", C, observed_deriv, C)
    return grad_m + K @ m, K


class Poisson(CVI):
    """CVI readout for Poisson observations with exponential link."""

    @classmethod
    @override
    def readout_objective(
        cls, params, y: Array, valid_y: Array, m: Array, V: Array, penalty: Array
    ) -> float | Array:
        """Average Poisson readout objective per observed time step."""
        return poisson_readout_objective(params, y, valid_y, m, V, penalty)

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

    supported_loadings = frozenset({"ridge", "ard", "fixed"})

    @classmethod
    @override
    def solve_ard(
        cls, settings: dict, params: Params, y: Array, valid_y: Array, m: Array, V: Array
    ) -> tuple[Params, float | Array]:
        """Fit Poisson loadings against a Laplace approximation of the evidence."""
        return ard_fit_laplace(
            cls.readout_objective, params, y, valid_y, m, V,
            alpha_shape=settings["alpha_shape"],
            alpha_rate=settings["alpha_rate"],
            iterations=settings["iterations"],
        )

    @classmethod
    @override
    def check_ard_fit(cls, y: Array, valid_y: Array, n_components: int) -> None:
        """The flat-bias loading integral needs bounded, positive counts."""
        counts = jnp.sum(valid_y, axis=(0, 1))
        totals = jnp.sum(y, axis=(0, 1))
        if not bool(jnp.all((totals > 0) & (totals < jnp.exp(MAX_LOGRATE) * counts))):
            raise ValueError(
                "Poisson ARD requires positive total counts and a mean count "
                "below exp(MAX_LOGRATE) for every feature"
            )

    @classmethod
    def initialize_params(
        cls,
        y: Array,
        valid_y: Array,
        n_factors: int,
        *,
        random_state: int,
        penalty: Array | None = None,
        params: Params | None = None,
    ) -> Params:
        """Initialise Poisson readout via FA followed by LBFGS optimisation.

        Parameters
        ----------
        y : Array
            Observation tensor shaped ``(sequences, time, obs_dim)``; leading
            axes are flattened.
        valid_y : Array
            Observation mask aligned with ``y``.
        n_factors : int
            Number of latent factors.
        random_state : int
            Seed used for factor analysis.
        penalty : Array, optional
            Diagonal loading penalty shaped ``(latent_dim (K) + 1,)``, with index
            0 the bias. Defaults to no penalty.
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
        if penalty is None:
            penalty = jnp.zeros(n_factors + 1)
        C, d = lbfgs_solve(
            (C, d),
            partial(
                poisson_readout_objective, y=y, valid_y=valid_y, m=m, V=V,
                penalty=penalty,
            ),
        )

        return Params(C=C, d=d, R=jnp.asarray(0.0))
