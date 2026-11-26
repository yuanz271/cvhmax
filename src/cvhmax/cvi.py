from typing import ClassVar

from sklearn.decomposition import FactorAnalysis

from jax import Array, numpy as jnp

from .readout import (
    NONFINITE_EVIDENCE,
    NONFINITE_PRECISION,
    READOUT_OK,
    Loading,
    Params,
    fitted_alpha,
    flattened,
    gradient_ridge_fit,
)


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


def initialization_data(y: Array, valid_y: Array) -> tuple[Array, Array, Array]:
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
    Registration happens on import; the package imports the built-in ones.
    A readout supplies the latent E-step and its own readout M-step, so the
    loading kind it accepts and the way it is fitted live together here.

    The readout hooks (``readout_objective``, ``solve_ridge``, ``solve_ard``)
    receive the posterior and data flattened over sequences and time by
    :meth:`solve_readout`: ``y`` and ``valid_y`` are ``(samples, features)``,
    ``m`` is ``(samples, K)`` and ``V`` is ``(samples, K, K)``.
    """

    #: Loading kinds this readout can fit. ``"ridge"`` and ``"fixed"`` are
    #: generic; a readout adds ``"ard"`` only when it implements the evidence.
    supported_loadings: ClassVar[frozenset[Loading]] = frozenset({"ridge", "fixed"})

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
        cls, loading: Loading, settings: dict, params: Params, y: Array, valid_y: Array,
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
        cls, loading: Loading, params: Params, score: float | Array
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
        raise NotImplementedError(f"{cls.__name__} must implement update_pseudo")

    @classmethod
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
        raise NotImplementedError(f"{cls.__name__} must implement initialize_info")

    @classmethod
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

        An initialised readout must start with non-zero loadings. With ``C = 0`` the E-step returns
        zero latents, so the loading gradient vanishes and EM cannot leave that
        corner; the built-in readouts start from factor analysis for this reason.

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
        raise NotImplementedError(f"{cls.__name__} must implement initialize_params")
