from abc import abstractmethod
from functools import partial
from typing import ClassVar, override

from sklearn.decomposition import FactorAnalysis

from jax import Array, numpy as jnp, vmap
from jax.numpy.linalg import multi_dot
from jax.scipy.linalg import cho_factor, cho_solve
from .readout import (
    LoadingTerms,
    Params,
    ard_fit_conjugate,
    ard_fit_laplace,
    flattened,
    ridge_fit,
)
from .utils import lbfgs_solve, sequence_info_repr


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


class CVI:
    """Base class for conjugate variational inference readouts.

    Subclasses register by name in `CVI.registry` for lookup via `observation`.
    A readout supplies the latent E-step and its own readout M-step, so the
    loading kind it accepts and the way it is fitted live together here.
    """

    #: Loading kinds this readout can fit. ``"ridge"`` and ``"fixed"`` are
    #: generic; a readout adds ``"ard"`` only when it implements the evidence.
    supported_loadings: ClassVar[frozenset[str]] = frozenset({"ridge", "fixed"})

    #: Whether :meth:`loading_terms` is implemented. Declared rather than
    #: inferred so a conjugate readout is named in one place, and ``None`` never
    #: has to travel as a sentinel for "not supported".
    has_conjugate_loadings: ClassVar[bool] = False

    registry: ClassVar[dict] = dict()

    def __init_subclass__(cls, *args, **kwargs):
        """Register subclasses for lookup by likelihood name."""
        super().__init_subclass__(*args, **kwargs)
        CVI.registry[cls.__name__] = cls

    @classmethod
    def loading_terms(
        cls, y: Array, valid_y: Array, m: Array, V: Array
    ) -> LoadingTerms:
        """Conjugate loading representation; requires :attr:`has_conjugate_loadings`.

        A readout that exposes this makes its dependence on the loadings
        quadratic, which is what lets a fit integrate them out instead of
        profiling them. It is called only when the readout declares
        :attr:`has_conjugate_loadings`, so a readout that does not implement it
        never reaches here.

        Parameters
        ----------
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
        LoadingTerms
            Conjugate representation.
        """
        raise NotImplementedError(
            f"{cls.__name__} declares no conjugate loading representation"
        )

    @classmethod
    def readout_objective(
        cls, params, y: Array, valid_y: Array, m: Array, V: Array, penalty: Array
    ):
        """Scalar readout objective including the loading penalty.

        Average over **observed time steps**, not over entries. ARD undoes this
        averaging to recover a per-feature curvature, so a different convention
        silently rescales its Hessian and evidence. The penalty the readout
        receives is in the same units as its data term, with one entry per
        coefficient of ``[d, C]`` and index 0 the bias.

        Parameters
        ----------
        params : tuple[Array, Array]
            Loading matrix and bias vector `(C, d)`.
        y : Array
            Observations shaped `(sequences, time, obs_dim (N))`.
        valid_y : Array
            Observation mask aligned with `y`.
        m : Array
            Posterior means shaped `(sequences, time, latent_dim (K))`.
        V : Array
            Posterior covariances shaped `(sequences, time, latent_dim (K), latent_dim (K))`.
        penalty : Array
            Diagonal loading penalty shaped `(latent_dim (K) + 1,)`.

        Returns
        -------
        Array
            Scalar objective.
        """
        raise NotImplementedError(
            f"{cls.__name__} exposes neither loading_terms nor a readout objective"
        )

    @classmethod
    def solve_readout(
        cls, loading: str, settings: dict, params: Params, y: Array, valid_y: Array,
        m: Array, V: Array,
    ) -> tuple[Params, float | Array]:
        """Fit the readout at the given loading kind.

        ``"ridge"`` and ``"fixed"`` are handled here for every readout. ``"ard"``
        dispatches to :meth:`solve_ard`, so a readout that declares it overrides
        that hook and nothing else.
        """
        if loading == "fixed":
            return params, jnp.nan
        if loading == "ridge":
            y, valid_y, m, V = flattened(y, valid_y, m, V)
            return ridge_fit(cls, params, y, valid_y, m, V, settings["alpha"])
        return cls.solve_ard(settings, params, y, valid_y, m, V)

    @classmethod
    def solve_ard(
        cls, settings: dict, params: Params, y: Array, valid_y: Array, m: Array, V: Array
    ) -> tuple[Params, float | Array]:
        """ARD readout M-step; overridden by readouts that declare ``"ard"``."""
        raise NotImplementedError(
            f"{cls.__name__} declares ARD but does not implement solve_ard"
        )

    @classmethod
    def check_readout_fit(
        cls, loading: str, y: Array, valid_y: Array, n_components: int
    ) -> None:
        """Validate that this readout and loading kind can be fitted to the data."""
        if loading == "ard":
            cls.check_ard_fit(y, valid_y, n_components)

    @classmethod
    def check_ard_fit(cls, y: Array, valid_y: Array, n_components: int) -> None:
        """Constraint ARD imposes on the data; none by default."""

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


def _gaussian_loading_moments(
    y: Array, valid_y: Array, m: Array, V: Array
) -> LoadingTerms:
    """Conjugate loading representation of the Gaussian readout.

    See :class:`~cvhmax.readout.LoadingTerms`. The bias is index 0.
    """
    n_time = m.shape[0]
    design = jnp.concatenate((jnp.ones((n_time, 1), dtype=m.dtype), m), axis=1)
    outer = design[:, :, None] * design[:, None, :]
    outer = outer.at[:, 1:, 1:].add(V)
    observed = valid_y.astype(m.dtype)
    return LoadingTerms(
        precision=jnp.einsum("tn,tij->nij", observed, outer),
        linear=jnp.einsum("tn,ti->ni", jnp.where(valid_y, y, 0.0), design),
        counts=jnp.sum(observed, axis=0),
        squared=jnp.sum(jnp.where(valid_y, y**2, 0.0), axis=0),
    )


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
    def loading_terms(
        cls, y: Array, valid_y: Array, m: Array, V: Array
    ) -> LoadingTerms:
        """Conjugate Gaussian loading representation."""
        y = y.reshape(-1, y.shape[-1])
        valid_y = valid_y.reshape(y.shape)
        m = m.reshape(-1, m.shape[-1])
        V = V.reshape(-1, *V.shape[-2:])
        return _gaussian_loading_moments(y, valid_y, m, V)

    supported_loadings = frozenset({"ridge", "ard", "fixed"})
    has_conjugate_loadings = True

    @classmethod
    @override
    def solve_ard(
        cls, settings: dict, params: Params, y: Array, valid_y: Array, m: Array, V: Array
    ) -> tuple[Params, float | Array]:
        """Fit Gaussian loadings against the exact conditional evidence."""
        y, valid_y, m, V = flattened(y, valid_y, m, V)
        terms = cls.loading_terms(y, valid_y, m, V)
        return ard_fit_conjugate(
            terms, params, y, valid_y, m, V,
            alpha=params.alpha if params.alpha is not None else settings["alpha"],
            alpha_shape=settings["alpha_shape"],
            alpha_rate=settings["alpha_rate"],
            iterations=settings["iterations"],
        )

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
    y = y.reshape(-1, y.shape[-1])
    valid_y = valid_y.reshape(y.shape)
    m = m.reshape(-1, m.shape[-1])
    V = V.reshape(-1, *V.shape[-2:])
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
        y, valid_y, m, V = flattened(y, valid_y, m, V)
        return ard_fit_laplace(
            cls, params, y, valid_y, m, V,
            initial_alpha=settings["alpha"],
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
