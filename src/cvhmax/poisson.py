from functools import partial
from typing import override

from jax import Array, grad, numpy as jnp, vmap

from .cvi import CVI, fa_init, initialization_data
from .readout import Params, ard_fit_laplace
from .utils import lbfgs_solve

MAX_LOGRATE = 7.0


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

        y, valid_y, filled = initialization_data(y, valid_y)
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
