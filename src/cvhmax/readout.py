"""Readout fitting: penalty convention, solvers, and loading evidence.

These are pure functions over a likelihood's data term. The observation model
decides *which* fit to run, per loading kind; the mechanics live here so the
same code serves every observation model.

All functions expect leading sequence and time axes already collapsed by
:func:`flattened`; the readout is fitted per feature across all observed steps.
"""

import math
from typing import TYPE_CHECKING, Any, NamedTuple

import equinox as eqx
import jax
import numpy as np
from jax import Array, numpy as jnp

from .utils import lbfgs_solve

if TYPE_CHECKING:
    # Import-cycle guard: ``cvi`` imports this module at runtime, so the readout
    # type is only named here for annotations.
    from .cvi import CVI

# Damping on the local curvature. A feature whose likelihood is locally flat
# (saturated probabilities, empty counts) otherwise yields a singular Hessian,
# which breaks both the Newton step and the evidence log-determinant.
CURVATURE_JITTER = 1e-10

# Work per Laplace precision update. Three damped Newton steps track the moving
# loading mode closely enough that the MacKay step sees a settled curvature;
# nine further steps refine the mode once, before the evidence is evaluated. A
# halving ladder of fifteen scales is enough backtracking for the objectives
# these readouts expose.
NEWTON_STEPS_PER_UPDATE = 3
EVIDENCE_SETTLE_STEPS = 9
BACKTRACKING_STEPS = 15

# Per-kind defaults. ``alpha`` is held fixed for ridge and fixed readouts and is
# only a starting point for ARD, so the two kinds keep their own default.
DEFAULTS: dict[str, dict[str, Any]] = {
    "ridge": {"alpha": 0.1},
    "ard": {"alpha": 1.0, "alpha_shape": 1.0 + 1e-6, "alpha_rate": 1e-6, "iterations": 50},
    "fixed": {"alpha": 0.1},
}


class Params(eqx.Module):
    """Container of readout parameters.

    Shared by the built-in Gaussian and Poisson readouts. CVHM's convergence
    metric compares ``C``, ``d``, and ``R``; ARD also compares the loading
    basis and ``alpha``.

    Attributes
    ----------
    C : Array
        Loading matrix with shape ``(obs_dim (N), latent_dim (K))``.
    d : Array
        Bias vector with shape ``(obs_dim (N),)``.
    R : Array
        Diagonal Gaussian observation covariance with shape ``(N, N)``.
        For Poisson readouts, the scalar sentinel ``jnp.asarray(0.0)``.
    alpha : Array or None
        Loading precisions with shape ``(latent_dim (K),)`` when supplied by
        the loading kind, otherwise ``None``. For ARD, a large value
        indicates stronger shrinkage of a loading column; the precision ranking
        can help assess effective dimensionality but does not guarantee the
        true rank.
    """

    C: Array
    d: Array
    R: Array
    alpha: Array | None = None


class LoadingTerms(NamedTuple):
    """Conjugate representation of a readout's dependence on ``(d, C)``.

    The expected readout log-likelihood is ``-0.5 beta' precision beta
    + beta' linear`` up to terms independent of ``beta = [d, C]``, which is what
    makes a Gaussian loading prior conjugate and lets a fit integrate
    the loadings out rather than profile them. The bias is index 0.

    Attributes
    ----------
    precision : Array
        Shape ``(features (N), K + 1, K + 1)``.
    linear : Array
        Shape ``(N, K + 1)``.
    counts : Array
        Observed count per feature, shape ``(N,)``.
    squared : Array
        Sum of squared observations per feature, shape ``(N,)``, needed for the
        data-dependent Gaussian normalisation.
    """

    precision: Array
    linear: Array
    counts: Array
    squared: Array


def portable_json(value: Any, name: str) -> Any:
    """Convert one loading-configuration value to finite JSON primitives."""
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
        return [portable_json(item, name) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError(f"{name} keys must be strings")
        return {key: portable_json(item, f"{name}.{key}") for key, item in value.items()}
    raise ValueError(
        f"{name} has unsupported type {type(value).__name__}; "
        "use JSON-compatible scalars, lists, or arrays"
    )


def portable_config(config: dict | None) -> dict | None:
    """Return loading-configuration keywords as finite JSON primitives."""
    if config is None:
        return None
    if not isinstance(config, dict):
        raise ValueError("loading_config must be a dict or None")
    return portable_json(config, "loading_config")


def flattened(
    y: Array, valid_y: Array, m: Array, V: Array
) -> tuple[Array, Array, Array, Array]:
    """Collapse sequence and time into one axis."""
    y = y.reshape(-1, y.shape[-1])
    valid_y = valid_y.reshape(y.shape)
    m = m.reshape(-1, m.shape[-1])
    V = V.reshape(-1, *V.shape[-2:])
    return y, valid_y, m, V


def loading_settings(
    loading: str, config: dict | None, n_latents: int | None = None
) -> dict[str, Any]:
    """Merge user configuration over the defaults for one loading kind.

    Parameters
    ----------
    loading : str
        Loading kind, one of :data:`DEFAULTS`.
    config : dict or None
        Overrides for the kind's defaults.
    n_latents : int, optional
        When given, a per-column ``alpha`` must have exactly this length.
    """
    defaults = DEFAULTS[loading]
    settings = {**defaults, **(config or {})}
    unknown = set(settings) - set(defaults)
    if unknown:
        raise ValueError(
            f"unknown loading_config keys for {loading!r}: {sorted(unknown)}"
        )

    alpha = np.asarray(settings["alpha"], dtype=float)
    if alpha.ndim > 1:
        raise ValueError("alpha must be a scalar or one value per latent column")
    if n_latents is not None and alpha.shape not in ((), (n_latents,)):
        raise ValueError(
            f"alpha has shape {alpha.shape}; expected a scalar or ({n_latents},)"
        )
    positive = loading == "ard"
    if not np.all(np.isfinite(alpha)) or np.any(alpha < 0) or (positive and np.any(alpha == 0)):
        bound = "positive" if positive else "non-negative"
        raise ValueError(f"{loading!r} alpha must be finite and {bound}")

    if loading == "ard":
        shape = float(settings["alpha_shape"])
        rate = float(settings["alpha_rate"])
        iterations = float(settings["iterations"])
        if not math.isfinite(shape) or shape <= 1:
            raise ValueError(
                f"alpha_shape must be finite and greater than one, got {shape!r}"
            )
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError(f"alpha_rate must be finite and positive, got {rate!r}")
        if not math.isfinite(iterations) or iterations < 1 or not iterations.is_integer():
            raise ValueError(f"iterations must be an integer >= 1, got {iterations!r}")
        settings = {**settings, "iterations": int(iterations)}
    return settings


def penalty(alpha: Array, n_latents: int) -> Array:
    """Return the penalty diagonal, zero at the bias index."""
    return jnp.concatenate((jnp.zeros(1), jnp.broadcast_to(alpha, (n_latents,))))


def stack_readout(params: Params) -> Array:
    """Stack readout coefficients as ``[d, C]`` with the bias in column 0."""
    return jnp.concatenate((params.d[:, None], params.C), axis=1)


def split_readout(beta: Array) -> tuple[Array, Array]:
    """Split ``[d, C]`` coefficients into ``(C, d)``."""
    return beta[:, 1:], beta[:, 0]


def unstack_readout(beta: Array, R: Array, alpha: Array) -> Params:
    """Split ``[d, C]`` coefficients back into readout parameters."""
    C, d = split_readout(beta)
    return Params(C=C, d=d, R=R, alpha=alpha)


def mackay_update(
    alpha: Array, covariance_diagonal: Array, loadings: Array,
    alpha_shape: float, alpha_rate: float,
) -> Array:
    """Per-column precision after one MacKay fixed-point step.

    ``covariance_diagonal`` is ``(features, K + 1)`` including the unpenalised
    bias column, which is dropped here so callers never slice it themselves.
    """
    effective = 1.0 - alpha[None, :] * covariance_diagonal[:, 1:]
    denominator = jnp.sum(loadings**2, axis=0) + 2.0 * alpha_rate
    return (jnp.sum(effective, axis=0) + 2.0 * alpha_shape - 2.0) / denominator


def conjugate_solve(
    terms: LoadingTerms,
    penalty_diagonal: Array,
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
    posterior = terms.precision / variance[:, None, None] + jnp.diag(penalty_diagonal)[None]
    beta = jnp.linalg.solve(
        posterior, (terms.linear / variance[:, None])[..., None]
    )[..., 0]
    C, d = split_readout(beta)
    residual = jnp.where(valid_y, y - (m @ C.T + d), 0.0)
    uncertainty = jnp.einsum("nk,tkl,nl->tn", C, V, C)
    variance = jnp.sum(
        jnp.where(valid_y, residual**2 + uncertainty, 0.0), axis=0
    ) / terms.counts
    return unstack_readout(beta, jnp.diag(variance), penalty_diagonal[1:])


def ridge_fit(
    likelihood,
    params: Params,
    y: Array,
    valid_y: Array,
    m: Array,
    V: Array,
    alpha: Array,
) -> tuple[Params, Array]:
    """Fit loadings at a fixed precision, using a closed form where one exists."""
    penalty_diagonal = penalty(alpha, m.shape[-1])
    if likelihood.has_conjugate_loadings:
        terms = likelihood.loading_terms(y, valid_y, m, V)
        return conjugate_solve(terms, penalty_diagonal, params, y, valid_y, m, V), jnp.nan
    C, d = lbfgs_solve(
        (params.C, params.d),
        lambda readout: likelihood.readout_objective(
            readout, y, valid_y, m, V, penalty=penalty_diagonal
        ),
    )
    objective = likelihood.readout_objective(
        (C, d), y, valid_y, m, V, penalty=jnp.zeros_like(penalty_diagonal)
    )
    return Params(C=C, d=d, R=params.R, alpha=penalty_diagonal[1:]), objective


def ard_fit_conjugate(
    terms: LoadingTerms,
    params: Params,
    y: Array,
    valid_y: Array,
    m: Array,
    V: Array,
    *,
    alpha: Array,
    alpha_shape: float,
    alpha_rate: float,
    iterations: int,
) -> tuple[Params, Array]:
    """Fit precisions and observation variances against the exact evidence.

    Both stationary points have closed forms, so this is a fixed-point iteration
    rather than an optimisation. ``alpha`` is the starting precision.
    """
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
        loadings, bias = split_readout(beta)
        alpha_next = mackay_update(alpha_, diagonal, loadings, alpha_shape, alpha_rate)
        residual = jnp.where(valid_y, y - (m @ loadings.T + bias), 0.0)
        uncertainty = jnp.einsum("nk,tkl,nl->tn", loadings, V, loadings)
        rss = jnp.sum(jnp.where(valid_y, residual**2 + uncertainty, 0.0), axis=0)
        # The residual degrees of freedom remove each loading's effective count
        # plus one for the unpenalised bias.
        effective_count = 1.0 - alpha_[None, :] * diagonal[:, 1:]
        variance_next = rss / (
            terms.counts - 1.0 - jnp.sum(effective_count, axis=-1)
        )
        return (alpha_next, variance_next), None

    initial = (jnp.broadcast_to(alpha, (n_latents,)), jnp.diag(params.R))
    (alpha, variance), _ = jax.lax.scan(step, initial, None, length=iterations)

    _, beta, _ = solve(alpha, variance)
    evidence = ard_evidence(
        terms,
        jnp.log(alpha),
        jnp.log(variance),
        alpha_shape=alpha_shape,
        alpha_rate=alpha_rate,
    )
    return unstack_readout(beta, jnp.diag(variance), alpha), evidence


def ard_fit_laplace(
    likelihood: type["CVI"],
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
    """Fit loadings and shared precisions using a local Laplace approximation.

    Each feature has its own bias and loading MAP, while the loading precisions
    are shared across features. The fixed-point update treats the MAP Hessian as
    locally constant; its full derivative through the MAP is not included.
    """
    n_latents = m.shape[-1]
    beta = stack_readout(params)
    alpha = params.alpha if params.alpha is not None else initial_alpha
    alpha = jnp.broadcast_to(alpha, (n_latents,))

    def objective(beta_n, alpha_, y_n, mask_n):
        penalty_diagonal = jnp.concatenate((jnp.zeros(1), alpha_))
        # The readout objective averages over observed steps. Undo that
        # averaging before forming the per-feature posterior curvature.
        return jnp.sum(mask_n) * likelihood.readout_objective(
            split_readout(beta_n[None, :]),
            y_n[:, None], mask_n[:, None], m, V, penalty_diagonal,
        )

    gradient = jax.grad(objective)

    def curvature(beta_n, alpha_, y_n, mask_n):
        """Local curvature, damped so a flat feature stays positive definite."""
        hessian = jax.hessian(objective)(beta_n, alpha_, y_n, mask_n)
        return hessian + CURVATURE_JITTER * jnp.eye(hessian.shape[-1])

    def newton_step(beta_n, alpha_, y_n, mask_n):
        grad = gradient(beta_n, alpha_, y_n, mask_n)
        direction = jnp.linalg.solve(curvature(beta_n, alpha_, y_n, mask_n), grad)
        scales = 0.5 ** jnp.arange(BACKTRACKING_STEPS)
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
    curvatures = jax.vmap(curvature, in_axes=(0, None, 1, 1))

    def update(carry, _):
        beta_, alpha_ = carry
        beta_ = jax.lax.fori_loop(
            0,
            NEWTON_STEPS_PER_UPDATE,
            lambda _, value: newton(value, alpha_, y, valid_y),
            beta_,
        )
        local = curvatures(beta_, alpha_, y, valid_y)
        covariance = jnp.linalg.solve(
            local, jnp.broadcast_to(jnp.eye(n_latents + 1), local.shape)
        )
        loadings, _ = split_readout(beta_)
        alpha_ = mackay_update(
            alpha_, jnp.diagonal(covariance, axis1=1, axis2=2), loadings,
            alpha_shape, alpha_rate,
        )
        return (beta_, alpha_), None

    (beta, alpha), _ = jax.lax.scan(update, (beta, alpha), None, length=iterations)
    # Refine the loadings at the final precision before evaluating the evidence.
    beta = jax.lax.fori_loop(
        0,
        EVIDENCE_SETTLE_STEPS,
        lambda _, value: newton(value, alpha, y, valid_y),
        beta,
    )
    local = curvatures(beta, alpha, y, valid_y)
    _, logdet = jnp.linalg.slogdet(local)
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
    return unstack_readout(beta, params.R, alpha), evidence


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
