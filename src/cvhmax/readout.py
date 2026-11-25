"""Readout fitting: penalty convention, solvers, and loading evidence.

These are pure functions over a likelihood's data term. An estimator decides
*which* fit to run and with what hyperparameters; the mechanics live here so
that the same code serves every observation model.

All functions expect leading sequence and time axes already collapsed by
:func:`flattened`; the readout is fitted per feature across all observed steps.
"""

import math
from typing import Any

import jax
import numpy as np
from jax import Array, numpy as jnp

from .cvi import Params, Poisson
from .utils import LoadingTerms, lbfgs_solve

Penalty = Array


def portable_json(value: Any, name: str) -> Any:
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
    """Return estimator constructor keywords as finite JSON primitives."""
    if config is None:
        return None
    if not isinstance(config, dict):
        raise ValueError("estimator_config must be a dict or None")
    return portable_json(config, "estimator_config")


def flattened(
    y: Array, valid_y: Array, m: Array, V: Array
) -> tuple[Array, Array, Array, Array]:
    """Collapse sequence and time into one axis."""
    y = y.reshape(-1, y.shape[-1])
    valid_y = valid_y.reshape(y.shape)
    m = m.reshape(-1, m.shape[-1])
    V = V.reshape(-1, *V.shape[-2:])
    return y, valid_y, m, V


def penalty(alpha: Array, n_latents: int) -> Penalty:
    """Return the penalty diagonal, zero at the bias index."""
    return jnp.concatenate((jnp.zeros(1), jnp.broadcast_to(alpha, (n_latents,))))


def stack_readout(params: Params) -> Array:
    """Stack readout coefficients as ``[d, C]`` with the bias in column 0."""
    return jnp.concatenate((params.d[:, None], params.C), axis=1)


def unstack_readout(beta: Array, R: Array, alpha: Array) -> Params:
    """Split ``[d, C]`` coefficients back into readout parameters."""
    return Params(C=beta[:, 1:], d=beta[:, 0], R=R, alpha=alpha)


def conjugate_solve(
    terms: LoadingTerms,
    penalty_: Penalty,
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
    posterior = terms.precision / variance[:, None, None] + jnp.diag(penalty_)[None]
    beta = jnp.linalg.solve(
        posterior, (terms.linear / variance[:, None])[..., None]
    )[..., 0]
    C, d = beta[:, 1:], beta[:, 0]
    residual = jnp.where(valid_y, y - (m @ C.T + d), 0.0)
    uncertainty = jnp.einsum("nk,tkl,nl->tn", C, V, C)
    variance = jnp.sum(
        jnp.where(valid_y, residual**2 + uncertainty, 0.0), axis=0
    ) / terms.counts
    return Params(C=C, d=d, R=jnp.diag(variance), alpha=penalty_[1:])


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
    penalty_ = penalty(alpha, m.shape[-1])
    terms = likelihood.loading_terms(y, valid_y, m, V)
    if terms is not None:
        return conjugate_solve(terms, penalty_, params, y, valid_y, m, V), jnp.nan
    C, d = lbfgs_solve(
        (params.C, params.d),
        lambda readout: likelihood.readout_objective(
            readout, y, valid_y, m, V, penalty=penalty_
        ),
    )
    objective = likelihood.readout_objective(
        (C, d), y, valid_y, m, V, penalty=jnp.zeros_like(penalty_)
    )
    return Params(C=C, d=d, R=params.R, alpha=penalty_[1:]), objective


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
        loadings, bias = beta[:, 1:], beta[:, 0]
        effective = 1.0 - jnp.concatenate((jnp.zeros(1), alpha_))[None, :] * diagonal
        alpha_next = (
            jnp.sum(effective[:, 1:], axis=0) + 2.0 * alpha_shape - 2.0
        ) / (jnp.sum(loadings**2, axis=0) + 2.0 * alpha_rate)
        residual = jnp.where(valid_y, y - (m @ loadings.T + bias), 0.0)
        uncertainty = jnp.einsum("nk,tkl,nl->tn", loadings, V, loadings)
        rss = jnp.sum(jnp.where(valid_y, residual**2 + uncertainty, 0.0), axis=0)
        variance_next = rss / (terms.counts - jnp.sum(effective, axis=-1))
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
    return (
        Params(C=beta[:, 1:], d=beta[:, 0], R=jnp.diag(variance), alpha=alpha),
        evidence,
    )


def ard_fit_laplace(
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
    beta = stack_readout(params)
    alpha = params.alpha if params.alpha is not None else initial_alpha
    alpha = jnp.broadcast_to(alpha, (n_latents,))

    def objective(beta_n, alpha_, y_n, mask_n):
        penalty_ = jnp.concatenate((jnp.zeros(1), alpha_))
        # The readout objective averages over observed steps. Undo that
        # averaging before forming the per-feature posterior curvature.
        return jnp.sum(mask_n) * likelihood.readout_objective(
            (beta_n[None, 1:], beta_n[:1]),
            y_n[:, None], mask_n[:, None], m, V, penalty_,
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
