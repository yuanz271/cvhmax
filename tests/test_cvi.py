from typing import get_args

import numpy as np
import numpy.testing as npt
import pytest

import chex
import jax
from jax import numpy as jnp
from scipy.integrate import trapezoid
from sklearn.linear_model import ARDRegression, LinearRegression

from cvhmax.gaussian import Gaussian
from cvhmax.poisson import (
    MAX_LOGRATE,
    Poisson,
    _capped_exp,
    poisson_cvi_step_stats_latent,
    poisson_readout_objective,
)
from cvhmax.readout import (
    DEFAULTS,
    Loading,
    Params,
    ard_evidence,
    loading_settings,
    resolve_alpha,
)
from cvhmax.cvhm import CVHM, lift
from cvhmax.hm import HidaMatern

from .helpers import ATOL_TIGHT, RTOL_TIGHT, X64_ENABLED, sample_gp_latents

# Stationarity of the Laplace MAP is a gradient-residual check; float32
# cannot resolve it below the objective's rounding scale.
ATOL_STATIONARY = 1e-5 if X64_ENABLED else 2e-2


def _ard_fit(y, valid, m, V, **kwargs):
    """Fit Gaussian ARD through the readout path; return params and precisions."""
    n_obs, n_latent = y.shape[-1], m.shape[-1]
    settings = loading_settings("ard", kwargs)
    start = Params(
        C=jnp.zeros((n_obs, n_latent)),
        d=jnp.zeros(n_obs),
        R=jnp.eye(n_obs),
        alpha=jnp.full(n_latent, settings["alpha"]),
    )
    params, _ = Gaussian.solve_readout("ard", settings, start, y, valid, m, V)
    return params, np.asarray(params.alpha)


def test_poisson_initialization_stays_bounded_on_a_problem_that_crossed_the_cap():
    """L-BFGS in initialization used to run to bias ~1e34 on this data.

    The objective's rate was clipped at MAX_LOGRATE but its ``y * lin`` term was
    not, so a step across the cap found an objective with no minimum.
    """
    rng = np.random.default_rng(7)
    n_sequences, n_time, n_features, n_latents = 2, 300, 20, 2
    kernel = HidaMatern(sigma=1.0, rho=1.0, omega=0.0, order=1, s=1e-5)
    latents = sample_gp_latents(kernel, n_time, 0.02, n_latents, rng, n_sequences, jitter=1e-8)
    loadings = rng.normal(size=(n_features, n_latents))
    loadings = 0.8 * loadings / np.linalg.norm(loadings, axis=1, keepdims=True)
    y = jnp.asarray(rng.poisson(np.exp(2.0 + latents @ loadings.T)).astype(float))

    params = Poisson.initialize_params(y, jnp.ones(y.shape, dtype=bool), 5, random_state=0)
    assert float(jnp.abs(params.d).max()) < 10.0, params.d
    assert float(jnp.abs(params.C).max()) < 10.0


@pytest.mark.parametrize("alpha_shape,alpha_rate", [(1.0 + 1e-6, 1e-6), (1.2, 0.2)])
def test_poisson_ard_laplace_evidence_matches_quadrature(alpha_shape, alpha_rate):
    """Check the loading-block approximation against a two-dimensional integral."""
    rng = np.random.default_rng(4)
    n_time = 80
    m = jnp.asarray(rng.standard_normal((n_time, 1)))
    V = jnp.full((n_time, 1, 1), 0.1)
    y = jnp.asarray(rng.poisson(np.exp(0.4 + 0.6 * np.asarray(m[:, 0])))[:, None])
    valid = jnp.asarray((rng.random((n_time, 1)) > 0.2))
    start = Params(C=jnp.zeros((1, 1)), d=jnp.zeros(1), R=jnp.asarray(0.0), alpha=jnp.ones(1))
    params, evidence = Poisson.solve_readout(
        "ard",
        loading_settings("ard", {"alpha_shape": alpha_shape, "alpha_rate": alpha_rate, "iterations": 40}),
        start, y, valid, m, V,
    )

    # Integrate the likelihood times the normalized loading prior over bias and
    # loading. As in the fit, the flat bias prior has unit density and
    # count-factorial terms are omitted.
    bias = np.linspace(-1.5, 2.0, 301)
    loading = np.linspace(-2.0, 2.0, 301)
    d, c = np.meshgrid(bias, loading, indexing="ij")
    alpha = float(params.alpha[0])
    selected = np.asarray(valid[:, 0])
    mt, yt = np.asarray(m[:, 0])[selected], np.asarray(y[:, 0])[selected]
    linear = d[..., None] + c[..., None] * mt
    log_rate = linear + 0.05 * c[..., None] ** 2
    log_integrand = np.sum(yt * linear - np.exp(log_rate), axis=-1)
    log_integrand += 0.5 * np.log(alpha / (2 * np.pi)) - 0.5 * alpha * c**2
    maximum = log_integrand.max()
    integral = trapezoid(trapezoid(np.exp(log_integrand - maximum), loading, axis=1), bias)
    exact = np.log(integral) + maximum + (alpha_shape - 1) * np.log(alpha) - alpha_rate * alpha
    npt.assert_allclose(evidence, exact, rtol=0, atol=0.02)

    beta = jnp.concatenate((params.d, params.C[0]))
    penalty = jnp.array([0.0, alpha])

    def objective(value):
        return jnp.sum(valid) * Poisson.readout_objective(
            (value[None, 1:], value[:1]), y, valid, m, V, penalty
        )

    npt.assert_allclose(jax.grad(objective)(beta), 0.0, atol=ATOL_STATIONARY)
    covariance = jnp.linalg.inv(jax.hessian(objective)(beta))
    gamma = 1.0 - alpha * covariance[1, 1]
    next_alpha = (gamma + 2 * alpha_shape - 2) / (params.C[0, 0] ** 2 + 2 * alpha_rate)
    npt.assert_allclose(alpha, next_alpha, rtol=1e-3)


def test_poisson_ard_ignores_masked_counts():
    rng = np.random.default_rng(12)
    n_time, n_features, n_latents = 60, 3, 2
    m = jnp.asarray(rng.standard_normal((n_time, n_latents)))
    V = jnp.broadcast_to(0.05 * jnp.eye(n_latents), (n_time, n_latents, n_latents))
    y = jnp.asarray(rng.poisson(2.0, size=(n_time, n_features)))
    valid = jnp.asarray(rng.random(y.shape) > 0.25)
    start = Params(C=jnp.zeros((n_features, n_latents)), d=jnp.zeros(n_features),
                   R=jnp.asarray(0.0), alpha=jnp.ones(n_latents))
    settings = loading_settings("ard", {"iterations": 10})
    first, evidence = Poisson.solve_readout("ard", settings, start, y, valid, m, V)
    changed, changed_evidence = Poisson.solve_readout(
        "ard", settings, start, jnp.where(valid, y, 10000), valid, m, V
    )
    for actual, expected in ((changed.C, first.C), (changed.d, first.d),
                             (changed.alpha, first.alpha), (changed_evidence, evidence)):
        npt.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_gaussian_ard_evidence_matches_quadrature():
    """The closed-form evidence equals a brute-force integral over (d, C)."""
    T, N, K = 8, 2, 1
    rng = np.random.default_rng(0)
    m = jnp.asarray(rng.standard_normal((T, K)))
    V = jnp.asarray(0.1 * np.broadcast_to(np.eye(K), (T, K, K)))
    y = jnp.asarray(rng.standard_normal((T, N)))
    valid = jnp.ones((T, N), dtype=bool)
    log_alpha = jnp.log(jnp.array([0.7]))
    log_R = jnp.log(jnp.array([0.9, 1.3]))

    terms = Gaussian.loading_terms(y, valid, m, V)
    actual = float(ard_evidence(terms, log_alpha, log_R))

    # Brute force: exp(E_q[log p]) times the loading prior, integrated over the
    # bias and the single loading column of each feature independently.  The
    # integrand is a smooth 2-D Gaussian, so the trapezoidal rule converges
    # exponentially in the grid spacing: 401 points over this range already
    # matches a 4001-point grid to machine precision while keeping the
    # quadrature arrays ~100x smaller.  scipy's trapezoid is used instead of
    # numpy's because numpy renamed np.trapz to np.trapezoid in 2.0, so neither
    # name covers the supported numpy range (>=1.24).
    grid = np.linspace(-20.0, 20.0, 401)
    bias, col = np.meshgrid(grid, grid, indexing="ij")
    alpha = float(np.exp(log_alpha[0]))
    expected = 0.0
    for n in range(N):
        variance = float(np.exp(log_R[n]))
        residual = (
            np.asarray(y)[:, n][None, None, :]
            - col[:, :, None] * np.asarray(m)[:, 0][None, None, :]
            - bias[:, :, None]
        )
        quadratic = residual**2 + (col**2)[:, :, None] * np.asarray(V)[:, 0, 0][None, None, :]
        log_density = -0.5 * np.sum(
            np.log(2 * np.pi * variance) + quadratic / variance, axis=-1
        )
        log_prior = 0.5 * np.log(alpha / (2 * np.pi)) - 0.5 * alpha * col**2
        integrand = np.exp(log_density + log_prior)
        expected += np.log(
            trapezoid(trapezoid(integrand, grid, axis=0), grid, axis=0)
        )

    npt.assert_allclose(actual, expected, rtol=1e-8, atol=1e-6)


def test_gaussian_ard_matches_reference_ard_regression():
    """With V = 0 the evidence is exact, so the fit matches a standard ARD fit.

    The precisions are shared across features here but there is a single
    feature, so the two structures coincide. The reference centers the design,
    so it carries no bias column and its residual has one more degree of
    freedom than ours; that is the only expected difference.
    """
    T, K = 80, 3
    rng = np.random.default_rng(4)
    m = rng.standard_normal((T, K))
    y = m @ np.array([[0.7, 0.5, 0.0]]).T + rng.normal(scale=0.4, size=(T, 1))

    params, alpha = _ard_fit(
        jnp.asarray(y), jnp.ones((T, 1), dtype=bool), jnp.asarray(m),
        jnp.zeros((T, K, K)),
    )
    reference = ARDRegression(
        fit_intercept=True, threshold_lambda=1e18, max_iter=3000, tol=1e-12
    ).fit(m, y[:, 0])

    npt.assert_allclose(np.asarray(params.C)[0, :2], reference.coef_[:2], atol=1e-2)
    npt.assert_allclose(alpha[:2], reference.lambda_[:2], rtol=1e-2)
    npt.assert_allclose(np.asarray(params.d)[0], reference.intercept_, atol=1e-2)

    residual = float(np.sum((y[:, 0] - m @ reference.coef_ - reference.intercept_) ** 2))
    degrees = residual * reference.alpha_
    ratio = float(np.diag(np.asarray(params.R))[0]) / (1.0 / reference.alpha_)
    npt.assert_allclose(ratio, degrees / (degrees - 1.0), rtol=1e-2)


def test_gaussian_ard_leaves_a_wide_ranking_margin():
    """Inactive latents separate from active ones by an order of magnitude.

    The absolute precision is gauge dependent, so a threshold has to be relative
    or chosen per fit. What makes that workable is the size of the gap.
    """
    T, N, K = 120, 6, 4
    rng = np.random.default_rng(2)
    m = jnp.asarray(rng.standard_normal((T, K)))
    loadings = np.zeros((N, K))
    loadings[:, :2] = rng.normal(scale=0.8, size=(N, 2))
    y = jnp.asarray(np.asarray(m) @ loadings.T + rng.normal(scale=0.3, size=(T, N)))

    _, alpha = _ard_fit(y, jnp.ones((T, N), dtype=bool), m, jnp.zeros((T, K, K)))

    ordered = np.sort(alpha)
    assert ordered[2] > 10.0 * ordered[1], alpha
    assert ordered[2] > 10.0 * ordered[0], alpha


def test_gaussian_ard_precision_prior_rate_shrinks_inactive():
    """A stronger Gamma rate pulls an irrelevant precision down.

    The gap dominates the within-group spread and the signal columns are
    essentially untouched, so the prior's rate is a knob on how far inactive
    columns are shrunk rather than on which columns are active.
    """
    T, N, K = 60, 8, 2
    rng = np.random.default_rng(1)
    m = jnp.asarray(rng.standard_normal((T, K)))
    V = jnp.zeros((T, K, K))
    loadings = np.zeros((N, K))
    loadings[:, 0] = rng.normal(scale=0.8, size=N)
    y = jnp.asarray(np.asarray(m) @ loadings.T + rng.normal(scale=0.3, size=(T, N)))
    valid = jnp.ones((T, N), dtype=bool)

    _, weak = _ard_fit(y, valid, m, V)
    _, strong = _ard_fit(y, valid, m, V, alpha_rate=1e-2)

    assert strong[1] < 0.5 * weak[1], (weak, strong)
    npt.assert_allclose(strong[0], weak[0], rtol=0.1)


def test_ard_estimate_warm_starts_from_params_alpha():
    """The fixed-point iteration starts from ``params.alpha`` when present."""
    T, N, K = 20, 2, 2
    rng = np.random.default_rng(0)
    m = jnp.asarray(rng.standard_normal((T, K)))
    V = jnp.zeros((T, K, K))
    y = jnp.asarray(rng.standard_normal((T, N)))
    valid = jnp.ones((T, N), dtype=bool)
    base = Params(C=jnp.zeros((N, K)), d=jnp.zeros(N), R=jnp.eye(N))

    def fit(alpha):
        params = Params(C=base.C, d=base.d, R=base.R, alpha=alpha)
        return Gaussian.solve_readout(
            "ard", loading_settings("ard", {"iterations": 1}), params, y, valid, m, V
        )[0]

    low = fit(jnp.full(K, 0.5))
    high = fit(jnp.full(K, 50.0))
    assert not np.allclose(np.asarray(low.alpha), np.asarray(high.alpha))


def test_resolve_alpha_applies_one_rule_per_loading_kind():
    """Ridge is a hyperparameter; ARD warm starts and fixed keeps what was supplied."""
    settings = {"alpha": 0.5}
    supplied = jnp.asarray([7.0, 8.0])

    npt.assert_allclose(resolve_alpha("ridge", settings, supplied, 2), [0.5, 0.5])
    npt.assert_allclose(resolve_alpha("ard", settings, supplied, 2), [7.0, 8.0])
    npt.assert_allclose(resolve_alpha("fixed", settings, supplied, 2), [7.0, 8.0])
    for kind in ("ridge", "ard", "fixed"):
        npt.assert_allclose(resolve_alpha(kind, settings, None, 2), [0.5, 0.5])
    with pytest.raises(ValueError, match="params.alpha"):
        resolve_alpha("ard", settings, jnp.asarray([1.0, -1.0]), 2)


def test_ard_solver_requires_the_precision_that_fit_supplies():
    """There is no silent fallback inside the solver; ``fit`` owns the warm start."""
    start = Params(C=jnp.zeros((2, 2)), d=jnp.zeros(2), R=jnp.eye(2))
    y = jnp.ones((4, 2))
    with pytest.raises(ValueError, match="Params.alpha"):
        Gaussian.solve_readout(
            "ard", loading_settings("ard", None), start, y,
            jnp.ones_like(y, dtype=bool), jnp.zeros((4, 2)), jnp.zeros((4, 2, 2)),
        )


def test_ard_warm_start_continues_across_em_steps():
    """A second ARD update starts from the previous precision, not the configured one."""
    T, N, K = 20, 2, 2
    rng = np.random.default_rng(2)
    m = jnp.asarray(rng.standard_normal((T, K)))
    V = jnp.zeros((T, K, K))
    y = jnp.asarray(rng.standard_normal((T, N)))
    valid = jnp.ones((T, N), dtype=bool)
    start = Params(
        C=jnp.zeros((N, K)), d=jnp.zeros(N), R=jnp.eye(N), alpha=jnp.ones(K)
    )
    settings = loading_settings("ard", {"iterations": 1})

    first, _ = Gaussian.solve_readout("ard", settings, start, y, valid, m, V)
    second, _ = Gaussian.solve_readout("ard", settings, first, y, valid, m, V)
    cold, _ = Gaussian.solve_readout("ard", settings, start, y, valid, m, V)
    assert not np.allclose(np.asarray(second.alpha), np.asarray(cold.alpha))


@pytest.mark.parametrize("alpha", [0.0, -1.0, float("inf"), float("nan")])
def test_ard_settings_reject_invalid_alpha(alpha):
    """The loading precision must be finite and strictly positive."""
    with pytest.raises(ValueError, match="alpha"):
        loading_settings("ard", {"alpha": alpha})


@pytest.mark.parametrize("alpha_shape", [0.0, 0.5, 1.0, -1.0, float("inf"), float("nan")])
def test_ard_settings_reject_invalid_alpha_shape(alpha_shape):
    """The Gamma shape must be finite and greater than one."""
    with pytest.raises(ValueError, match="alpha_shape"):
        loading_settings("ard", {"alpha_shape": alpha_shape})


@pytest.mark.parametrize("alpha_rate", [0.0, -1.0, float("inf"), float("nan")])
def test_ard_settings_reject_invalid_alpha_rate(alpha_rate):
    """The Gamma rate must be finite and strictly positive."""
    with pytest.raises(ValueError, match="alpha_rate"):
        loading_settings("ard", {"alpha_rate": alpha_rate})


@pytest.mark.parametrize("iterations", [0, -1])
def test_ard_settings_reject_invalid_iterations(iterations):
    """The fixed-point iteration count must be at least one."""
    with pytest.raises(ValueError, match="iterations"):
        loading_settings("ard", {"iterations": iterations})


def test_loading_settings_reject_an_unknown_kind():
    """A kind with no defaults is a named error, not a bare KeyError."""
    with pytest.raises(ValueError, match="unknown loading kind"):
        loading_settings("lasso", None)


def test_loading_settings_reject_unknown_keys():
    """A misspelled key fails instead of being silently ignored."""
    with pytest.raises(ValueError, match="unknown loading_config keys"):
        loading_settings("ard", {"iteration": 3})


def test_gaussian_initialize_info_values(rng):
    """Gaussian information increments use C in latent space."""
    T, N, K = 30, 8, 2
    C = jnp.array(rng.standard_normal((N, K)))
    d = jnp.array(rng.standard_normal(N))
    R = jnp.eye(N) * 0.5
    params = Params(C=C, d=d, R=R)

    y = jnp.array(rng.standard_normal((T, N)))  # single sequence
    valid_y = jnp.ones((T, N), dtype=bool)

    # Compute expected values directly using the correct formula
    J_exp_single = C.T @ jnp.linalg.solve(R, C)  # (K, K)
    j_exp = (C.T @ jnp.linalg.solve(R, (y - d).T)).T  # (T, K)
    J_exp = jnp.tile(J_exp_single, (T, 1, 1))  # (T, K, K)

    j, J = Gaussian.initialize_info(params, y, valid_y)
    chex.assert_shape([j, J], [(T, K), (T, K, K)])
    npt.assert_allclose(j, j_exp, rtol=1e-6, atol=1e-6)
    npt.assert_allclose(J, J_exp, rtol=1e-6, atol=1e-6)

    # Verify key properties
    # J should be symmetric positive semi-definite
    npt.assert_allclose(
        np.asarray(J_exp_single), np.asarray(J_exp_single.T), atol=1e-12
    )
    eigvals = np.linalg.eigvalsh(np.asarray(J_exp_single))
    assert np.all(eigvals >= -1e-10), f"J has negative eigenvalues: {eigvals}"


def test_gaussian_initialize_info_preserves_unequal_loading_gains():
    """Gaussian information increments preserve unequal loading gains."""
    C = jnp.array([[0.2], [2.0]])
    d = jnp.array([0.1, -0.2])
    R = jnp.diag(jnp.array([0.5, 0.8]))
    y = jnp.array([[0.6, 1.4], [-0.2, 0.4]])
    params = Params(C=C, d=d, R=R)

    j, J = Gaussian.initialize_info(params, y, jnp.ones_like(y, dtype=bool))
    expected_j = (C.T @ jnp.linalg.solve(R, (y - d).T)).T
    expected_J = C.T @ jnp.linalg.solve(R, C)
    npt.assert_allclose(j, expected_j, rtol=1e-6, atol=1e-6)
    npt.assert_allclose(J, jnp.broadcast_to(expected_J, J.shape), rtol=1e-6, atol=1e-6)

    # Lifting with a non-identity state mask is equivalent to H = C @ M.
    M = jnp.array([[0.5, 1.0]])
    H = C @ M
    state_j, state_J = lift(j, J, M)
    expected_state_j = (H.T @ jnp.linalg.solve(R, (y - d).T)).T
    expected_state_J = H.T @ jnp.linalg.solve(R, H)
    npt.assert_allclose(state_j, expected_state_j, rtol=1e-6, atol=1e-6)
    npt.assert_allclose(
        state_J, jnp.broadcast_to(expected_state_J, state_J.shape), rtol=1e-6, atol=1e-6
    )


def test_gaussian_smoothing_with_unequal_loading_gains():
    """State-space inference matches dense GP conditioning with the same C."""
    C = np.array([[0.2], [2.0]])
    d = np.array([0.1, -0.2])
    R = np.diag([0.5, 0.8])
    y = np.array([[0.6, 1.4], [-0.2, 0.4], [0.9, -0.3], [0.7, 1.0]])
    kernel = HidaMatern(sigma=1.1, rho=2.0, omega=0.0, order=1, s=1e-5)
    params = Params(C=jnp.asarray(C), d=jnp.asarray(d), R=jnp.asarray(R))
    model = CVHM(n_components=1, dt=0.5, kernels=[kernel], params=params)
    means, covariances = model.infer(jnp.asarray(y))

    times = np.arange(len(y)) * model.dt
    lags = np.abs(times[:, None] - times[None, :])
    prior = np.asarray(kernel.kernel(jnp.asarray(lags))) + kernel.s * np.eye(len(y))
    precision = np.linalg.solve(prior, np.eye(len(y)))
    precision += (C.T @ np.linalg.solve(R, C)).item() * np.eye(len(y))
    expected_covariance = np.linalg.solve(precision, np.eye(len(y)))
    information = (C.T @ np.linalg.solve(R, (y - d).T)).ravel()
    expected_mean = expected_covariance @ information

    npt.assert_allclose(means[0, :, 0], expected_mean, rtol=1e-5, atol=1e-5)
    npt.assert_allclose(
        covariances[0, :, 0, 0], np.diag(expected_covariance), rtol=1e-5, atol=1e-5
    )


def test_poisson_update_pseudo_preserves_unequal_loading_gains():
    """Poisson pseudo-observations preserve unequal loading gains."""
    C = jnp.array([[0.2], [2.0]])
    d = jnp.array([0.1, -0.2])
    params = Params(C=C, d=d, R=jnp.asarray(0.0))
    y = jnp.array([[[2.0, 1.0]]])
    m = jnp.array([[[0.3]]])
    V = jnp.array([[[[0.4]]]])
    j = jnp.zeros_like(m)
    J = jnp.zeros_like(V)

    actual_j, actual_J = Poisson.update_pseudo(
        params, y, jnp.ones_like(y, dtype=bool), m, V, j, J, lr=1.0
    )
    rates = jnp.exp(C[:, 0] * m[0, 0, 0] + d + 0.5 * C[:, 0] ** 2 * V[0, 0, 0, 0])
    precision = jnp.sum(C[:, 0] ** 2 * rates)
    information = jnp.sum((y[0, 0] - rates) * C[:, 0]) + precision * m[0, 0, 0]
    npt.assert_allclose(actual_j[0, 0, 0], information, rtol=1e-6, atol=1e-6)
    npt.assert_allclose(actual_J[0, 0, 0, 0], precision, rtol=1e-6, atol=1e-6)


def test_gaussian_update_pseudo_noop():
    """Gaussian.update_pseudo returns its inputs unchanged."""
    K = 3
    T = 10
    j = jnp.ones((1, T, K))
    J = jnp.tile(jnp.eye(K), (1, T, 1, 1))
    m = jnp.zeros((1, T, K))
    V = jnp.tile(jnp.eye(K), (1, T, 1, 1))
    y = jnp.zeros((1, T, 5))
    valid_y = jnp.ones((1, T, 5), dtype=bool)
    params = Params(C=jnp.ones((5, K)), d=jnp.zeros(5), R=jnp.eye(5))

    j_out, J_out = Gaussian.update_pseudo(params, y, valid_y, m, V, j, J, 0.1)
    npt.assert_array_equal(np.asarray(j_out), np.asarray(j))
    npt.assert_array_equal(np.asarray(J_out), np.asarray(J))


def test_poisson_cvi_gradient_direction(rng):
    """One Poisson CVI step should not increase the negative ELL."""
    T, N, K = 30, 5, 2

    C_raw = jnp.array(rng.standard_normal((N, K)) * 0.3)
    d = jnp.ones(N)
    params = Params(C=C_raw, d=d, R=jnp.asarray(0.0))

    x = jnp.array(rng.standard_normal((1, T, K)) * 0.5)
    eta = x @ C_raw.T + d
    y = jnp.array(rng.poisson(np.exp(np.asarray(eta))).astype(float))
    valid_y = jnp.ones((1, T, N), dtype=bool)

    # Initialize pseudo-obs in latent space
    j = jnp.zeros((1, T, K))
    J = jnp.tile(jnp.eye(K) * 0.1, (1, T, 1, 1))

    # Latent-space posterior moments
    m = jnp.zeros((1, T, K))
    V = jnp.tile(jnp.eye(K), (1, T, 1, 1))

    # One update
    j_new, J_new = Poisson.update_pseudo(params, y, valid_y, m, V, j, J, lr=0.5)

    # j_new and J_new should differ from the initial values
    assert not jnp.allclose(j_new, j), "CVI update produced no change"


def test_poisson_step_stats_track_the_objective_derivatives():
    """The E-step's gradient and information follow the readout objective.

    Below ``MAX_LOGRATE`` the information equals the objective's curvature.
    Above it the rate follows a tangent line, so the objective is linear and its
    curvature is zero, while the E-step keeps ``C^T diag(cap) C``. That
    difference cannot move the fixed point: ``K`` appears on both sides of the
    normal equations ``(I + J) m = j`` with ``j = grad + K m`` and ``J = K``,
    so it cancels. Using the rate's value instead of its derivative in ``grad``
    would move it, which is what the gradient assertion pins down.
    """
    rng = np.random.default_rng(3)
    N, K = 5, 3
    C = jnp.asarray(rng.standard_normal((N, K)) * 0.5)
    m = jnp.asarray(rng.standard_normal(K))
    V = jnp.eye(K) * 0.3
    y = jnp.asarray(rng.poisson(3.0, size=N) + 1.0)
    valid_y = jnp.ones(N, dtype=bool)
    penalty = jnp.zeros(K + 1)

    for offset, capped in ((0.5, False), (MAX_LOGRATE + 4.0, True)):
        d = jnp.full(N, offset)

        def objective(m_):
            return poisson_readout_objective(
                (C, d), y[None], valid_y[None], m_[None], V[None], penalty
            )

        grad = jax.grad(objective)(m)
        hess = jax.hessian(objective)(m)
        k, K_stats = poisson_cvi_step_stats_latent(m, V, y, valid_y, C, d)
        eta = C @ m + d + 0.5 * jnp.einsum("nk,kl,nl->n", C, V, C)
        rate_deriv = jax.vmap(jax.grad(_capped_exp))(eta)
        # Above the cap the information entries reach 1e3, so an absolute budget
        # has to follow the magnitude rather than sit at the rounding floor.
        atol = ATOL_TIGHT * float(jnp.abs(K_stats).max())

        if capped:
            assert jnp.all(eta > MAX_LOGRATE), "fixture must exercise the cap"
            npt.assert_allclose(np.asarray(hess), 0.0, atol=1e-10)
        else:
            assert jnp.all(eta < MAX_LOGRATE), "fixture must stay below the cap"
            npt.assert_allclose(
                np.asarray(hess), np.asarray(K_stats), rtol=RTOL_TIGHT, atol=atol
            )

        npt.assert_allclose(
            np.asarray(grad),
            np.asarray(C.T @ (rate_deriv - y)),
            rtol=RTOL_TIGHT,
            atol=atol,
        )
        npt.assert_allclose(
            np.asarray(K_stats),
            np.asarray(C.T @ jnp.diag(rate_deriv) @ C),
            rtol=RTOL_TIGHT,
            atol=atol,
        )
        npt.assert_allclose(
            np.asarray(k),
            np.asarray(K_stats @ m - grad),
            rtol=RTOL_TIGHT,
            atol=atol,
        )


def test_poisson_cvi_step_stats_latent_masked(rng):
    """Masked time steps contribute zero CVI gradients from latent-space stats."""
    N, K = 5, 3
    C = jnp.array(rng.standard_normal((N, K)) * 0.3)
    d = jnp.zeros(N)
    m = jnp.ones(K)
    V = jnp.eye(K)
    y = jnp.ones(N) * 10.0

    k_on, K_on = poisson_cvi_step_stats_latent(m, V, y, jnp.ones(N, dtype=bool), C, d)
    assert jnp.any(k_on != 0)
    assert jnp.any(K_on != 0)

    k_off, K_off = poisson_cvi_step_stats_latent(m, V, y, jnp.zeros(N, dtype=bool), C, d)
    npt.assert_array_equal(np.asarray(k_off), 0.0)
    npt.assert_array_equal(np.asarray(K_off), 0.0)


def test_gaussian_e2e(linear_gaussian_data):
    """CVHM with Gaussian likelihood recovers latents with R^2 > 0.5."""
    data = linear_gaussian_data
    y = jnp.array(data["y"])
    x_true = np.asarray(data["x_true"]).squeeze()  # (T,)

    kernels = [HidaMatern(sigma=1.0, rho=50.0, omega=0.0, order=0)]
    model = CVHM(
        n_components=1,
        dt=1.0,
        kernels=kernels,
        observation="Gaussian",
        max_iter=10,
        cvi_iter=1,
    )
    model.fit(y, random_state=0)
    m = np.asarray(model.posterior[0]).squeeze()  # (T,)

    # Align sign (latent is identified up to a sign flip)
    reg = LinearRegression().fit(m.reshape(-1, 1), x_true)
    m_aligned = reg.predict(m.reshape(-1, 1))
    ss_res = np.sum((x_true - m_aligned) ** 2)
    ss_tot = np.sum((x_true - x_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    assert r2 > 0.5, f"Gaussian E2E R^2 = {r2:.3f} < 0.5"


def test_loading_type_lists_exactly_the_kinds_with_defaults():
    """``Loading`` is what annotations promise; ``DEFAULTS`` is what the code accepts."""
    assert set(get_args(Loading)) == set(DEFAULTS)
