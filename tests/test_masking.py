"""Mathematical and integration tests for feature-level observation masks."""

import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from cvhmax.readout import Params, loading_settings
from cvhmax.gaussian import Gaussian
from cvhmax.poisson import (
    MAX_LOGRATE,
    Poisson,
    _capped_exp,
    poisson_cvi_step_stats_latent,
    poisson_readout_objective,
)
from cvhmax.cvhm import CVHM, lift, project
from cvhmax.hm import HidaMatern
from cvhmax.utils import pad_sequences

from .helpers import sample_gp_latents


def test_masked_gaussian_information_matches_observed_rows():
    C = jnp.array([[0.2, 0.8], [1.3, -0.4], [0.7, 0.3]])
    d = jnp.array([0.1, -0.2, 0.4])
    R = jnp.diag(jnp.array([0.5, 0.8, 1.2]))
    y = jnp.array([[1.2, 20.0, -0.5], [10.0, 0.4, 0.2]])
    mask = jnp.array([[True, False, True], [False, True, True]])
    j, J = Gaussian.initialize_info(Params(C=C, d=d, R=R), y, mask)
    M = jnp.array([[0.5, 0.0, 0.0, 0.0], [0.0, 0.0, 1.2, 0.0]])
    state_j, state_J = lift(j, J, M)

    for t, indices in enumerate(([0, 2], [1, 2])):
        H = np.asarray(C)[indices] @ np.asarray(M)
        variance = np.diag(np.asarray(R))[indices]
        residual = np.asarray(y - d)[t, indices]
        expected_j = H.T @ (residual / variance)
        expected_J = H.T @ (H / variance[:, None])
        npt.assert_allclose(state_j[t], expected_j, atol=1e-10, rtol=1e-10)
        npt.assert_allclose(state_J[t], expected_J, atol=1e-10, rtol=1e-10)


def test_masked_gaussian_smoothing_matches_dense_gp():
    C = np.array([[0.2], [2.0], [-0.7]])
    d = np.array([0.1, -0.2, 0.3])
    variance = np.array([0.5, 0.8, 1.2])
    y = np.array([[0.6, 1.4, 0.2], [-0.2, 0.4, -0.1], [4.0, 8.0, 3.0], [0.7, 1.0, 1.2]])
    mask = np.array([[True, False, True], [False, True, True],
                     [False, False, False], [True, True, False]])
    kernel = HidaMatern(sigma=1.1, rho=2.0, order=1)
    model = CVHM(
        n_components=1, dt=0.5, kernels=[kernel],
        params=Params(C=jnp.asarray(C), d=jnp.asarray(d), R=jnp.diag(jnp.asarray(variance))),
    )
    mean, covariance = model.infer(jnp.asarray(y), jnp.asarray(mask))
    times = np.arange(len(y)) * model.dt
    prior = np.asarray(kernel.kernel(jnp.asarray(np.abs(times[:, None] - times))))
    prior = prior + kernel.s * np.eye(len(y))
    info = np.sum(mask * C[:, 0] * (y - d) / variance, axis=1)
    precision = np.linalg.solve(prior, np.eye(len(y)))
    precision += np.diag(np.sum(mask * C[:, 0] ** 2 / variance, axis=1))
    expected_covariance = np.linalg.solve(precision, np.eye(len(y)))
    expected_mean = expected_covariance @ info
    npt.assert_allclose(mean[0, :, 0], expected_mean, rtol=1e-5, atol=1e-5)
    npt.assert_allclose(
        covariance[0, :, 0, 0], np.diag(expected_covariance), rtol=1e-5, atol=1e-5
    )


def test_masked_poisson_information_and_objective():
    C = jnp.array([[0.2, 0.5], [0.7, -0.3], [-0.4, 0.6]])
    d = jnp.array([0.1, 0.3, -0.2])
    y = jnp.array([[2.0, 4.0, 1.0], [3.0, 1.0, 2.0]])
    mask = jnp.array([[True, False, True], [False, True, True]])
    means = jnp.array([[0.2, -0.1], [-0.3, 0.4]])
    covariances = jnp.broadcast_to(jnp.array([[0.4, 0.1], [0.1, 0.5]]), (2, 2, 2))
    gamma = 0.3
    penalty = jnp.concatenate((jnp.zeros(1), jnp.full(2, gamma)))

    def reference(params):
        C, d = params
        total = 0.0
        for t, indices in enumerate(([0, 2], [1, 2])):
            Ct, dt, yt = C[jnp.array(indices)], d[jnp.array(indices)], y[t, jnp.array(indices)]
            linear = Ct @ means[t] + dt
            quad = jnp.einsum("ni,ij,nj->n", Ct, covariances[t], Ct)
            log_rate = jnp.minimum(linear + 0.5 * quad, MAX_LOGRATE)
            total += jnp.sum(jnp.exp(log_rate) - yt * linear)
        return (total + 0.5 * gamma * jnp.sum(C**2)) / 2

    def objective(params):
        return poisson_readout_objective(params, y, mask, means, covariances, penalty)

    npt.assert_allclose(objective((C, d)), reference((C, d)), rtol=1e-10, atol=1e-10)
    actual_grad = jax.grad(objective)((C, d))
    expected_grad = jax.grad(reference)((C, d))
    for actual, expected in zip(actual_grad, expected_grad):
        npt.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)

    j, J = poisson_cvi_step_stats_latent(means[0], covariances[0], y[0], mask[0], C, d)
    observed_C = np.asarray(C)[[0, 2]]
    observed_d = np.asarray(d)[[0, 2]]
    observed_y = np.asarray(y[0])[[0, 2]]
    m, V = np.asarray(means[0]), np.asarray(covariances[0])
    rates = np.exp(observed_C @ m + observed_d + 0.5 * np.einsum("ni,ij,nj->n", observed_C, V, observed_C))
    expected_J = observed_C.T @ (rates[:, None] * observed_C)
    expected_j = (observed_y - rates) @ observed_C + expected_J @ m
    npt.assert_allclose(j, expected_j, rtol=1e-10, atol=1e-10)
    npt.assert_allclose(J, expected_J, rtol=1e-10, atol=1e-10)


def test_poisson_objective_penalises_each_latent_column_by_its_own_precision():
    """Unequal precisions weight columns, not features.

    Square ``C`` makes a transposed broadcast silently wrong rather than a shape
    error, so equal-precision tests cannot see it.
    """
    rng = np.random.default_rng(3)
    n_features = n_latents = 3
    C = jnp.asarray(rng.normal(size=(n_features, n_latents)))
    d = jnp.zeros(n_features)
    y = jnp.asarray(rng.poisson(2.0, size=(5, n_features)).astype(float))
    valid = jnp.ones_like(y, dtype=bool)
    m = jnp.asarray(rng.normal(size=(5, n_latents)))
    V = jnp.zeros((5, n_latents, n_latents))
    precisions = jnp.asarray([0.2, 1.1, 3.0])

    penalised = poisson_readout_objective(
        (C, d), y, valid, m, V, jnp.concatenate((jnp.zeros(1), precisions))
    )
    plain = poisson_readout_objective((C, d), y, valid, m, V, jnp.zeros(n_latents + 1))

    expected = 0.5 * jnp.sum(precisions * jnp.sum(C**2, axis=0)) / len(y)
    npt.assert_allclose(penalised - plain, expected, rtol=1e-10)


def test_capped_exp_is_exp_below_the_cap_and_its_tangent_above():
    """Exact value and slope below the cap, continuous at it, finite where exp overflows."""
    below = jnp.linspace(-5.0, MAX_LOGRATE, 40)
    npt.assert_allclose(_capped_exp(below), jnp.exp(below), rtol=1e-12)
    npt.assert_allclose(jax.vmap(jax.grad(_capped_exp))(below), jnp.exp(below), rtol=1e-12)

    cap = jnp.exp(MAX_LOGRATE)
    above = MAX_LOGRATE + jnp.asarray([1e-9, 1.0, 10.0, 1000.0])  # exp(1007) overflows float64
    npt.assert_allclose(_capped_exp(above), cap * (1.0 + above - MAX_LOGRATE), rtol=1e-12)
    npt.assert_allclose(jax.vmap(jax.grad(_capped_exp))(above), cap, rtol=1e-12)
    assert bool(jnp.all(jnp.isfinite(jax.vmap(jax.grad(_capped_exp))(above))))
    npt.assert_allclose(_capped_exp(MAX_LOGRATE + 1e-12), cap, rtol=1e-9)


def test_poisson_objective_rises_beyond_the_cap_through_the_variance_term():
    """The posterior-variance term can cross the cap too; the objective must still rise."""
    y = jnp.asarray([[3.0]])
    valid = jnp.ones_like(y, dtype=bool)
    m, V = jnp.ones((1, 1)), jnp.ones((1, 1, 1))
    values = jnp.stack([
        poisson_readout_objective(
            (jnp.asarray([[s]]), jnp.zeros(1)), y, valid, m, V, jnp.zeros(2)
        )
        for s in (5.0, 10.0, 30.0, 100.0)  # lin + q/2 = s + s**2/2, past the cap from s = 3
    ])
    assert bool(jnp.all(jnp.diff(values) > 0)), values


def test_poisson_objective_rises_beyond_the_rate_cap():
    """Past MAX_LOGRATE the objective increases with the log rate.

    A flat clip on the rate alone leaves ``-y * lin`` free to fall without limit,
    so an optimiser that overshoots the cap runs away to infinity.
    """
    y = jnp.asarray([[3.0]])
    valid = jnp.ones_like(y, dtype=bool)
    m, V, C = jnp.zeros((1, 1)), jnp.zeros((1, 1, 1)), jnp.zeros((1, 1))
    penalty = jnp.zeros(2)
    log_rates = MAX_LOGRATE + jnp.asarray([0.0, 1.0, 10.0, 100.0])
    values = jnp.stack([
        poisson_readout_objective((C, jnp.asarray([x])), y, valid, m, V, penalty)
        for x in log_rates
    ])
    assert bool(jnp.all(jnp.diff(values) > 0)), values


def test_all_observed_poisson_objective_preserves_step_scaling():
    C = jnp.array([[0.2], [0.7]])
    d = jnp.array([0.1, -0.2])
    y = jnp.array([[1.0, 3.0], [2.0, 1.0]])
    m = jnp.array([[0.2], [-0.1]])
    V = jnp.array([[[0.3]], [[0.4]]])
    linear = m @ C.T + d
    log_rate = jnp.minimum(linear + 0.5 * V[:, 0, 0, None] * C[:, 0] ** 2, MAX_LOGRATE)
    expected = (jnp.sum(jnp.exp(log_rate) - y * linear) + 0.15 * jnp.sum(C**2)) / len(y)
    actual = poisson_readout_objective(
        (C, d), y, jnp.ones_like(y, dtype=bool), m, V,
        jnp.concatenate((jnp.zeros(1), jnp.full(1, 0.3))),
    )
    npt.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_poisson_ridge_reports_precision_and_unpenalized_objective():
    y = jnp.array([[1., 2.], [3., 1.], [2., 4.], [1., 1.]])
    valid = jnp.ones_like(y, dtype=bool)
    m = jnp.array([[-1.], [0.], [1.], [0.5]])
    V = jnp.full((4, 1, 1), 0.2)
    start = Params(C=jnp.array([[0.2], [0.3]]), d=jnp.zeros(2),
                   R=jnp.asarray(0.), alpha=jnp.array([9.]))
    updated, objective = Poisson.solve_readout(
        "ridge", loading_settings("ridge", {"alpha": 0.7}), start, y, valid, m, V
    )
    expected = poisson_readout_objective((updated.C, updated.d), y, valid, m, V, jnp.zeros(2))
    npt.assert_allclose(updated.alpha, [0.7])
    npt.assert_allclose(objective, expected)
    assert np.isfinite(objective)


def test_ridge_fit_replaces_stale_precision_before_em():
    y = jnp.asarray(np.random.default_rng(42).normal(size=(8, 2)))
    params = Params(C=jnp.array([[0.2], [0.3]]), d=jnp.zeros(2),
                    R=jnp.eye(2), alpha=jnp.asarray(9.0))
    kernel = HidaMatern(1.0, 1.0, 0.0, 0)
    model = CVHM(1, 1.0, [kernel], params=params, loading_config={"alpha": 0.7},
                 max_iter=1, min_iter=1, cvi_iter=0).fit(y, random_state=0)
    npt.assert_allclose(model.params.alpha, [0.7])


def test_masked_gaussian_m_step_matches_conditional_objective():
    y = np.array([[1.0, 4.0], [0.5, 0.7], [-0.3, 1.2], [0.8, -0.5]])
    mask = np.array([[True, False], [True, True], [False, True], [True, True]])
    means = np.array([[0.3, -0.1], [0.5, 0.2], [-0.2, 0.4], [0.1, -0.3]])
    V = np.broadcast_to(np.array([[0.4, 0.05], [0.05, 0.3]]), (len(y), 2, 2))
    variances = np.array([0.6, 0.9])
    params = Params(C=jnp.zeros((2, 2)), d=jnp.zeros(2), R=jnp.diag(jnp.asarray(variances)), alpha=jnp.array([9., 9.]))
    updated, _ = Gaussian.solve_readout(
        "ridge", loading_settings("ridge", None), params,
        jnp.asarray(y[None]), jnp.asarray(mask[None]),
        jnp.asarray(means[None]), jnp.asarray(V[None]),
    )

    X = np.column_stack([np.ones(len(y)), means])
    expected_beta, expected_var = [], []
    for n in range(y.shape[1]):
        selected = mask[:, n]
        A = X[selected].T @ X[selected]
        A[1:, 1:] += V[selected].sum(axis=0)
        b = X[selected].T @ y[selected, n]
        penalised = np.array([0.0, 1.0, 1.0])  # flat bias, penalised loadings
        beta = np.linalg.solve(A + 0.1 * variances[n] * np.diag(penalised), b)
        residual = y[selected, n] - X[selected] @ beta
        variance = np.mean(residual**2 + np.einsum("i,tij,j->t", beta[1:], V[selected], beta[1:]))
        expected_beta.append(beta)
        expected_var.append(variance)
    expected_beta = np.asarray(expected_beta)
    npt.assert_allclose(updated.d, expected_beta[:, 0], rtol=1e-10, atol=1e-10)
    npt.assert_allclose(updated.C, expected_beta[:, 1:], rtol=1e-10, atol=1e-10)
    npt.assert_allclose(updated.R, np.diag(expected_var), rtol=1e-10, atol=1e-10)
    npt.assert_allclose(updated.alpha, [0.1, 0.1])

    def objective(readout):
        C, d, r = map(np.asarray, (readout.C, readout.d, np.diag(readout.R)))
        residual = y - (means @ C.T + d)
        uncertainty = np.einsum("ni,tij,nj->tn", C, V, C)
        likelihood = 0.5 * np.sum(np.where(mask, np.log(r) + (residual**2 + uncertainty) / r, 0))
        return likelihood + 0.05 * np.sum(C**2)

    assert objective(updated) <= objective(params) + 1e-10


def _model(observation="Gaussian"):
    C = jnp.array([[0.2], [0.8], [-0.4]])
    d = jnp.array([0.1, 0.2, -0.1])
    R = jnp.eye(3) if observation == "Gaussian" else jnp.asarray(0.0)
    return CVHM(
        n_components=1, dt=0.5, kernels=[HidaMatern(order=0)],
        params=Params(C=C, d=d, R=R), observation=observation,
        max_iter=3, cvi_iter=1,
    )


@pytest.mark.parametrize("observation", ["Gaussian", "Poisson"])
def test_broadcast_masks_and_final_posterior(observation):
    rng = np.random.default_rng(0)
    y = rng.normal(size=(2, 6, 3)) if observation == "Gaussian" else rng.poisson(3, size=(2, 6, 3))
    y = jnp.asarray(y, dtype=float)
    model = _model(observation)
    time_mask = jnp.array([[[True], [True], [False], [True], [True], [True]],
                           [[True], [False], [True], [True], [True], [True]]])
    feature_mask = jnp.array([[[True, False, True]], [[False, True, True]]])
    masks = (
        (time_mask, jnp.broadcast_to(time_mask, y.shape)),
        (time_mask[..., 0], jnp.broadcast_to(time_mask, y.shape)),
        (feature_mask, jnp.broadcast_to(feature_mask, y.shape)),
    )
    for mask, broadcast in masks:
        for actual, expected in zip(model.infer(y, mask), model.infer(y, broadcast)):
            npt.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)
        compact, full = _model(observation), _model(observation)
        compact.fit(y, mask, random_state=0)
        full.fit(y, broadcast, random_state=0)
        for actual, expected in zip(compact.posterior, full.posterior):
            npt.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)
    for actual, expected in zip(model.infer(y), model.infer(y, jnp.ones_like(y, dtype=bool))):
        npt.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)

    mask = jnp.broadcast_to(feature_mask, y.shape).at[0, 2, 0].set(False)
    model.fit(y, valid_y=mask, random_state=0)
    projected = project(*model.latent, model.latent_mask())
    fresh = model.infer(y, valid_y=mask)
    for cached, from_state, inferred in zip(model.posterior, projected, fresh):
        npt.assert_allclose(cached, from_state, rtol=1e-6, atol=1e-8)
        npt.assert_allclose(cached, inferred, rtol=1e-6, atol=1e-8)


def test_single_sequence_broadcast_masks():
    y = jnp.asarray(np.random.default_rng(7).normal(size=(6, 3)))
    model = _model()
    time_mask = jnp.array([[True], [True], [False], [True], [True], [True]])
    feature_mask = jnp.array([[True, False, True]])
    masks = (
        (time_mask, jnp.broadcast_to(time_mask, y.shape)),
        (time_mask[:, 0], jnp.broadcast_to(time_mask, y.shape)),
        (feature_mask, jnp.broadcast_to(feature_mask, y.shape)),
    )
    for mask, broadcast in masks:
        for actual, expected in zip(model.infer(y, mask), model.infer(y, broadcast)):
            npt.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)
    legacy, expanded = _model(), _model()
    legacy.fit(y, time_mask[:, 0], random_state=0)
    expanded.fit(y, time_mask, random_state=0)
    for actual, expected in zip(legacy.posterior, expanded.posterior):
        npt.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)


def test_masked_gaussian_model_round_trip(tmp_path):
    y = jnp.asarray(np.random.default_rng(5).normal(size=(2, 8, 3)))
    mask = jnp.array([[[True, False, True]], [[False, True, True]]])
    model = _model().fit(y, mask, random_state=0)
    path = tmp_path / "model.cvhmax"
    model.save(path)
    restored = CVHM.load(path)
    for actual, expected in zip(restored.infer(y, mask), model.posterior):
        npt.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)
    npt.assert_allclose(restored.params.R, jnp.diag(jnp.diag(restored.params.R)))


def test_missing_values_are_excluded_from_fit_and_inference():
    y = jnp.asarray(np.random.default_rng(3).normal(size=(2, 8, 3)))
    mask = jnp.array([[[True, False, True]], [[False, True, True]]])
    altered = jnp.where(jnp.broadcast_to(mask, y.shape), y, 500.0)
    first, second = _model(), _model()
    first.params = second.params = None
    first.fit(y, mask, random_state=0)
    second.fit(altered, mask, random_state=0)
    for a, b in zip((first.params.C, first.params.d, first.params.R),
                    (second.params.C, second.params.d, second.params.R)):
        npt.assert_allclose(a, b, rtol=1e-8, atol=1e-8)
    for a, b in zip(first.posterior, second.posterior):
        npt.assert_allclose(a, b, rtol=1e-8, atol=1e-8)


def test_fit_requires_each_feature_to_be_observed():
    y = jnp.ones((2, 4, 3))
    mask = jnp.array([[[True, False, True]], [[True, False, True]]])
    model = _model()
    assert all(np.isfinite(value).all() for value in model.infer(y, mask))
    with pytest.raises(ValueError, match="observed at least once"):
        model.fit(y, mask)


def _ard_model(**kwargs):
    kernel = HidaMatern(sigma=1.0, rho=1.0, omega=0.0, order=0)
    kwargs.setdefault("min_iter", 1)
    return CVHM(
        n_components=2,
        dt=1.0,
        kernels=[kernel, kernel],
        observation="Gaussian",
        loading="ard",
        **kwargs,
    )


def test_ard_fit_rejects_sparse_feature_counts():
    """ARD needs more than ``n_components + 1`` observed time steps per feature."""
    n_time, n_features = 6, 2
    rng = np.random.default_rng(0)
    y = jnp.asarray(rng.standard_normal((1, n_time, n_features)))
    mask = jnp.ones((1, n_time, n_features), dtype=bool)
    # Feature 0 has exactly n_components + 1 = 3 observations.
    mask = mask.at[0, :, 0].set(False)
    mask = mask.at[0, :3, 0].set(True)
    model = _ard_model(max_iter=1, cvi_iter=0)
    with pytest.raises(ValueError, match="more than"):
        model.fit(y, mask, random_state=0)


def test_ard_fit_accepts_sufficient_feature_counts():
    """A feature with more than ``n_components + 1`` observations is accepted."""
    n_time, n_features = 8, 2
    rng = np.random.default_rng(0)
    y = jnp.asarray(rng.standard_normal((1, n_time, n_features)))
    mask = jnp.ones((1, n_time, n_features), dtype=bool)
    model = _ard_model(max_iter=1, cvi_iter=0)
    model.fit(y, mask, random_state=0)
    assert np.isfinite(np.asarray(model.params.alpha)).all()


def test_fit_rejects_invalid_params_alpha():
    """A caller-supplied ``params.alpha`` must be finite and strictly positive."""
    n_time, n_features = 8, 2
    rng = np.random.default_rng(0)
    y = jnp.asarray(rng.standard_normal((1, n_time, n_features)))
    params = Params(
        C=jnp.zeros((n_features, 2)),
        d=jnp.zeros(n_features),
        R=jnp.eye(n_features),
        alpha=jnp.array([-1.0, 1.0]),
    )
    model = _ard_model(params=params, max_iter=1, cvi_iter=0)
    with pytest.raises(ValueError, match="params.alpha"):
        model.fit(y, random_state=0)


@pytest.mark.parametrize("alpha, evidence", [(-1.0, 0.0), (1.0, jnp.nan)])
def test_ard_fit_rejects_invalid_alpha_or_evidence(monkeypatch, alpha, evidence):
    """A fit stops loudly when ARD precisions or evidence become invalid."""
    n_time, n_features = 8, 2
    rng = np.random.default_rng(0)
    y = jnp.asarray(rng.standard_normal((1, n_time, n_features)))
    model = _ard_model(max_iter=1, cvi_iter=0)
    original = Gaussian.solve_readout

    def invalid_solve(cls, loading, settings, params, y, valid_y, m, V):
        updated, _ = original(loading, settings, params, y, valid_y, m, V)
        return (
            Params(
                C=updated.C,
                d=updated.d,
                R=updated.R,
                alpha=jnp.full_like(updated.alpha, alpha),
            ),
            evidence,
        )

    monkeypatch.setattr(Gaussian, "solve_readout", classmethod(invalid_solve))
    with pytest.raises(FloatingPointError, match="non-positive|non-finite"):
        model.fit(y, random_state=0)


@pytest.mark.parametrize("observation", ["Gaussian", "Poisson"])
def test_joint_sequences_share_latent_coordinates(observation):
    """Two partially observed sequences recover a common synthetic trajectory."""
    rng = np.random.default_rng(42)
    n_time, n_features, n_latents = 256, 12, 2
    kernel = HidaMatern(sigma=1.0, rho=6.0, omega=0.0, order=1)
    latent = sample_gp_latents(kernel, n_time, 1.0, n_latents, rng)[0]
    angles = 2 * np.pi * np.arange(n_features) / n_features
    C = 0.7 * np.column_stack((np.cos(angles), np.sin(angles)))
    linear = latent @ C.T
    if observation == "Gaussian":
        y = linear[None, ...] + rng.normal(scale=np.sqrt(0.05), size=(2, n_time, n_features))
    else:
        y = rng.poisson(np.exp(np.log(8.0) + linear), size=(2, n_time, n_features)).astype(float)

    available = np.zeros((2, n_features), dtype=bool)
    available[0, :8] = True
    available[1, 4:] = True
    s, t, n = np.indices(y.shape)
    holdout = available[:, None, :] & ((s + t + n) % 5 == 0)
    training_mask = available[:, None, :] & ~holdout
    model = CVHM(
        n_components=n_latents, dt=1.0, kernels=[kernel, kernel],
        observation=observation,
        max_iter=100, tol=1e-4, min_iter=2, convergence_patience=2,
        cvi_iter=5,
    )
    model.fit(jnp.asarray(y), jnp.asarray(training_mask), random_state=42)
    means, covariances = map(np.asarray, model.posterior)
    estimated_C = np.asarray(model.params.C)
    predictions = np.einsum("btk,nk->btn", means, estimated_C) + np.asarray(model.params.d)
    if observation == "Poisson":
        variance = np.einsum("ni,btij,nj->btn", estimated_C, covariances, estimated_C)
        predictions = np.exp(np.minimum(predictions + 0.5 * variance, MAX_LOGRATE))

    heldout_r2 = []
    for sequence in range(2):
        count = training_mask[sequence].sum(axis=0)
        baseline = np.divide(
            np.where(training_mask[sequence], y[sequence], 0).sum(axis=0),
            count, out=np.zeros(n_features), where=count > 0,
        )
        target = y[sequence][holdout[sequence]]
        expected = predictions[sequence][holdout[sequence]]
        reference = np.broadcast_to(baseline, (n_time, n_features))[holdout[sequence]]
        r2 = 1 - np.sum((target - expected) ** 2) / np.sum((target - reference) ** 2)
        heldout_r2.append(r2)
        assert r2 > 0, (observation, sequence, r2)

    truth = np.broadcast_to(latent, means.shape)
    U, _, Vt = np.linalg.svd(means.reshape(-1, n_latents).T @ truth.reshape(-1, n_latents))
    aligned = means @ (U @ Vt)
    latent_r2 = []
    for sequence in range(2):
        score = 1 - np.sum((aligned[sequence] - latent) ** 2) / np.sum(latent**2)
        latent_r2.append(score)
        assert score > 0.5, (observation, sequence, score)
    assert np.isfinite(covariances).all()
    print(
        f"{observation}: iterations={model.n_iter_}, converged={model.converged_}, "
        f"heldout_r2={np.round(heldout_r2, 4).tolist()}, "
        f"latent_r2={np.round(latent_r2, 4).tolist()}"
    )


def test_overlapping_features_vdp_example(tmp_path):
    """The VdP example fits overlapping sequences in a shared coordinate space."""
    script = """
import sys
import numpy as np
sys.path.insert(0, "examples")  # the demos import their shared helper as a sibling
from demo_vdp import run_overlapping_features_case
model, scores = run_overlapping_features_case(T=100, max_iter=3, out_path=sys.argv[1])
assert model.posterior[0].shape == (2, 100, 2)
assert model.params.C.shape == (60, 2)
assert scores['recorded_features'] == [40, 40]
assert scores['overlap_features'] == 20
assert np.isfinite(scores['heldout_r2']).all()
assert np.isfinite(scores['latent_r2']).all()
"""
    env = {**os.environ, "JAX_ENABLE_X64": "1", "JAX_PLATFORMS": "cpu", "MPLBACKEND": "Agg"}
    subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "overlap.pdf")],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def test_pad_sequences_broadcasts_feature_and_time_masks():
    ys = [jnp.ones((3, 3)), jnp.ones((5, 3))]
    masks = [jnp.array([[True, False, True]]), jnp.array([[True], [True], [False], [True], [True]])]
    _, mask, lengths = pad_sequences(ys, masks)
    assert mask.shape == (2, 5, 3)
    npt.assert_array_equal(mask[0, :3], np.broadcast_to(masks[0], (3, 3)))
    npt.assert_array_equal(mask[0, 3:], False)
    npt.assert_array_equal(mask[1], np.broadcast_to(masks[1], (5, 3)))
    npt.assert_array_equal(lengths, [3, 5])

    time_masks = [jnp.array([True, False, True]), jnp.array([True, True, False, True, True])]
    _, legacy_mask, _ = pad_sequences(ys, time_masks)
    npt.assert_array_equal(legacy_mask[0, :3], np.broadcast_to(time_masks[0][:, None], (3, 3)))
    npt.assert_array_equal(legacy_mask[1], np.broadcast_to(time_masks[1][:, None], (5, 3)))
