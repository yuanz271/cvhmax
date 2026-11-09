"""Mathematical and integration tests for neuron-level observation masks."""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from cvhmax.cvi import (
    MAX_LOGRATE,
    Gaussian,
    Params,
    poisson_cvi_bin_stats_latent,
    poisson_trial_nell,
)
from cvhmax.cvhm import CVHM, lift, project
from cvhmax.hm import HidaMatern
from cvhmax.utils import pad_trials


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

    def reference(params):
        C, d = params
        total = 0.0
        for t, indices in enumerate(([0, 2], [1, 2])):
            Ct, dt, yt = C[jnp.array(indices)], d[jnp.array(indices)], y[t, jnp.array(indices)]
            linear = Ct @ means[t] + dt
            quad = jnp.einsum("ni,ij,nj->n", Ct, covariances[t], Ct)
            eta = jnp.minimum(linear + 0.5 * quad, MAX_LOGRATE)
            total += jnp.sum(jnp.exp(eta) - yt * eta)
        return (total + gamma * jnp.linalg.norm(C)) / 2

    def objective(params):
        return poisson_trial_nell(params, y, mask, means, covariances, gamma)

    npt.assert_allclose(objective((C, d)), reference((C, d)), rtol=1e-10, atol=1e-10)
    actual_grad = jax.grad(objective)((C, d))
    expected_grad = jax.grad(reference)((C, d))
    for actual, expected in zip(actual_grad, expected_grad):
        npt.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)

    j, J = poisson_cvi_bin_stats_latent(means[0], covariances[0], y[0], mask[0], C, d)
    observed_C = np.asarray(C)[[0, 2]]
    observed_d = np.asarray(d)[[0, 2]]
    observed_y = np.asarray(y[0])[[0, 2]]
    m, V = np.asarray(means[0]), np.asarray(covariances[0])
    rates = np.exp(observed_C @ m + observed_d + 0.5 * np.einsum("ni,ij,nj->n", observed_C, V, observed_C))
    expected_J = observed_C.T @ (rates[:, None] * observed_C)
    expected_j = (observed_y - rates) @ observed_C + expected_J @ m
    npt.assert_allclose(j, expected_j, rtol=1e-10, atol=1e-10)
    npt.assert_allclose(J, expected_J, rtol=1e-10, atol=1e-10)


def test_all_observed_poisson_objective_preserves_bin_scaling():
    C = jnp.array([[0.2], [0.7]])
    d = jnp.array([0.1, -0.2])
    y = jnp.array([[1.0, 3.0], [2.0, 1.0]])
    m = jnp.array([[0.2], [-0.1]])
    V = jnp.array([[[0.3]], [[0.4]]])
    eta = jnp.minimum(m @ C.T + d + 0.5 * V[:, 0, 0, None] * C[:, 0] ** 2, MAX_LOGRATE)
    expected = (jnp.sum(jnp.exp(eta) - y * eta) + 0.3 * jnp.linalg.norm(C)) / len(y)
    actual = poisson_trial_nell((C, d), y, jnp.ones_like(y, dtype=bool), m, V, gamma=0.3)
    npt.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_masked_gaussian_m_step_matches_conditional_objective():
    y = np.array([[1.0, 4.0], [0.5, 0.7], [-0.3, 1.2], [0.8, -0.5]])
    mask = np.array([[True, False], [True, True], [False, True], [True, True]])
    means = np.array([[0.3, -0.1], [0.5, 0.2], [-0.2, 0.4], [0.1, -0.3]])
    V = np.broadcast_to(np.array([[0.4, 0.05], [0.05, 0.3]]), (len(y), 2, 2))
    variances = np.array([0.6, 0.9])
    params = Params(C=jnp.zeros((2, 2)), d=jnp.zeros(2), R=jnp.diag(jnp.asarray(variances)))
    updated, _ = Gaussian.update_readout(
        params, jnp.asarray(y[None]), jnp.asarray(mask[None]),
        jnp.asarray(means[None]), jnp.asarray(V[None]),
    )

    X = np.column_stack([np.ones(len(y)), means])
    expected_beta, expected_var = [], []
    for n in range(y.shape[1]):
        selected = mask[:, n]
        A = X[selected].T @ X[selected]
        A[1:, 1:] += V[selected].sum(axis=0)
        b = X[selected].T @ y[selected, n]
        beta = np.linalg.solve(A + 0.1 * variances[n] * np.eye(3), b)
        residual = y[selected, n] - X[selected] @ beta
        variance = np.mean(residual**2 + np.einsum("i,tij,j->t", beta[1:], V[selected], beta[1:]))
        expected_beta.append(beta)
        expected_var.append(variance)
    expected_beta = np.asarray(expected_beta)
    npt.assert_allclose(updated.d, expected_beta[:, 0], rtol=1e-10, atol=1e-10)
    npt.assert_allclose(updated.C, expected_beta[:, 1:], rtol=1e-10, atol=1e-10)
    npt.assert_allclose(updated.R, np.diag(expected_var), rtol=1e-10, atol=1e-10)

    def objective(readout):
        C, d, r = map(np.asarray, (readout.C, readout.d, np.diag(readout.R)))
        residual = y - (means @ C.T + d)
        uncertainty = np.einsum("ni,tij,nj->tn", C, V, C)
        likelihood = 0.5 * np.sum(np.where(mask, np.log(r) + (residual**2 + uncertainty) / r, 0))
        return likelihood + 0.05 * (np.sum(C**2) + np.sum(d**2))

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
    neuron_mask = jnp.array([[[True, False, True]], [[False, True, True]]])
    masks = (
        (time_mask, jnp.broadcast_to(time_mask, y.shape)),
        (time_mask[..., 0], jnp.broadcast_to(time_mask, y.shape)),
        (neuron_mask, jnp.broadcast_to(neuron_mask, y.shape)),
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

    mask = jnp.broadcast_to(neuron_mask, y.shape).at[0, 2, 0].set(False)
    model.fit(y, valid_y=mask, random_state=0)
    projected = project(*model.latent, model.latent_mask())
    fresh = model.infer(y, valid_y=mask)
    for cached, from_state, inferred in zip(model.posterior, projected, fresh):
        npt.assert_allclose(cached, from_state, rtol=1e-6, atol=1e-8)
        npt.assert_allclose(cached, inferred, rtol=1e-6, atol=1e-8)


def test_single_trial_broadcast_masks():
    y = jnp.asarray(np.random.default_rng(7).normal(size=(6, 3)))
    model = _model()
    time_mask = jnp.array([[True], [True], [False], [True], [True], [True]])
    neuron_mask = jnp.array([[True, False, True]])
    masks = (
        (time_mask, jnp.broadcast_to(time_mask, y.shape)),
        (time_mask[:, 0], jnp.broadcast_to(time_mask, y.shape)),
        (neuron_mask, jnp.broadcast_to(neuron_mask, y.shape)),
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


def test_fit_requires_each_neuron_to_be_observed():
    y = jnp.ones((2, 4, 3))
    mask = jnp.array([[[True, False, True]], [[True, False, True]]])
    model = _model()
    assert all(np.isfinite(value).all() for value in model.infer(y, mask))
    with pytest.raises(ValueError, match="observed at least once"):
        model.fit(y, mask)


@pytest.mark.parametrize("observation", ["Gaussian", "Poisson"])
def test_joint_sessions_share_latent_coordinates(observation):
    """Two partially observed sessions recover a common synthetic trajectory."""
    rng = np.random.default_rng(42)
    n_time, n_neurons, n_latents = 256, 12, 2
    kernel = HidaMatern(sigma=1.0, rho=6.0, omega=0.0, order=1)
    lags = np.abs(np.arange(n_time)[:, None] - np.arange(n_time)[None, :])
    prior = np.asarray(kernel.kernel(jnp.asarray(lags))) + kernel.s * np.eye(n_time)
    latent = np.linalg.cholesky(prior) @ rng.standard_normal((n_time, n_latents))
    angles = 2 * np.pi * np.arange(n_neurons) / n_neurons
    C = 0.7 * np.column_stack((np.cos(angles), np.sin(angles)))
    linear = latent @ C.T
    if observation == "Gaussian":
        y = linear[None, ...] + rng.normal(scale=np.sqrt(0.05), size=(2, n_time, n_neurons))
    else:
        y = rng.poisson(np.exp(np.log(8.0) + linear), size=(2, n_time, n_neurons)).astype(float)

    available = np.zeros((2, n_neurons), dtype=bool)
    available[0, :8] = True
    available[1, 4:] = True
    s, t, n = np.indices(y.shape)
    holdout = available[:, None, :] & ((s + t + n) % 5 == 0)
    training_mask = available[:, None, :] & ~holdout
    model = CVHM(
        n_components=n_latents, dt=1.0, kernels=[kernel, kernel],
        observation=observation,
        max_iter=100, tol=1e-4, min_iter=2, convergence_patience=2,
        cvi_iter=5, lr=0.1,
    )
    model.fit(jnp.asarray(y), jnp.asarray(training_mask), random_state=42)
    means, covariances = map(np.asarray, model.posterior)
    estimated_C = np.asarray(model.params.C)
    predictions = np.einsum("btk,nk->btn", means, estimated_C) + np.asarray(model.params.d)
    if observation == "Poisson":
        variance = np.einsum("ni,btij,nj->btn", estimated_C, covariances, estimated_C)
        predictions = np.exp(np.minimum(predictions + 0.5 * variance, MAX_LOGRATE))

    heldout_r2 = []
    for session in range(2):
        count = training_mask[session].sum(axis=0)
        baseline = np.divide(
            np.where(training_mask[session], y[session], 0).sum(axis=0),
            count, out=np.zeros(n_neurons), where=count > 0,
        )
        target = y[session][holdout[session]]
        expected = predictions[session][holdout[session]]
        reference = np.broadcast_to(baseline, (n_time, n_neurons))[holdout[session]]
        r2 = 1 - np.sum((target - expected) ** 2) / np.sum((target - reference) ** 2)
        heldout_r2.append(r2)
        assert r2 > 0, (observation, session, r2)

    truth = np.broadcast_to(latent, means.shape)
    U, _, Vt = np.linalg.svd(means.reshape(-1, n_latents).T @ truth.reshape(-1, n_latents))
    aligned = means @ (U @ Vt)
    latent_r2 = []
    for session in range(2):
        score = 1 - np.sum((aligned[session] - latent) ** 2) / np.sum(latent**2)
        latent_r2.append(score)
        assert score > 0, (observation, session, score)
    assert np.isfinite(covariances).all()
    print(
        f"{observation}: iterations={model.n_iter_}, converged={model.converged_}, "
        f"heldout_r2={np.round(heldout_r2, 4).tolist()}, "
        f"latent_r2={np.round(latent_r2, 4).tolist()}"
    )


def test_pad_trials_broadcasts_neuron_and_time_masks():
    ys = [jnp.ones((3, 3)), jnp.ones((5, 3))]
    masks = [jnp.array([[True, False, True]]), jnp.array([[True], [True], [False], [True], [True]])]
    _, mask, lengths = pad_trials(ys, masks)
    assert mask.shape == (2, 5, 3)
    npt.assert_array_equal(mask[0, :3], np.broadcast_to(masks[0], (3, 3)))
    npt.assert_array_equal(mask[0, 3:], False)
    npt.assert_array_equal(mask[1], np.broadcast_to(masks[1], (5, 3)))
    npt.assert_array_equal(lengths, [3, 5])

    time_masks = [jnp.array([True, False, True]), jnp.array([True, True, False, True, True])]
    _, legacy_mask, _ = pad_trials(ys, time_masks)
    npt.assert_array_equal(legacy_mask[0, :3], np.broadcast_to(time_masks[0][:, None], (3, 3)))
    npt.assert_array_equal(legacy_mask[1], np.broadcast_to(time_masks[1][:, None], (5, 3)))
