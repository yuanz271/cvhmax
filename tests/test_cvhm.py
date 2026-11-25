import jax
import numpy as np
import pytest
from jax import numpy as jnp

from cvhmax.cvhm import CVHM, lift, project, readout_change, sde2gp
from cvhmax.cvi import Params
from cvhmax.hm import HidaMatern, sample_matern
from cvhmax.utils import real_repr

X64_ENABLED = jax.config.read("jax_enable_x64")
ATOL_TIGHT = 1e-10 if X64_ENABLED else 1e-6
ATOL_MED = 1e-12 if X64_ENABLED else 1e-6


@pytest.mark.parametrize("order", [0, 1, 2])
def test_correlation_scaling_matches_reference_transform(order):
    kernel = HidaMatern(sigma=1.5, rho=0.8, omega=2.0, order=order, s=0.0)
    model = CVHM(n_components=1, dt=0.5, kernels=[kernel])

    scale = np.asarray(kernel.state_scale())
    K0 = np.asarray(kernel.K(0.0))
    expected_K0 = scale[:, None] * K0 * scale[None, :]
    Q0 = np.asarray(model.Q0())

    np.testing.assert_allclose(
        Q0,
        np.asarray(real_repr(expected_K0)),
        atol=ATOL_TIGHT,
    )
    np.testing.assert_allclose(
        np.diag(Q0), np.ones(2 * kernel.nple), atol=ATOL_TIGHT
    )

    # The real observation coordinate is recovered from the normalized state
    # with the inverse function-coordinate scale.
    np.testing.assert_allclose(
        model.latent_mask()[0, 0], 1.0 / scale[0], atol=ATOL_TIGHT
    )

    Af = np.asarray(model.Af())
    Qf = np.asarray(model.Qf())
    np.testing.assert_allclose(Af @ Q0 @ Af.T + Qf, Q0, atol=ATOL_TIGHT)


@pytest.mark.parametrize("order", [0, 1, 2])
def test_scaled_dynamics_stable_for_supported_orders(order):
    """Correlation scaling keeps built-in filtering dynamics numerically valid."""
    kernel = HidaMatern(sigma=0.5, rho=0.3, omega=10.0, order=order, s=0.0)
    model = CVHM(n_components=1, dt=0.01, kernels=[kernel])

    K0 = np.asarray(model.Q0())
    Af = np.asarray(model.Af())
    Qf = np.asarray(model.Qf())
    eigvals = np.linalg.eigvalsh((Qf + Qf.T) / 2)
    residual = Af @ K0 @ Af.T + Qf - K0

    assert np.all(np.isfinite(K0))
    assert np.all(np.isfinite(Af))
    assert np.all(np.isfinite(Qf))
    assert eigvals.min() >= -1e-8
    np.testing.assert_allclose(residual, 0.0, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize(
    "orders",
    [
        [0],
        [0, 0],
        [1, 1],
        [2, 2],
        [0, 1],
        [1, 0],
        [0, 1, 2],
    ],
)
def test_latent_mask(orders):
    kernels = [HidaMatern(order=p) for p in orders]
    K = len(orders)
    L = 2 * sum(k.nple for k in kernels)

    model = CVHM(n_components=K, dt=1.0, kernels=kernels)
    M = model.latent_mask()

    assert M.shape == (K, L)
    # The mask contains the inverse state-correlation scale in the observed
    # coordinate, so its row sums are not exactly one when jitter is nonzero.
    assert jnp.all(M.sum(axis=1) > 0.0)
    assert int(jnp.count_nonzero(M)) == K

    # Each kernel block gets [1, 2, ..., nple]; M selects the first element.
    parts = [jnp.arange(1, k.nple + 1, dtype=float) for k in kernels]
    real_half = jnp.concatenate(parts)
    z = jnp.concatenate([real_half, jnp.zeros_like(real_half)])
    assert jnp.allclose(M @ z, jnp.ones(K))


def test_lift_shapes():
    """``lift`` maps latent-space info to state-space info with correct shapes."""
    T, K, L = 20, 3, 8
    M = jnp.zeros((K, L)).at[0, 0].set(1).at[1, 2].set(1).at[2, 4].set(1)
    j_latent = jnp.ones((T, K))
    J_latent = jnp.eye(K)[None, ...].repeat(T, axis=0)

    j, J = lift(j_latent, J_latent, M)

    assert j.shape == (T, L)
    assert J.shape == (T, L, L)


def test_lift_values():
    """``lift`` correctly applies M to information parameters."""
    K, L = 2, 4
    M = jnp.zeros((K, L)).at[0, 0].set(1).at[1, 2].set(1)
    j_latent = jnp.array([3.0, 5.0])
    J_latent = jnp.array([[2.0, 1.0], [1.0, 4.0]])

    j, J = lift(j_latent, J_latent, M)

    # j = j_latent @ M
    np.testing.assert_allclose(j, j_latent @ M)
    # J = M.T @ J_latent @ M
    np.testing.assert_allclose(J, M.T @ J_latent @ M)


def test_lift_batched():
    """``lift`` handles leading batch dimensions."""
    sequences, T, K, L = 2, 10, 2, 6
    M = jnp.zeros((K, L)).at[0, 0].set(1).at[1, 2].set(1)
    j_latent = jnp.ones((sequences, T, K))
    J_latent = jnp.eye(K)[None, None, ...].repeat(T, axis=1).repeat(sequences, axis=0)

    j, J = lift(j_latent, J_latent, M)

    assert j.shape == (sequences, T, L)
    assert J.shape == (sequences, T, L, L)
    np.testing.assert_allclose(j[0, 0], j_latent[0, 0] @ M)
    np.testing.assert_allclose(J[0, 0], M.T @ J_latent[0, 0] @ M)


def test_project_roundtrip():
    """``project`` recovers latent moments consistent with state embedding."""
    K, L = 2, 4
    M = jnp.zeros((K, L)).at[0, 0].set(1).at[1, 2].set(1)

    # Start from known latent moments
    m_true = jnp.array([[1.0, 2.0]])  # (1, K)
    V_true = jnp.array([[[0.5, 0.1], [0.1, 0.3]]])  # (1, K, K)

    # Build a full-rank state covariance embedding the latent moments
    V_state = jnp.eye(L) * 0.1 + M.T @ V_true[0] @ M
    Z_state = jnp.linalg.inv(V_state)
    m_state = m_true[0] @ M  # (L,)
    z_state = Z_state @ m_state  # (L,)

    # Shape: (sequences=1, time=1, ...)
    z = z_state[None, None, :]
    Z = Z_state[None, None, :, :]

    m_rec, V_rec = project(z, Z, M)

    # m_rec = solve(Z, z) @ M.T = m_state @ M.T = m_true
    np.testing.assert_allclose(m_rec[0, 0], m_true[0], atol=ATOL_TIGHT)
    # V_rec = M @ inv(Z) @ M.T
    V_exp = M @ jnp.linalg.inv(Z_state) @ M.T
    np.testing.assert_allclose(V_rec[0, 0], V_exp, atol=ATOL_TIGHT)


def test_project_matches_sde2gp():
    """``project`` delegates to ``sde2gp`` and returns identical results."""
    T, K, L = 5, 2, 4
    rng = np.random.default_rng(99)
    M = jnp.zeros((K, L)).at[0, 0].set(1).at[1, 2].set(1)

    z = jnp.array(rng.standard_normal((1, T, L)))
    Z_raw = rng.standard_normal((1, T, L, L))
    Z = jnp.einsum("...ij,...kj->...ik", Z_raw, Z_raw) + 10 * jnp.eye(L)

    m1, V1 = project(z, Z, M)
    m2, V2 = sde2gp(z, Z, M)

    np.testing.assert_allclose(np.asarray(m1), np.asarray(m2), atol=ATOL_MED)
    np.testing.assert_allclose(np.asarray(V1), np.asarray(V2), atol=ATOL_MED)


def test_default_poisson_cvi_refines_posterior():
    """The default resolves a nonlinear Poisson posterior better than five steps."""
    rng = np.random.default_rng(7)
    n_time, n_features, n_latents, dt = 80, 8, 2, 0.2
    kernel = HidaMatern(sigma=1.0, rho=1.0, omega=0.0, order=1, s=1e-5)
    lags = np.abs(np.arange(n_time)[:, None] - np.arange(n_time)[None, :]) * dt
    prior = np.asarray(kernel.kernel(jnp.asarray(lags))) + 1e-5 * np.eye(n_time)
    latent = np.linalg.cholesky(prior) @ rng.standard_normal((n_time, n_latents))
    C = 0.9 * rng.standard_normal((n_features, n_latents))
    d = np.full(n_features, 1.5)
    y = jnp.asarray(rng.poisson(np.exp(latent @ C.T + d)).astype(float)[None])
    mask = jnp.ones(y.shape, dtype=bool)
    params = Params(C=jnp.asarray(C), d=jnp.asarray(d), R=jnp.asarray(0.0))

    def model(**kwargs):
        return CVHM(
            n_components=n_latents, dt=dt, kernels=[kernel] * n_latents,
            observation="Poisson", **kwargs,
        )

    default = model()
    assert default.cvi_iter == 20
    context = default._inference_context(1)
    means = [
        np.asarray(m._infer_state(params, y, mask, context)[2])
        for m in (model(cvi_iter=5), default, model(cvi_iter=80))
    ]
    reference = means[-1]
    old_error, default_error = [
        np.linalg.norm(mean - reference) / np.linalg.norm(reference)
        for mean in means[:2]
    ]
    assert old_error > 0.1
    assert default_error < 0.05


def test_gaussian_inference_does_not_require_cvi_iterations():
    kernel = HidaMatern(sigma=1.0, rho=1.0, omega=0.0, order=1)
    y = jnp.asarray(np.random.default_rng(4).normal(size=(1, 20, 2)))
    mask = jnp.ones(y.shape, dtype=bool)
    params = Params(C=jnp.asarray([[0.4], [-0.7]]), d=jnp.zeros(2), R=jnp.eye(2))
    default = CVHM(n_components=1, dt=0.2, kernels=[kernel])
    no_iterations = CVHM(n_components=1, dt=0.2, kernels=[kernel], cvi_iter=0)
    context = default._inference_context(1)
    default_moments = default._infer_state(params, y, mask, context)[2:]
    direct_moments = no_iterations._infer_state(params, y, mask, context)[2:]
    for actual, expected in zip(default_moments, direct_moments):
        np.testing.assert_allclose(actual, expected, rtol=ATOL_TIGHT, atol=ATOL_TIGHT)


def test_CVHM(capsys):
    np.random.seed(1234)
    T = 2000
    n_obs = 50
    n_factors = 2
    dt = 1.0
    sigma = 1.0
    rho = 50.0

    kernels = [HidaMatern(sigma, rho, 0.0, 0) for k in range(n_factors)]
    # params = Params()

    x = np.column_stack(
        [sample_matern(T, dt, sigma, rho), sample_matern(T, dt, sigma, rho)]
    )

    C = np.random.rand(n_obs, n_factors)
    d = np.ones(n_obs) + 1

    y = np.random.poisson(np.exp(x @ C.T + np.expand_dims(d, 0)))
    y = jnp.array(y, dtype=float)
    y = jnp.expand_dims(y, 0)

    with capsys.disabled():
        model = CVHM(n_factors, dt, kernels, max_iter=20, observation="Poisson")
        result = model.fit(y)
    m, V = result.posterior
    m = m[0]
    V = V[0]

    assert m.shape == (T, n_factors)
    assert V.shape == (T, n_factors, n_factors)


def test_readout_change_is_invariant_to_glm_rotation():
    C = jnp.asarray([[1.0, 0.0], [0.0, 2.0], [1.0, -1.0]])
    theta = 0.37
    rotation = jnp.asarray(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    old = Params(C=C, d=jnp.zeros(3), R=jnp.eye(3))
    new = Params(C=C @ rotation, d=old.d, R=old.R)
    assert float(readout_change(new, old)) < 1e-6


def test_readout_change_detects_loading_gain():
    C = jnp.asarray([[0.2], [2.0]])
    old = Params(C=C, d=jnp.zeros(2), R=jnp.eye(2))
    new = Params(C=2 * C, d=old.d, R=old.R)
    np.testing.assert_allclose(readout_change(new, old), 1.0, atol=1e-6)


def test_readout_change_detects_small_loading_change():
    old = Params(C=jnp.array([[0.1]]), d=jnp.zeros(1), R=jnp.eye(1))
    new = Params(C=jnp.array([[0.2]]), d=old.d, R=old.R)
    np.testing.assert_allclose(readout_change(new, old), 0.1, atol=1e-10)


def test_readout_change_reduced_gram_matches_explicit():
    rng = np.random.default_rng(84)
    for n, k in ((5, 2), (2, 5), (50, 3)):
        old_C, new_C = rng.normal(size=(2, n, k))
        old = Params(C=jnp.asarray(old_C), d=jnp.zeros(n), R=jnp.eye(n))
        new = Params(C=jnp.asarray(new_C), d=old.d, R=old.R)
        expected = np.linalg.norm(new_C @ new_C.T - old_C @ old_C.T) / (
            (np.linalg.norm(new_C) + np.linalg.norm(old_C)) * max(1, np.linalg.norm(old_C))
        )
        np.testing.assert_allclose(readout_change(new, old), expected, rtol=1e-10)


def test_ard_change_detects_rotation_and_precision():
    C = jnp.array([[1., 0.], [0., 2.]])
    old = Params(C=C, d=jnp.zeros(2), R=jnp.eye(2), alpha=jnp.array([1., 2.]))
    rotated = Params(C=C @ jnp.array([[0., -1.], [1., 0.]]), d=old.d, R=old.R, alpha=old.alpha)
    assert float(readout_change(rotated, old)) < 1e-10
    assert float(readout_change(rotated, old, ard=True)) > 0.05
    changed_alpha = Params(C=C, d=old.d, R=old.R, alpha=old.alpha * 2)
    np.testing.assert_allclose(readout_change(changed_alpha, old, ard=True), np.log(2))


def test_readout_change_detects_a_change_across_features():
    """Only latent rotations are gauge; re-weighting features is a real change."""
    C = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    old = Params(C=C, d=jnp.zeros(3), R=jnp.eye(3))
    new = Params(C=C[jnp.asarray([2, 1, 0])], d=old.d, R=old.R)
    assert float(readout_change(new, old)) > 0.3


def test_ard_fit_does_not_converge_while_precisions_change(monkeypatch):
    from cvhmax.estimator import ARDEstimator

    def changing_precision(self, likelihood, params, y, valid_y, m, V):
        return Params(C=params.C, d=params.d, R=params.R, alpha=2 * params.alpha), jnp.array(0.0)

    monkeypatch.setattr(ARDEstimator, "estimate", changing_precision)
    params = Params(C=jnp.array([[0.2], [0.4]]), d=jnp.zeros(2),
                    R=jnp.eye(2), alpha=jnp.ones(1))
    kernel = HidaMatern(1.0, 1.0, 0.0, 0)
    model = CVHM(1, 1.0, [kernel], params=params, estimator="ARDEstimator",
                 max_iter=4, min_iter=1, convergence_patience=1, cvi_iter=0)
    model.fit(jnp.asarray(np.random.default_rng(8).normal(size=(8, 2))), random_state=0)
    assert model.n_iter_ == 4
    assert not model.converged_
    np.testing.assert_allclose(model.params.alpha, [16.0])


def test_convergence_stops_a_fit_and_reports_diagnostics():
    model = CVHM(
        n_components=1,
        dt=0.5,
        kernels=[HidaMatern(sigma=1.2, rho=2.0, omega=0.3, order=1, s=1e-5)],
        lr=0.2,
        max_iter=10,
        cvi_iter=1,
        tol=1.0,
    ).fit(jnp.asarray(np.random.default_rng(0).normal(size=(1, 16, 2))), random_state=0)
    assert model.converged_ is True
    assert model.min_iter <= model.n_iter_ < model.max_iter


def test_ard_recovers_latent_dimensionality_in_the_full_fit():
    """Inside the EM loop, ARD zeroes surplus latents that fixed ridge only shrinks."""
    rng = np.random.default_rng(0)
    n_time, n_features, n_true, n_fit = 160, 16, 2, 4
    kernel = HidaMatern(sigma=1.0, rho=8.0, omega=0.0, order=1)
    lags = np.abs(np.arange(n_time)[:, None] - np.arange(n_time)[None, :])
    prior = np.asarray(kernel.kernel(jnp.asarray(lags))) + kernel.s * np.eye(n_time)
    chol = np.linalg.cholesky(prior)
    C = rng.normal(scale=0.8, size=(n_features, n_true))
    latents = np.stack([chol @ rng.standard_normal((n_time, n_true)) for _ in range(2)])
    y = latents @ C.T + rng.normal(scale=0.3, size=(2, n_time, n_features))

    def loading_norms(estimator):
        model = CVHM(
            n_components=n_fit,
            dt=1.0,
            kernels=[kernel] * n_fit,
            observation="Gaussian",
            estimator=estimator,
            max_iter=200,
            tol=1e-4,
            min_iter=5,
            convergence_patience=3,
        ).fit(jnp.asarray(y), random_state=0)
        norms = np.sort(np.linalg.norm(np.asarray(model.params.C), axis=0))[::-1]
        return norms, np.sort(np.asarray(model.params.alpha))

    ard_norms, ard_alpha = loading_norms("ARDEstimator")
    ridge_norms, _ = loading_norms("RidgeEstimator")

    # ARD: surplus columns are pruned to numerical zero and carry far larger precision.
    assert ard_norms[n_true:].max() < 1e-6 * ard_norms[n_true - 1]
    assert ard_alpha[n_true] > 100 * ard_alpha[n_true - 1]
    # Ridge only shrinks them, leaving nonzero loadings.
    assert ridge_norms[n_true:].min() > 1e-3 * ridge_norms[n_true - 1]


def test_frozen_estimator_leaves_the_supplied_readout_untouched():
    """A frozen fit infers latents under the given readout and never refits it."""
    rng = np.random.default_rng(0)
    kernels = [HidaMatern(1.0, 8.0, 0.0, 1)]
    start = Params(
        C=jnp.asarray(rng.normal(size=(4, 1))),
        d=jnp.asarray(rng.normal(size=4)),
        R=jnp.diag(jnp.asarray([0.5, 0.6, 0.7, 0.8])),
    )
    y = jnp.asarray(rng.normal(size=(1, 40, 4)))

    model = CVHM(
        n_components=1, dt=1.0, kernels=kernels, observation="Gaussian",
        estimator="FrozenEstimator", params=start, max_iter=5,
    ).fit(y, random_state=0)

    for name in ("C", "d", "R"):
        np.testing.assert_array_equal(
            np.asarray(getattr(model.params, name)), np.asarray(getattr(start, name))
        )
    assert np.isfinite(np.asarray(model.posterior[0])).all()


def test_ard_estimator_rejects_a_non_conjugate_readout():
    """ARD needs a conjugate loading block, which Poisson does not provide."""
    kernels = [HidaMatern(1.0, 1.0, 0.0, 0)]
    with pytest.raises(ValueError, match="conjugate loading"):
        CVHM(
            n_components=1,
            dt=1.0,
            kernels=kernels,
            observation="Poisson",
            estimator="ARDEstimator",
        )


def test_estimators_are_exported_at_top_level():
    """The documented estimator classes are importable from the package root."""
    import cvhmax
    from cvhmax import estimator as estimator_module

    for name in ("Estimator", "RidgeEstimator", "ARDEstimator", "FrozenEstimator"):
        assert name in cvhmax.__all__
        assert getattr(cvhmax, name) is getattr(estimator_module, name)


def test_progress_callback_is_ordered_and_idempotent(monkeypatch):
    class FakeProgress:
        def __init__(self):
            self.add_task_calls = []
            self.update_calls = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def add_task(self, description, total, **fields):
            self.add_task_calls.append(
                {"description": description, "total": total, "fields": fields}
            )
            return 123

        def update(self, task_id, **kwargs):
            self.update_calls.append((task_id, kwargs))

    fake_pbar = FakeProgress()

    import cvhmax.cvhm as cvhm_mod
    from cvhmax.cvi import Params
    from cvhmax.estimator import Estimator

    monkeypatch.setattr(cvhm_mod, "training_progress", lambda: fake_pbar)

    class StaticEstimator(Estimator):
        """Fixed readout parameters and no objective, to exercise progress."""

        def initialize_params(
            self, likelihood, y, valid_y, n_factors, *, random_state, params=None
        ) -> Params:
            obs_dim = y.shape[-1]
            return Params(
                C=jnp.zeros((obs_dim, n_factors), dtype=y.dtype),
                d=jnp.zeros(obs_dim, dtype=y.dtype),
                R=jnp.zeros((obs_dim, obs_dim), dtype=y.dtype),
            )

        def estimate(self, likelihood, params, y, valid_y, m, V):
            return params, jnp.nan

    recorded = []
    import jax.debug as jdebug

    def record_debug_callback(
        callback, *args, ordered=False, partitioned=False, **kwargs
    ):
        recorded.append(
            {
                "callback": callback,
                "args": args,
                "ordered": ordered,
                "partitioned": partitioned,
                "kwargs": kwargs,
            }
        )

    monkeypatch.setattr(jdebug, "callback", record_debug_callback)

    n_devices = len(jax.devices())
    sequences = max(1, n_devices)
    T = 4
    obs_dim = 3
    n_components = 1

    y = jnp.zeros((sequences, T, obs_dim), dtype=jnp.float64)
    valid_y = jnp.ones((sequences, T, 1), dtype=bool)

    kernels = [HidaMatern(1.0, 1.0, 0.0, 0)]
    model = CVHM(
        n_components=n_components,
        dt=1.0,
        kernels=kernels,
        observation="Poisson",
        estimator="StaticEstimator",
        max_iter=3,
    )

    model.fit(y, valid_y=valid_y, random_state=0)

    assert any(call["ordered"] is True for call in recorded), recorded

    cb_calls = [call for call in recorded if len(call["args"]) == 2]
    assert cb_calls, recorded

    cb = cb_calls[0]["callback"]
    cb(0, 1.23)
    cb(1, np.nan)
    cb(2, 4.56)

    assert fake_pbar.add_task_calls, "Expected add_task() to be called"
    assert fake_pbar.update_calls, "Expected update() to be called"
    assert fake_pbar.add_task_calls[0]["fields"]["nell_display"] == "n/a"

    for _, kwargs in fake_pbar.update_calls:
        assert "completed" in kwargs
        assert "advance" not in kwargs
        assert 1 <= kwargs["completed"] <= 3
        assert isinstance(kwargs.get("nell"), float)
        expected_display = "n/a" if not np.isfinite(kwargs["nell"]) else f"{kwargs['nell']:.3f}"
        assert kwargs["nell_display"] == expected_display
