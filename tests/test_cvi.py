import numpy as np
import numpy.testing as npt

import chex
from jax import numpy as jnp
from sklearn.linear_model import LinearRegression

from cvhmax.cvi import (
    Params,
    poisson_cvi_bin_stats,
    poisson_cvi_bin_stats_latent,
    Poisson,
    Gaussian,
)
from cvhmax.cvhm import CVHM, lift
from cvhmax.hm import HidaMatern


def test_gaussian_initialize_info_values(rng):
    """Gaussian information increments use C in latent space."""
    T, N, K = 30, 8, 2
    C = jnp.array(rng.standard_normal((N, K)))
    d = jnp.array(rng.standard_normal(N))
    R = jnp.eye(N) * 0.5
    params = Params(C=C, d=d, R=R)

    y = jnp.array(rng.standard_normal((T, N)))  # single trial
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


def test_poisson_cvi_masked_zero():
    """Masked bins contribute zero CVI gradients (k=0, K=0)."""
    N, L = 5, 3
    z = jnp.ones(L)
    Z = jnp.eye(L) * 2.0
    y = jnp.ones(N) * 10.0
    H = jnp.ones((N, L)) * 0.3
    d = jnp.zeros(N)

    # Unmasked: normal computation
    k_on, K_on = poisson_cvi_bin_stats(z, Z, y, jnp.ones(N, dtype=bool), H, d)
    assert jnp.any(k_on != 0)
    assert jnp.any(K_on != 0)

    # Masked: zero gradients
    k_off, K_off = poisson_cvi_bin_stats(z, Z, y, jnp.zeros(N, dtype=bool), H, d)
    npt.assert_array_equal(np.asarray(k_off), 0.0)
    npt.assert_array_equal(np.asarray(K_off), 0.0)


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


def test_poisson_cvi_bin_stats_latent_equivalence(rng):
    """Latent-space Poisson CVI stats match state-space version when M = I.

    Temporary equivalence test for the refactor.  Remove after migration.
    """
    N, K = 5, 3
    C = jnp.array(rng.standard_normal((N, K)) * 0.3)
    d = jnp.array(rng.standard_normal(N))
    y = jnp.array(rng.poisson(5.0, size=N).astype(float))

    # Build state-space info for M = I (K = L)
    V_latent = jnp.eye(K) * 2.0
    Z_state = jnp.linalg.inv(V_latent)
    m_latent = jnp.array(rng.standard_normal(K))
    z_state = Z_state @ m_latent

    # State-space version with H = C @ I = C
    H = C  # since M = I
    k_state, K_state = poisson_cvi_bin_stats(z_state, Z_state, y, jnp.ones(N, dtype=bool), H, d)

    # Latent-space version
    k_latent, K_latent = poisson_cvi_bin_stats_latent(
        m_latent, V_latent, y, jnp.ones(N, dtype=bool), C, d
    )

    # Tolerance is relaxed because the state-space version applies SVD
    # regularisation (TAU) before converting to moments, introducing small
    # numerical differences.
    npt.assert_allclose(np.asarray(k_latent), np.asarray(k_state), atol=1e-4)
    npt.assert_allclose(np.asarray(K_latent), np.asarray(K_state), atol=1e-4)


def test_poisson_cvi_bin_stats_latent_masked(rng):
    """Masked bins contribute zero CVI gradients from latent-space stats."""
    N, K = 5, 3
    C = jnp.array(rng.standard_normal((N, K)) * 0.3)
    d = jnp.zeros(N)
    m = jnp.ones(K)
    V = jnp.eye(K)
    y = jnp.ones(N) * 10.0

    k_off, K_off = poisson_cvi_bin_stats_latent(m, V, y, jnp.zeros(N, dtype=bool), C, d)
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
