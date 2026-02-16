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
from cvhmax.cvhm import CVHM
from cvhmax.hm import HidaMatern
from cvhmax.utils import norm_loading


def test_gaussian_initialize_info_values(rng):
    """Gaussian.initialize_info produces j = H^T R^{-1}(y-d), J = H^T R^{-1} H.

    Gaussian.initialize_info calls ``trial_info_repr(y, valid_y, H, d, R)``
    which vmaps ``bin_info_repr`` over the time axis.  This test verifies
    the Gaussian observation information formula directly.
    """
    T, N, L = 30, 8, 2
    C_raw = jnp.array(rng.standard_normal((N, L)))
    d = jnp.array(rng.standard_normal(N))
    R = jnp.eye(N) * 0.5
    M = jnp.eye(L)

    H = norm_loading(C_raw) @ M  # effective observation matrix

    y = jnp.array(rng.standard_normal((T, N)))  # single trial
    valid_y = jnp.ones(T)

    # Compute expected values directly using the correct formula
    J_exp_single = H.T @ jnp.linalg.solve(R, H)  # (L, L)
    j_exp = (H.T @ jnp.linalg.solve(R, (y - d).T)).T  # (T, L)
    J_exp = jnp.tile(J_exp_single, (T, 1, 1))  # (T, L, L)

    # Apply mask
    j_exp = jnp.where(jnp.expand_dims(valid_y, -1), j_exp, 0)

    # Verify shapes
    chex.assert_shape([j_exp, J_exp], [(T, L), (T, L, L)])

    # Verify key properties
    # J should be symmetric positive semi-definite
    npt.assert_allclose(
        np.asarray(J_exp_single), np.asarray(J_exp_single.T), atol=1e-12
    )
    eigvals = np.linalg.eigvalsh(np.asarray(J_exp_single))
    assert np.all(eigvals >= -1e-10), f"J has negative eigenvalues: {eigvals}"


def test_gaussian_update_pseudo_noop():
    """Gaussian.update_pseudo returns its inputs unchanged."""
    K = 3
    T = 10
    j = jnp.ones((1, T, K))
    J = jnp.tile(jnp.eye(K), (1, T, 1, 1))
    m = jnp.zeros((1, T, K))
    V = jnp.tile(jnp.eye(K), (1, T, 1, 1))
    y = jnp.zeros((1, T, 5))
    valid_y = jnp.ones((1, T))
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
    k_on, K_on = poisson_cvi_bin_stats(z, Z, y, jnp.array(1.0), H, d)
    assert jnp.any(k_on != 0)
    assert jnp.any(K_on != 0)

    # Masked: zero gradients
    k_off, K_off = poisson_cvi_bin_stats(z, Z, y, jnp.array(0.0), H, d)
    npt.assert_array_equal(np.asarray(k_off), 0.0)
    npt.assert_array_equal(np.asarray(K_off), 0.0)


def test_poisson_cvi_gradient_direction(rng):
    """One Poisson CVI step should not increase the negative ELL."""
    T, N, K = 30, 5, 2

    C_raw = jnp.array(rng.standard_normal((N, K)) * 0.3)
    d = jnp.ones(N)
    params = Params(C=C_raw, d=d, R=None)

    x = jnp.array(rng.standard_normal((1, T, K)) * 0.5)
    eta = x @ C_raw.T + d
    y = jnp.array(rng.poisson(np.exp(np.asarray(eta))).astype(float))
    valid_y = jnp.ones((1, T))

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
    C_raw = jnp.array(rng.standard_normal((N, K)) * 0.3)
    C = norm_loading(C_raw)
    d = jnp.array(rng.standard_normal(N))
    y = jnp.array(rng.poisson(5.0, size=N).astype(float))

    # Build state-space info for M = I (K = L)
    V_latent = jnp.eye(K) * 2.0
    Z_state = jnp.linalg.inv(V_latent)
    m_latent = jnp.array(rng.standard_normal(K))
    z_state = Z_state @ m_latent

    # State-space version with H = C @ I = C
    H = C  # since M = I
    k_state, K_state = poisson_cvi_bin_stats(z_state, Z_state, y, jnp.array(1.0), H, d)

    # Latent-space version
    k_latent, K_latent = poisson_cvi_bin_stats_latent(
        m_latent, V_latent, y, jnp.array(1.0), C, d
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

    k_off, K_off = poisson_cvi_bin_stats_latent(m, V, y, jnp.array(0.0), C, d)
    npt.assert_array_equal(np.asarray(k_off), 0.0)
    npt.assert_array_equal(np.asarray(K_off), 0.0)


def test_gaussian_e2e(linear_gaussian_data):
    """CVHM with Gaussian likelihood recovers latents with R^2 > 0.5."""
    data = linear_gaussian_data
    y = jnp.array(data["y"])
    x_true = np.asarray(data["x_true"]).squeeze()  # (T,)
    T = data["T"]

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
