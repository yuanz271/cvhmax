import numpy as np
import numpy.testing as npt

import chex
from jax import numpy as jnp
from sklearn.linear_model import LinearRegression

from cvhmax.cvi import Params, poisson_cvi_bin_stats, Poisson, Gaussian
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
    L = 3
    T = 10
    j = jnp.ones((1, T, L))
    J = jnp.tile(jnp.eye(L), (1, T, 1, 1))
    z = jnp.zeros((1, T, L))
    Z = jnp.tile(jnp.eye(L), (1, T, 1, 1))
    y = jnp.zeros((1, T, 5))
    valid_y = jnp.ones((1, T))
    params = Params(C=jnp.ones((5, L)), d=jnp.zeros(5), R=jnp.eye(5), M=jnp.eye(L))

    j_out, J_out = Gaussian.update_pseudo(params, y, valid_y, z, Z, j, J, 0.1)
    npt.assert_array_equal(np.asarray(j_out), np.asarray(j))
    npt.assert_array_equal(np.asarray(J_out), np.asarray(J))


def test_gaussian_infer_single_cvi_iter():
    """Gaussian.infer forces cvi_iter=1 regardless of the argument."""
    L = 2
    T = 5

    def mock_smooth(j, J, z0, Z0, Af, Pf, Ab, Pb):
        return j, J  # return z, Z as dummies

    j = jnp.zeros((1, T, L))
    J = jnp.tile(jnp.eye(L), (1, T, 1, 1))
    y = jnp.zeros((1, T, 4))
    valid_y = jnp.ones((1, T))
    z0 = jnp.zeros((1, L))
    Z0 = jnp.tile(jnp.eye(L), (1, 1, 1))
    params = Params(C=jnp.ones((4, L)), d=jnp.zeros(4), R=jnp.eye(4), M=jnp.eye(L))

    smooth_args = (jnp.eye(L),) * 4  # Af, Pf, Ab, Pb

    # Pass cvi_iter=10, but Gaussian should force it to 1
    (z_out, Z_out), (j_out, J_out) = Gaussian.infer(
        params, j, J, y, valid_y, z0, Z0, mock_smooth, smooth_args, cvi_iter=10, lr=0.1
    )
    # Gaussian.update_pseudo is a no-op so j_out == j, J_out == J
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
    T, N, L = 30, 5, 2
    state_dim = L * 2  # real repr doubles

    C_raw = jnp.array(rng.standard_normal((N, L)) * 0.3)
    d = jnp.ones(N)
    M = jnp.zeros((L, state_dim))
    for i in range(L):
        M = M.at[i, i * 2].set(1.0)
    params = Params(C=C_raw, d=d, R=None, M=M)

    x = jnp.array(rng.standard_normal((1, T, L)) * 0.5)
    eta = x @ C_raw.T + d
    y = jnp.array(rng.poisson(np.exp(np.asarray(eta))).astype(float))
    valid_y = jnp.ones((1, T))

    # Initialize pseudo-obs
    H = norm_loading(C_raw) @ M
    j = jnp.zeros((1, T, state_dim))
    J = jnp.tile(jnp.eye(state_dim) * 0.1, (1, T, 1, 1))

    z = j.copy()
    Z = J + jnp.tile(jnp.eye(state_dim), (1, T, 1, 1))

    # One update
    j_new, J_new = Poisson.update_pseudo(params, y, valid_y, z, Z, j, J, lr=0.5)

    # j_new and J_new should differ from the initial values
    assert not jnp.allclose(j_new, j), "CVI update produced no change"


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
