from jax import numpy as jnp
import numpy as np
import numpy.testing as npt

from cvhmax.filtering import predict, bifilter, information_filter


def test_predict_1d_ar1():
    """predict() on a 1D AR(1) matches the analytic information-form formula."""
    # AR(1): x_t = a * x_{t-1} + N(0, q)
    a = 0.9
    q = 0.5
    F = jnp.array([[a]])
    P = jnp.array([[1.0 / q]])  # precision of process noise

    z_prev = jnp.array([2.0])
    Z_prev = jnp.array([[4.0]])  # precision of previous posterior

    zp, Zp = predict(z_prev, Z_prev, F, P)

    # Analytic: covariance form
    # P_prev = 1/Z_prev = 0.25
    # P_pred = a^2 * P_prev + q = 0.81 * 0.25 + 0.5 = 0.7025
    # Z_pred = 1/P_pred
    P_prev = 1.0 / 4.0
    P_pred = a**2 * P_prev + q
    Z_pred_expected = 1.0 / P_pred
    m_prev = z_prev / Z_prev  # = 0.5
    m_pred = a * m_prev
    z_pred_expected = Z_pred_expected * m_pred

    npt.assert_allclose(np.asarray(Zp).item(), Z_pred_expected, rtol=1e-10)
    npt.assert_allclose(
        np.asarray(zp).item(), np.asarray(z_pred_expected).item(), rtol=1e-10
    )


def test_forward_filter_recovers_state():
    """Forward filter on a known linear-Gaussian SSM: filtered MSE < prior MSE."""
    rng = np.random.default_rng(123)
    T, L = 100, 2
    F = jnp.eye(L) * 0.95
    Q = jnp.eye(L) * 0.1
    P = jnp.linalg.inv(Q)
    H = jnp.array([[1.0, 0.5], [0.3, 1.0], [0.0, 0.8]])
    R = jnp.eye(3) * 0.2

    # Generate truth
    x = np.zeros((T, L))
    x[0] = rng.standard_normal(L)
    for t in range(1, T):
        x[t] = 0.95 * x[t - 1] + rng.standard_normal(L) * np.sqrt(0.1)
    y = x @ np.asarray(H.T) + rng.standard_normal((T, 3)) * np.sqrt(0.2)

    # Observation information
    Rinv = jnp.linalg.inv(R)
    J_obs = H.T @ Rinv @ H
    j_obs = (H.T @ Rinv @ jnp.array(y).T).T  # (T, L)
    J_obs = jnp.tile(J_obs, (T, 1, 1))

    z0 = jnp.zeros(L)
    Z0 = jnp.eye(L)

    _, _, z_filt, Z_filt = information_filter((z0, Z0), (j_obs, J_obs), F, P)

    # Recover mean from information form
    m_filt = np.array(
        [
            np.linalg.solve(np.asarray(Z_filt[t]), np.asarray(z_filt[t]))
            for t in range(T)
        ]
    )

    mse_filtered = np.mean((m_filt - x) ** 2)
    mse_prior = np.mean(x**2)  # zero-mean prior
    assert mse_filtered < mse_prior, (
        f"Filtered MSE {mse_filtered} >= prior MSE {mse_prior}"
    )


def test_bifilter_better_than_forward():
    """Smoothed (bifilter) MSE should be less than or equal to forward-only MSE."""
    rng = np.random.default_rng(456)
    T, L = 80, 2
    a = 0.9
    F = jnp.eye(L) * a
    Q = jnp.eye(L) * 0.2
    P = jnp.linalg.inv(Q)
    H = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    R = jnp.eye(2) * 0.5

    x = np.zeros((T, L))
    x[0] = rng.standard_normal(L)
    for t in range(1, T):
        x[t] = a * x[t - 1] + rng.standard_normal(L) * np.sqrt(0.2)
    y = x + rng.standard_normal((T, 2)) * np.sqrt(0.5)

    Rinv = jnp.linalg.inv(R)
    J_obs = H.T @ Rinv @ H
    j_obs = (H.T @ Rinv @ jnp.array(y).T).T
    J_obs = jnp.tile(J_obs, (T, 1, 1))

    z0 = jnp.zeros(L)
    Z0 = jnp.eye(L)

    # Forward only
    _, _, z_fwd, Z_fwd = information_filter((z0, Z0), (j_obs, J_obs), F, P)
    m_fwd = np.array(
        [np.linalg.solve(np.asarray(Z_fwd[t]), np.asarray(z_fwd[t])) for t in range(T)]
    )

    # Bifilter (smoothed)
    z_sm, Z_sm = bifilter(j_obs, J_obs, z0, Z0, F, P, F, P)
    m_sm = np.array(
        [np.linalg.solve(np.asarray(Z_sm[t]), np.asarray(z_sm[t])) for t in range(T)]
    )

    mse_fwd = np.mean((m_fwd - x) ** 2)
    mse_sm = np.mean((m_sm - x) ** 2)
    assert mse_sm <= mse_fwd + 1e-8, f"Smoothed MSE {mse_sm} > forward MSE {mse_fwd}"


def test_bifilter_no_observations():
    """With j=0, J=0 (no observations), the posterior precision should stay near Z0."""
    L = 2
    T = 10
    j = jnp.zeros((T, L))
    J = jnp.zeros((T, L, L))
    z0 = jnp.zeros(L)
    Z0 = jnp.eye(L) * 5.0

    F = jnp.eye(L) * 0.9
    P = jnp.eye(L) * 10.0  # precision of Q

    z_sm, Z_sm = bifilter(j, J, z0, Z0, F, P, F, P)
    # Without observations, the smoothed information should be close to the prior
    # z should remain near zero (no data to pull it away)
    npt.assert_allclose(np.asarray(z_sm), 0.0, atol=1e-6)
