from jax import numpy as jnp
import numpy as np

from cvhmax import filtering


def test_information_filter():
    y_ndim = 5
    x_ndim = 2
    n = 100
    omega = 10.0

    H = np.random.randn(y_ndim, x_ndim)
    R = np.eye(y_ndim)

    F = np.random.randn(x_ndim, x_ndim)
    # Q = np.eye(x_ndim) / omega**2
    Qinv = np.eye(x_ndim) * omega**2

    x = np.reshape(np.sin(np.arange(x_ndim * n) / omega), (x_ndim, n))
    y = H @ x + np.random.randn(y_ndim, n)

    K = H.T @ np.linalg.solve(R, H)
    # I = np.tile(I, (n, 1, 1))
    # assert I.shape == (n, x_ndim, x_ndim)

    k = H.T @ np.linalg.solve(R, y)
    assert k.shape == (x_ndim, n)
    k = k.T

    z = np.random.randn(x_ndim)
    Z = Qinv

    T = k.shape[0]
    K = jnp.tile(K, (T, 1, 1))

    _, state = filtering.information_filter_step((z, Z), (k[0], K[0]), F, Qinv)
    zp, Zp, z, Z = filtering.information_filter((z, Z), (k, K), F, Qinv)

    assert z.shape == (n, x_ndim)
    assert Z.shape == (n, x_ndim, x_ndim)
