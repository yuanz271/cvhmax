from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from cvhmax import filtering


def test_information_filter():    
    y_ndim = 5
    x_ndim = 2
    n = 100
    omega = 10.

    H = np.random.randn(y_ndim, x_ndim)
    R = np.eye(y_ndim)

    F = np.random.randn(x_ndim, x_ndim)
    Q = np.eye(x_ndim) / omega ** 2
    Qinv = np.eye(x_ndim) * omega ** 2
    
    x = np.reshape(np.sin(np.arange(x_ndim * n) / omega), (x_ndim, n))
    y = H @ x + np.random.randn(y_ndim, n)

    I = H.T @ np.linalg.solve(R, H)
    # I = np.tile(I, (n, 1, 1))
    # assert I.shape == (n, x_ndim, x_ndim)

    i = H.T @ np.linalg.solve(R, y)
    assert i.shape == (x_ndim, n)
    i = i.T
    
    z = np.random.randn(x_ndim)
    Z = Qinv
    
    T = i.shape[0]
    I = jnp.tile(I, (T, 1, 1))

    _, state = filtering.information_filter_step((z, Z), (i[0], I[0]), F, Qinv)
    zp, Zp, z, Z = filtering.information_filter((z, Z), (i, I), F, Qinv)
    
    assert z.shape == (n, x_ndim)
    assert Z.shape == (n, x_ndim, x_ndim)

    fig, ax = plt.subplots()
    ax.plot(z)
    fig.savefig('information_filter.pdf')
    plt.close(fig)
