import pytest
import numpy as np
import matplotlib.pyplot as plt
from jax import numpy as jnp

from cvhmax.cvhm import CVHM
from cvhmax.cvi import Params
from cvhmax.hm import HidaMatern
from cvhmax.hm import sample_matern


def test_CVHM(capsys):
    np.random.seed(1234)
    T = 2000
    n_obs = 20
    n_factors = 2
    dt = 1.
    sigma = 1.
    rho = 50.
    
    kernels = [HidaMatern(sigma, rho, 0., 0) for k in range(n_factors)]
    params = Params()

    x = np.column_stack([sample_matern(T, dt, sigma, rho), sample_matern(T, dt, sigma, rho)])

    # x = sample_matern np.sin(np.arange(2 * T) / 100).reshape(T, -1)
    C = params.C = np.random.randn(n_obs, n_factors)
    d = params.d = np.random.randn(n_obs) - 2
    params.R = np.eye(n_obs) * 2

    y = np.random.poisson(np.exp(x @ C.T + np.expand_dims(d, 0)))

    model = CVHM(n_factors, dt, kernels, params, max_iter=2, likelihood='Poisson')
    
    with capsys.disabled():
        result = model.fit(y)
    m, V = result.posterior
    m = m[0]
    V = V[0]

    assert m.shape == (T, n_factors)
    assert V.shape == (T, n_factors, n_factors)

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(m[:, 0], label="Inference", alpha=0.5)
    ax[0].plot(x[:, 0], label="Ground truth", alpha=0.5)

    ax[1].plot(m[:, 1], alpha=0.5)
    ax[1].plot(x[:, 1], alpha=0.5)
    ax[0].legend()
    fig.savefig('cvhm.pdf')
    plt.close(fig)
