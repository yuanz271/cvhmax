import numpy as np
import matplotlib.pyplot as plt
from jax import numpy as jnp, config
from sklearn.linear_model import LinearRegression

from cvhmax.cvhm import CVHM
from cvhmax.hm import HidaMatern
from cvhmax.hm import sample_matern

config.update("jax_enable_x64", True)


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

    with capsys.disabled():
        model = CVHM(n_factors, dt, kernels, max_iter=2, likelihood="Poisson")
        result = model.fit(y)
    m, V = result.posterior
    m = m[0]
    V = V[0]

    assert m.shape == (T, n_factors)
    assert V.shape == (T, n_factors, n_factors)

    m = LinearRegression().fit(m, x).predict(m)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(x[:, 0], label="Ground truth", alpha=0.5)
    axs[0, 1].plot(m[:, 0], label="Inference", alpha=0.5, color="r")
    axs[1, 0].plot(x[:, 1], label="Ground truth", alpha=0.5)
    axs[1, 1].plot(m[:, 1], label="Inference", alpha=0.5, color="r")
    axs[0, 0].legend()
    axs[0, 1].legend()
    axs[1, 0].legend()
    axs[1, 1].legend()
    fig.savefig("cvhm.pdf")
    plt.close(fig)
