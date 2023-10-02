import numpy as np
import matplotlib.pyplot as plt

from cvhmax.cvhm import CVHM
from cvhmax.cvi import Params
from cvhmax.hm import HidaMatern


def test_CVHM():
    np.random.seed(1234)
    T = 2000
    n_obs = 20
    n_factors = 2
    dt = 1.
    
    kernels = [HidaMatern(.05, .05, 0., 0) for k in range(n_factors)]
    params = Params()

    x = np.sin(np.arange(2 * T) / 100).reshape(T, -1)
    C = params.C = np.random.randn(n_obs, n_factors)
    d = params.d = np.random.randn(n_obs, 1)
    params.R = np.eye(n_obs) * 2

    y = x @ C.T + d.T + np.random.randn(T, n_obs) * 2

    model = CVHM(n_factors, dt, kernels, params)
    Af = model.Af()
    Qf = model.Qf()
    Ab = model.Ab()
    Qb = model.Qb()
    Q0 = model.Q0()
    M = model.mask()

    result = model.fit(y, max_em_iter=1)
    m, V = result.components_
    m = m[0]
    V = V[0]

    assert m.shape == (T, n_factors)
    assert V.shape == (T, n_factors, n_factors)

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(m[:, 0], label="Inference")
    ax[0].plot(x[:, 0], label="Ground truth")

    ax[1].plot(m[:, 1])
    ax[1].plot(x[:, 1])
    ax[0].legend()
    fig.savefig('cvhm.pdf')
    plt.close(fig)
