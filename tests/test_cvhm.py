import numpy as np
import jax
from jax import numpy as jnp, config

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
    y = jnp.expand_dims(y, 0)

    with capsys.disabled():
        model = CVHM(n_factors, dt, kernels, max_iter=20, likelihood="Poisson")
        result = model.fit(y)
    m, V = result.posterior
    m = m[0]
    V = V[0]

    assert m.shape == (T, n_factors)
    assert V.shape == (T, n_factors, n_factors)


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
                dict(description=description, total=total, fields=fields)
            )
            return 123

        def update(self, task_id, **kwargs):
            self.update_calls.append((task_id, kwargs))

    fake_pbar = FakeProgress()

    import cvhmax.cvhm as cvhm_mod
    from cvhmax.cvi import Params, Poisson

    monkeypatch.setattr(cvhm_mod, "training_progress", lambda: fake_pbar)

    def fake_initialize_params(
        cls, y, ymask, n_factors, lmask, *, random_state
    ) -> Params:
        obs_dim = y.shape[-1]
        C = jnp.zeros((obs_dim, n_factors), dtype=y.dtype)
        d = jnp.zeros(obs_dim, dtype=y.dtype)
        R = jnp.zeros((obs_dim, obs_dim), dtype=y.dtype)
        return Params(C=C, d=d, R=R, M=lmask)

    def fake_update_readout(cls, params, y, ymask, m, V) -> tuple[Params, float]:
        return params, jnp.nan

    monkeypatch.setattr(
        Poisson, "initialize_params", classmethod(fake_initialize_params)
    )
    monkeypatch.setattr(Poisson, "update_readout", classmethod(fake_update_readout))

    recorded = []
    import jax.debug as jdebug

    def record_debug_callback(
        callback, *args, ordered=False, partitioned=False, **kwargs
    ):
        recorded.append(
            dict(
                callback=callback,
                args=args,
                ordered=ordered,
                partitioned=partitioned,
                kwargs=kwargs,
            )
        )
        return None

    monkeypatch.setattr(jdebug, "callback", record_debug_callback)

    n_devices = len(jax.devices())
    trials = max(1, n_devices)
    T = 4
    obs_dim = 3
    n_components = 1

    y = jnp.zeros((trials, T, obs_dim), dtype=jnp.float64)
    ymask = jnp.ones((trials, T), dtype=jnp.uint8)

    kernels = [HidaMatern(1.0, 1.0, 0.0, 0)]
    model = CVHM(
        n_components=n_components,
        dt=1.0,
        kernels=kernels,
        likelihood="Poisson",
        max_iter=3,
    )

    model.fit(y, ymask=ymask, random_state=0)

    assert any(call["ordered"] is True for call in recorded), recorded

    cb_calls = [call for call in recorded if len(call["args"]) == 2]
    assert cb_calls, recorded

    cb = cb_calls[0]["callback"]
    cb(0, 1.23)
    cb(2, 4.56)

    assert fake_pbar.add_task_calls, "Expected add_task() to be called"
    assert fake_pbar.update_calls, "Expected update() to be called"

    for _, kwargs in fake_pbar.update_calls:
        assert "completed" in kwargs
        assert "advance" not in kwargs
        assert 1 <= kwargs["completed"] <= 3
        assert isinstance(kwargs.get("nell"), float)
