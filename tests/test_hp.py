import numpy as np
import numpy.testing as npt
from jax import config
from jax import numpy as jnp

from cvhmax import hm, hp
from cvhmax.hm import sample_matern
from cvhmax.hp import bound, unbound

config.update("jax_enable_x64", True)


def test_spectral_loss():
    n = 100
    dt = 1.0
    sigma = 1.0
    rho = 0.5
    m = sample_matern(n, dt, sigma, rho)
    m = np.expand_dims(m, 1)
    v = np.ones((n, 1, 1))
    # print(m.shape, v.shape)
    spec = [
        [{"sigma": 1.0, "rho": 1.0, "omega": 0.0, "order": 1}],
        [
            {"sigma": 1.0, "rho": 1.0, "omega": 0.0, "order": 0},
            {"sigma": 1.0, "rho": 1.0, "omega": 1.0, "order": 1},
        ],
    ]
    filter = [
        [{"sigma": True, "rho": True, "omega": True, "order": False}],
        [
            {"sigma": True, "rho": True, "omega": True, "order": False},
            {"sigma": True, "rho": True, "omega": True, "order": False},
        ],
    ]

    paramflat, paramdef, static = hp.spec2vec(spec, filter)
    log_p_s = hp.spectral_loss(
        paramflat, paramdef, static, hm.spectral_density, m, v, dt
    )
    print(log_p_s)

    arr = jnp.array([jnp.array(1.0), jnp.array(2.0)])
    print(arr, arr.shape)
    new_spec = hp.whittle(spec, filter, m, v, dt)
    print(new_spec)


# def test_whittle():
#     n = 100
#     dt = 1.
#     sigma = 1.
#     rho = .5
#     m = sample_matern(n, dt, sigma, rho)
#     m = np.expand_dims(m, 1)
#     v = np.ones((n, 1, 1))
#     print(m.shape, v.shape)
#     hyperparams = np.zeros(3)
#     sigma, rho, b = hp.whittle(hyperparams, m, v, dt)
#     print(sigma, rho, b)


# ---------------------------------------------------------------------------
# New tests
# ---------------------------------------------------------------------------


def test_bound_unbound_roundtrip():
    """softplus(inv_softplus(x)) should recover x for positive values."""
    for x in [0.01, 0.1, 1.0, 10.0, 100.0]:
        x_jnp = jnp.array(x)
        recovered = bound(unbound(x_jnp))
        npt.assert_allclose(
            float(recovered), x, rtol=1e-10, err_msg=f"Roundtrip failed for x={x}"
        )

    # Also test the reverse direction
    for u in [-5.0, -1.0, 0.0, 1.0, 5.0]:
        u_jnp = jnp.array(u)
        recovered = unbound(bound(u_jnp))
        npt.assert_allclose(
            float(recovered),
            u,
            atol=1e-10,
            err_msg=f"Reverse roundtrip failed for u={u}",
        )


def test_spectral_loss_finite():
    """spectral_loss returns a finite value (not NaN/Inf)."""
    n = 200
    dt = 1.0
    m = sample_matern(n, dt, 1.0, 1.0)
    m = np.expand_dims(m, 1)
    v = np.ones((n, 1, 1))

    spec = [[{"sigma": 1.0, "rho": 1.0, "omega": 0.0, "order": 0}]]
    filt = [[{"sigma": True, "rho": True, "omega": True, "order": False}]]

    paramflat, paramdef, static = hp.spec2vec(spec, filt)
    loss = hp.spectral_loss(paramflat, paramdef, static, hm.spectral_density, m, v, dt)
    assert jnp.isfinite(loss), f"spectral_loss returned {loss}"


def test_whittle_improves_loss():
    """whittle should reduce the spectral loss relative to the initial parameters."""
    sigma_true = 1.0
    rho_true = 1.0
    dt = 1.0
    n = 2000

    m = sample_matern(n, dt, sigma_true, rho_true)
    m = np.expand_dims(m, 1)
    v = np.ones((n, 1, 1)) * 0.01

    spec = [[{"sigma": 2.0, "rho": 0.1, "omega": 0.0, "order": 0}]]
    filt = [[{"sigma": True, "rho": True, "omega": False, "order": False}]]

    # Compute initial loss
    paramflat_init, paramdef, static = hp.spec2vec(spec, filt)
    loss_init = hp.spectral_loss(
        paramflat_init, paramdef, static, hm.spectral_density, m, v, dt
    )

    # Run whittle
    new_spec = hp.whittle(spec, filt, m, v, dt)

    # Compute loss at fitted params
    paramflat_fit, paramdef_fit, static_fit = hp.spec2vec(new_spec, filt)
    loss_fit = hp.spectral_loss(
        paramflat_fit, paramdef_fit, static_fit, hm.spectral_density, m, v, dt
    )

    assert jnp.isfinite(loss_fit), f"Fitted loss is not finite: {loss_fit}"
    assert loss_fit <= loss_init, (
        f"whittle did not improve loss: init={float(loss_init):.3f}, fit={float(loss_fit):.3f}"
    )
