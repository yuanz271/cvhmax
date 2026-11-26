"""A new observation model is a self-contained addition.

Extensibility is a testable property, not an intention: the readout below is
defined here and supports every loading kind without editing ``src/``. It also
documents the contract a new likelihood has to satisfy.
"""

from typing import override

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from cvhmax import CVHM
from cvhmax.cvi import CVI
from cvhmax.hm import HidaMatern

from .helpers import sample_gp_latents
from cvhmax.readout import ard_fit_laplace, loading_settings
from cvhmax.readout import Params


def _binomial_nll(eta, y):
    """Negative Bernoulli log-likelihood with a logit link, summed over steps."""
    return jnp.sum(jax.nn.softplus(eta) - y * eta)


class Bernoulli(CVI):
    """Bernoulli readout, used here to exercise the extension path.

    The expected log-likelihood is approximated at the latent mean. A
    production readout would justify its own treatment of the latent variance;
    this one keeps the gate test focused on the extension path.
    """

    supported_loadings = frozenset({"ridge", "ard", "fixed"})

    @classmethod
    def _step_nll(cls, params, m_t, V_t, y_t, mask_t):
        # ``readout_objective`` receives the raw ``(C, d)`` tuple, so this
        # helper works on tuples rather than on ``Params``.
        C, d = params
        eta = C @ m_t + d
        return jnp.sum(jnp.where(mask_t, jax.nn.softplus(eta) - y_t * eta, 0.0))

    @classmethod
    @override
    def readout_objective(cls, params, y, valid_y, m, V, penalty):
        steps = jax.vmap(cls._step_nll, in_axes=(None, 0, 0, 0, 0))(params, m, V, y, valid_y)
        C = params[0]
        return (jnp.sum(steps) + 0.5 * jnp.sum(penalty[1:][None, :] * C**2)) / jnp.sum(
            jnp.any(valid_y, axis=-1)
        )

    @classmethod
    def _natural_stats(cls, params, m, V, y, valid_y):
        """CVI natural parameters per time step, from the local curvature."""
        lead, n_latent = m.shape[:-1], m.shape[-1]
        m = m.reshape(-1, n_latent)
        V = V.reshape(-1, *V.shape[-2:])
        y = y.reshape(-1, y.shape[-1])
        valid_y = valid_y.reshape(y.shape)

        def one(m_t, V_t, y_t, mask_t):
            readout = (params.C, params.d)

            def step(mu):
                return cls._step_nll(readout, mu, V_t, y_t, mask_t)

            gradient = jax.grad(step)(m_t)
            curvature = jax.hessian(step)(m_t)
            return gradient + curvature @ m_t, curvature

        natural, curvature = jax.vmap(one, in_axes=0)(m, V, y, valid_y)
        return natural.reshape(*lead, n_latent), curvature.reshape(*lead, n_latent, n_latent)

    @classmethod
    @override
    def initialize_info(cls, params, y, valid_y):
        K = params.C.shape[1]
        zeros = jnp.zeros((y.shape[0], K))
        eye = jnp.broadcast_to(jnp.eye(K), (y.shape[0], K, K))
        return cls._natural_stats(params, zeros, eye, y, valid_y)

    @classmethod
    @override
    def update_pseudo(cls, params, y, valid_y, m, V, j, J, lr):
        new_j, new_J = cls._natural_stats(params, m, V, y, valid_y)
        return (1.0 - lr) * j + lr * new_j, (1.0 - lr) * J + lr * new_J

    @classmethod
    @override
    def initialize_params(cls, y, valid_y, n_factors, *, random_state, penalty=None, params=None):
        if params is not None:
            return params
        n_features = y.shape[-1]
        rate = (jnp.sum(y, axis=(0, 1)) + 0.5) / (jnp.sum(valid_y, axis=(0, 1)) + 1.0)
        # Non-zero loadings matter: with ``C = 0`` the E-step returns zero
        # latents, so the loading gradient vanishes and EM cannot leave the
        # corner. A real readout starts from factor analysis for the same
        # reason.
        loadings = 0.1 * jax.random.normal(
            jax.random.PRNGKey(random_state), (n_features, n_factors)
        )
        return Params(
            C=loadings,
            d=jnp.log(rate) - jnp.log1p(-rate),
            R=jnp.asarray(0.0),
        )

    @classmethod
    @override
    def solve_ard(cls, settings, params, y, valid_y, m, V):
        """Only ARD needs readout-specific code; ridge and fixed come for free."""
        return ard_fit_laplace(
            cls.readout_objective, params, y, valid_y, m, V,
            alpha_shape=settings["alpha_shape"],
            alpha_rate=settings["alpha_rate"],
            iterations=settings["iterations"],
        )


def _data(seed=0):
    rng = np.random.default_rng(seed)
    n_sequences, n_time, n_features, n_latents, dt = 2, 120, 12, 2, 0.1
    kernel = HidaMatern(sigma=1.0, rho=1.0, omega=0.0, order=1, s=1e-5)
    latent = sample_gp_latents(kernel, n_time, dt, n_latents, rng, n_sequences)
    loadings = rng.normal(size=(n_features, n_latents))
    loadings = 0.8 * loadings / np.linalg.norm(loadings, axis=1, keepdims=True)
    probability = 1.0 / (1.0 + np.exp(-0.8 * latent @ loadings.T))
    y = jnp.asarray(rng.binomial(1, probability).astype(float))
    return kernel, y, n_latents


def _fit(kernel, y, loading, **config):
    return CVHM(
        n_components=4,
        dt=0.1,
        kernels=[kernel] * 4,
        observation="Bernoulli",
        loading=loading,
        loading_config=config or None,
        max_iter=60,
        min_iter=5,
        tol=1e-3,
        convergence_patience=3,
    ).fit(y, random_state=0)


def test_new_readout_is_registered_by_name():
    assert CVI.registry["Bernoulli"] is Bernoulli
    assert Bernoulli.supported_loadings == frozenset({"ridge", "ard", "fixed"})


@pytest.mark.parametrize("loading", ["ridge", "fixed"])
def test_new_readout_supports_generic_loading_kinds(loading):
    kernel, y, _ = _data()
    model = _fit(kernel, y, loading)
    assert np.isfinite(np.asarray(model.params.C)).all()
    assert np.isfinite(np.asarray(model.posterior[0])).all()
    assert np.asarray(model.params.alpha).shape == (4,)


def test_new_readout_supports_ard_without_touching_the_package():
    """ARD for a user readout needs one override, not an estimator class.

    This asserts that the extension path works, not that a hand-written
    Bernoulli ARD ranks latents reliably: like Poisson, the loading scale and
    the latent scale trade off, so relevance on count data is a modelling
    question rather than a plumbing one.
    """
    kernel, y, _ = _data()
    model = _fit(kernel, y, "ard", iterations=5)
    alpha = np.asarray(model.params.alpha)
    assert alpha.shape == (4,)
    assert np.isfinite(alpha).all() and np.all(alpha > 0)
    assert np.isfinite(np.asarray(model.params.C)).all()
    assert np.isfinite(np.asarray(model.posterior[0])).all()
    assert np.isfinite(np.asarray(model.posterior[1])).all()


def test_new_readout_rejects_an_unknown_loading_kind():
    class RidgeOnly(CVI):
        supported_loadings = frozenset({"ridge"})

    try:
        with pytest.raises(ValueError, match="not supported by observation"):
            CVHM(1, 1.0, [HidaMatern(1.0, 1.0, 0.0, 0)], observation="RidgeOnly", loading="ard")
    finally:
        CVI.registry.pop("RidgeOnly", None)


def test_new_readout_inherits_the_shared_penalty_convention():
    """One penalty definition, so a new readout cannot invent its own bias rule."""
    settings = loading_settings("ridge", {"alpha": 0.3})
    params = Params(
        C=jnp.asarray([[1.0], [2.0]]), d=jnp.asarray([5.0, 5.0]), R=jnp.asarray(0.0)
    )
    y = jnp.asarray([[1.0], [0.0]])
    objective = Bernoulli.readout_objective(
        (params.C, params.d), y, jnp.ones_like(y, dtype=bool),
        jnp.zeros((2, 1)), jnp.zeros((2, 1, 1)), settings["alpha"] * jnp.ones(2),
    )
    # The bias is unpenalised: raising both biases by 10 shifts the objective only
    # through the data term, never through the penalty.
    shifted = Bernoulli.readout_objective(
        (params.C, params.d + 10.0), y, jnp.ones_like(y, dtype=bool),
        jnp.zeros((2, 1)), jnp.zeros((2, 1, 1)), settings["alpha"] * jnp.ones(2),
    )
    assert not np.allclose(np.asarray(objective), np.asarray(shifted))
