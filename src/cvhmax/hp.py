from functools import partial
import operator

import jax
from jax import nn
import jax.numpy as jnp
from jax.scipy import optimize
import equinox as eqx

from .hm import spectral_density


bound = nn.softplus  # ln(1 + e^x)


def unbound(x):
    """Map a softplus-bounded parameter back to the real line.

    Parameters
    ----------
    x : Array
        Positive parameter produced by `bound`.

    Returns
    -------
    Array
        Real-valued unconstrained representation.
    """
    return jnp.log(jnp.expm1(x))  # ln(e^x - 1)


def spec2vec(spec, filter):
    """Flatten a nested spectral specification into a trainable vector.

    Parameters
    ----------
    spec : PyTree
        Spectral hyperparameters organised per latent component.
    filter : PyTree
        Boolean mask indicating which entries are trainable.

    Returns
    -------
    tuple[Array, Any, PyTree]
        Flat parameter array, tree definition, and static structure.
    """
    # https://docs.kidger.site/equinox/all-of-equinox/
    # spec = [[{'sigma': 1., 'rho': 1., 'omega': 0., 'order': 1}], [{'sigma': 1., 'rho': 1., 'omega': 0., 'order': 0}, {'sigma': 1., 'rho': 1., 'omega': 1., 'order': 1}]]
    # filter = [[{'sigma': True, 'rho': True, 'omega': True, 'order': False}], [{'sigma': True, 'rho': True, 'omega': True, 'order': False}, {'sigma': True, 'rho': True, 'omega': True, 'order': False}]]

    params, static = eqx.partition(spec, filter)
    paramflat, paramdef = jax.tree_util.tree_flatten(params)
    paramflat = jnp.asarray(
        paramflat
    )  # NOTE: paramflat will zero-dim array entering and leaving JAX
    # TODO: maybe convert the trainable scalar into 0-dim JAX array before any calculation
    return paramflat, paramdef, static


def vec2spec(paramflat, paramdef, static):
    """Reconstruct the spectral specification PyTree from a flat vector.

    Parameters
    ----------
    paramflat : Array
        Flat parameter vector.
    paramdef : Any
        Tree definition returned by :func:`spec2vec`.
    static : PyTree
        Non-trainable structure paired with the trainable entries.

    Returns
    -------
    PyTree
        Spectral hyperparameters restored to their nested layout.
    """
    # paramflat = paramflat.tolist()
    params_tree = jax.tree_util.tree_unflatten(paramdef, paramflat)
    spec = eqx.combine(params_tree, static)
    return spec


def spectral_loss(paramflat, paramdef, static, spectral_density, m, V, dt, clip=1e-5):
    """Negative Whittle objective evaluated in the frequency domain.

    Parameters
    ----------
    paramflat : Array
        Flat vector of (potentially unbounded) spectral parameters.
    paramdef : Any
        Tree definition returned by :func:`spec2vec`.
    static : PyTree
        Static portion of the spectral specification.
    spectral_density : Callable
        Function mapping kernel specs to their power spectral density.
    m : Array
        Posterior means with shape `(time, latent_dim)`.
    V : Array
        Posterior covariances with shape `(time, latent_dim, latent_dim)`.
    dt : float
        Discretisation step of the time axis.
    clip : float, default=1e-5
        Floor applied to spectra and periodogram terms for stability.

    Returns
    -------
    float
        Aggregated Whittle loss over all latent components.
    """

    def kernel_loss(kernel_spec, freq):
        Sw = spectral_density(kernel_spec, freq)
        Sw = jnp.clip(Sw, min=clip)
        Sw = jnp.expand_dims(Sw, 1)

        return -0.5 * jnp.sum(jnp.log(Sw) + Zw / Sw)

    paramflat = bound(paramflat)  # input is unbounded
    latent_spec = vec2spec(paramflat, paramdef, static)

    T = m.shape[0]
    s = jnp.sqrt(jnp.diagonal(V, axis1=1, axis2=2))  # std
    w = jnp.hanning(T)  # for leakage
    w = jnp.expand_dims(w, 1)

    assert w.shape == m.shape

    m_w = jnp.fft.rfft(w * m, axis=0, norm="ortho")  # real input DFT
    s_w = jnp.fft.rfft(w * s, axis=0, norm="ortho")

    # 8 / 3 factor for hann window
    Zw = jnp.abs(m_w) ** 2 + jnp.abs(s_w) ** 2
    Zw = 2 * dt * Zw
    Zw = (8 / 3) * Zw

    Zw = jnp.clip(Zw, min=clip)

    # l2_sums = self._get_hyperparameter_l2_sum()

    freq = jnp.fft.rfftfreq(T, d=dt)  # DFT sample frequencies

    log_p_s = jax.tree.map(
        partial(kernel_loss, freq=freq),
        latent_spec,
        is_leaf=lambda x: isinstance(x, dict),
    )

    log_p_s = jax.tree_util.tree_reduce(operator.add, log_p_s)
    # log_p_s = sum(sum(log_p_s))

    # TODO: add L2 regularization
    # log_p_s += -0.5 * (1 / B) * l2_sums['b']
    # log_p_s += -0.5 * (1 / LS) * l2_sums['ls']
    # log_p_s += -0.5 * (1 / SIGMA) * l2_sums['sigma']

    return log_p_s


def whittle(latent_spec, filter, m, V, dt, clip=1e-5):
    """Fit spectral hyperparameters by minimising the Whittle loss.

    Parameters
    ----------
    latent_spec : PyTree
        Initial spectral specification. Each latent component is represented
        as a list of kernel dictionaries (e.g., `{"sigma": ..., "rho": ...}`).
    filter : PyTree
        Boolean mask indicating trainable entries.
    m : Array
        Posterior means with shape `(time, latent_dim)`.
    V : Array
        Posterior covariances with shape `(time, latent_dim, latent_dim)`.
    dt : float
        Discretisation step of the time axis.
    clip : float, default=1e-5
        Floor applied to spectra and periodogram terms for stability.

    Returns
    -------
    PyTree
        Optimised spectral specification.

    Examples
    --------
    >>> latent_spec = [[{"sigma": 1.0, "rho": 1.0, "omega": 0.0, "order": 0}]]
    >>> trainable = [[{"sigma": True, "rho": True, "omega": True, "order": False}]]
    >>> spec = whittle(latent_spec, trainable, m, V, dt)
    """
    # flatten hyperparams
    paramflat, paramdef, static = spec2vec(latent_spec, filter)
    # unbound hyperparams
    paramflat = unbound(paramflat)

    opt = optimize.minimize(
        spectral_loss,
        x0=paramflat,
        args=(paramdef, static, spectral_density, m, V, dt, clip),
        method="BFGS",
    )
    paramflat = opt.x
    # bound hyperparams
    paramflat = bound(paramflat)
    # unflatten
    latent_spec = vec2spec(paramflat, paramdef, static)

    return latent_spec
