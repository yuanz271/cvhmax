from abc import abstractmethod
from dataclasses import dataclass, field

from jax import numpy as jnp, random as jrandom
from jax.scipy.optimize import minimize
from jaxtyping import Array

from .utils import info_repr, norm_loading


@dataclass
class Params:
    C: Array = field(init=False)
    d: Array = field(init=False)
    R: Array = field(init=False)
    M: Array = field(init=False)
    
    def initialize(self, n_obs, n_factors, *, random_state):
        key = jrandom.PRNGKey(random_state)
        Ckey, dkey = jrandom.split(key)
        self.C = jrandom.normal(Ckey, shape=(n_obs, n_factors)) / n_obs
        self.d = jrandom.normal(dkey, shape=(n_obs, 1))
        self.R = jnp.eye(n_obs)

    def nC(self) -> Array:
        return norm_loading(self.C)


class CVI:
    @abstractmethod
    def fit(self, y, m, V):
        pass

    @abstractmethod
    def pseudo_observation(self, y):
        pass


def observation_estimate(y, m, V, lam=0.1):
    """
    OLS
    w = (z'z)^-1 z'y
    """
    y = jnp.vstack(y)
    m = jnp.vstack(m)

    T, y_dim = y.shape
    _, z_dim = m.shape

    m1 = jnp.column_stack([jnp.ones((T, 1)), m])

    assert m1.shape == (T, z_dim + 1)

    zy = m1.T @ y  # (z + 1, t) (t, y) -> (z + 1, y)
    zz = m1.T @ m1  # (z + 1, t) (t, z + 1) -> (z + 1, z + 1)
    eye = jnp.eye(zz.shape[0])
    w = jnp.linalg.solve(zz + lam * eye, zy)  # (z + 1, z + 1) (z + 1, y) -> (z + 1, y)
    
    r = y - m1 @ w  # (t, y)
    R = r.T @ r / T  # (y, y)

    d, C = jnp.split(w, [1], axis=0)  # (1, y), (z, y)
    
    d = d.T
    C = C.T
    assert d.shape == (y_dim, 1)
    assert C.shape == (y_dim, z_dim)

    return C, d, R


class Gaussian(CVI):
    def __init__(self) -> None:
        super().__init__()

    def pseudo_observation(self, y):
        C = self.params.C
        d = self.params.d
        R = self.params.R
        M = self.params.M

        H = C @ M

        return info_repr(y, H, d, R)
    
    def fit(self, y, m, P):
        C, d, R = observation_estimate(y, m, P)
        self.params.C = C
        self.params.d = d
        self.params.R = R


def v2w(v, shape):
    w = v.reshape(shape)
    return jnp.split(w, [1], axis=1)


def nell(w, shape, y, m, V):
    d, C = v2w(w)

    eta = m @ C.T + d.T
    quad = C @ jnp.linalg.solve(V, C.T)  # (Y, Z) (T, Z, Z) (Z, Y) -> (T, Y, Y)
    quad = jnp.diagonal(quad, axis1=1, axis2=2)  # (T, Y)
    lam = jnp.exp(eta + .5 * quad)
    return jnp.mean(jnp.sum(lam - eta * y, axis=-1))


class Poisson(CVI):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, y, m, V):
        C = self.params.C
        d = self.params.d
        w = jnp.column_stack([d, C])
        shape = w.shape
        w = w.flatten()

        opt = minimize(nell, w, args=(shape, y, m, V))
        w_opt = opt.x

        d, C = v2w(w_opt)

        self.params.C = C
        self.params.d = d
    
    def pseudo_observation(self, j, J, y, m, V, lr):
        C = self.params.C
        d = self.params.d
        eta = m @ C.T + d.T
        quad = C @ jnp.linalg.solve(V, C.T)  # (Y, Z) (T, Z, Z) (Z, Y) -> (T, Y, Y)
        quad = jnp.diagonal(quad, axis1=1, axis2=2)  # (T, Y)
        lam = jnp.exp(eta + .5 * quad)
        g_m = (y - lam) @ C
        g_V = -.5 * jnp.einsum('ni, tn, nj -> tij', C, lam, C)

        dJ = -2 * g_V
        dj = g_m - 2 * g_V @ jnp.expand_dims(m, axis=-1)

        j = (1 - lr) * j + j * dj
        J = (1 - lr) * J + j * dJ
        
        return j, J
