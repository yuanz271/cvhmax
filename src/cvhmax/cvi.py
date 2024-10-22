from abc import abstractmethod
from dataclasses import dataclass, field
from functools import partial

from jax import numpy as jnp, random as jrandom, vmap, lax
from jax.scipy.linalg import cholesky, solve_triangular, cho_factor, cho_solve, inv
from jax.scipy.optimize import minimize
from jaxtyping import Array

from .utils import info_repr, norm_loading


@dataclass
class Params:
    C: Array = field(init=False)
    d: Array = field(init=False)
    R: Array = field(init=False, default=None)
    M: Array = field(init=False)
    
    def initialize(self, n_obs, n_factors, *, random_state):
        key = jrandom.key(random_state)
        Ckey, dkey = jrandom.split(key)
        self.C = jrandom.normal(Ckey, shape=(n_obs, n_factors)) / n_obs
        self.d = jrandom.normal(dkey, shape=(n_obs, 1))
        self.R = jnp.eye(n_obs)

    def nC(self) -> Array:
        return norm_loading(self.C)


class CVI:
    @classmethod
    @abstractmethod    
    def update_readout(cls, *args, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def update_pseudo(cls, *args, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def init_info(cls, params, y, A, Q):
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
    @staticmethod
    def update_pseudo(params, y):
        C = params.C
        d = params.d
        R = params.R
        M = params.M

        H = C @ M

        return info_repr(y, H, d, R)
    
    @staticmethod
    def update_readout(params, y, m, P):
        C, d, R = observation_estimate(y, m, P)
        params = Params(C=C, d=d, R=R, M=params.M)
        return params
    

def v2w(v, shape):
    w = v.reshape(shape)
    return jnp.split(w, [1], axis=1)


def poisson_nell(w, shape, y, m, V):
    d, C = v2w(w, shape)
    
    def _nell(y_t, m_t, V_t):
        eta = C @ m_t @ + d
        quad = jnp.einsum("ni,in->n", C, V_t @ C.T)  # (Y, Z) (T, Z, Z) (Z, Y) -> (T, Y, Y)
        lam = jnp.exp(eta + .5 * quad)
        return jnp.sum(lam - eta * y_t, axis=-1)
    
    return jnp.mean(vmap(_nell)(y, m, V))


def poisson_cvi_stats(z, Z, y, H, d):
    """
    z = V^-1 m
    Z = -0.5 V^-1
    <=>
    m = Vz = -0.5 * Z^-1 z
    V = -0.5Z^-1
    """
    Zcho = cho_factor(Z)
    m = -0.5 * cho_solve(Zcho, z)  # Vj
    eta = m @ H.mT + d
    quad = jnp.einsum("nl, ln -> n", H, -0.5 * cho_solve(Zcho, H.mT))  # CVC'
    lam = jnp.exp(eta + 0.5 * quad)
    grad_m = (y - lam) @ H
    grad_V = -0.5 * jnp.einsum("ni, n, nj -> ij", H, lam, H)

    J_update = -2 * grad_V
    j_update = grad_m - 2 * grad_V @ m
    return j_update, J_update


class Poisson(CVI):
    @classmethod
    def init_info(cls, params, y, A, Q):
        """Initialize pseudo observation
        """
        C = params.C
        M = params.M
        H = C @ M
        d = params.d
        
        z0 = jnp.zeros(Q.shape[0])
        Z0 = Qinv = inv(Q)
        Qcho = cho_factor(Q)
        QiA = cho_solve(Qcho, A)
        AtQiA = A.mT @ QiA
        
        def forward(carry, yt):
            ztm1, Ztm1 = carry
            cho = cholesky(Ztm1 + AtQiA)
            S = solve_triangular(cho, QiA.mT)
            Zp = Qinv - S.mT @ S
            zp = QiA @ cho_solve((cho, False), ztm1)
            Zp = 0.5 * (Zp + Zp.mT)
                        
            j, J = poisson_cvi_stats(zp, Zp, yt, H, d)

            zt = zp + j
            Zt = Zp + J
            Zt = 0.5 * (Zt + Zt.mT)
        
            return (zt, Zt), (j, J)
        
        _, (j, J) = lax.scan(forward, (z0, Z0), y)

        return j, J

    @classmethod
    def update_readout(cls, params, y, m, V):
        C = params.C
        d = params.d
        M = params.M
        R = params.R
        selected_m = m @ M.T  # (T, Z) (Z, sZ) -> (T, sZ)
        selected_V = vmap(lambda v: M @ v @ M.T)(V)  # (T, Z, Z) -> (T, sZ, sZ)

        w = jnp.column_stack([d, C])
        shape = w.shape
        w = w.flatten()

        opt = minimize(poisson_nell, w, args=(shape, y, selected_m, selected_V), method="BFGS")
        w_opt = opt.x

        d, C = v2w(w_opt, shape)
        
        params = Params()
        params.C = C
        params.d = d
        params.R = R
        params.M = M
        return params
    
    @classmethod
    def update_pseudo(cls, params, z, Z, j, J, y, lr):
        """
        :param params: readout
        :param z: 1st posterior natural param of latent
        :param Z: 2nd posterior natural param of latent
        :param j: 1st natural param of posterior pseudo observation
        :param J: 2nd natural param of posterior pseudo observation
        :param y: observation
        :param lr: learning rate
        """
        C = params.C
        M = params.M
        H = C @ M
        d = params.d

        # eta = m @ C.T + d.T
        # quad = C @ jnp.linalg.solve(V, C.T)  # (Y, Z) (T, Z, Z) (Z, Y) -> (T, Y, Y)
        # quad = jnp.diagonal(quad, axis1=1, axis2=2)  # (T, Y)
        # lam = jnp.exp(eta + .5 * quad)
        # g_m = (y - lam) @ C
        # g_V = -.5 * jnp.einsum('ni, tn, nj -> tij', C, lam, C)

        # dJ = -2 * g_V
        # dj = g_m - 2 * g_V @ jnp.expand_dims(m, axis=-1)
        
        dj, dJ = vmap(partial(poisson_cvi_stats, H=H, d=d))(z, Z, y)

        j = (1 - lr) * j + lr * dj
        J = (1 - lr) * J + lr * dJ
        
        return j, J
