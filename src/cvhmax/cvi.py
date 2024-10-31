from abc import abstractmethod
from dataclasses import dataclass, field
from functools import partial

from jax import numpy as jnp, random as jrandom, vmap, lax, nn
from jax.numpy.linalg import inv, solve, multi_dot
from jax.scipy.linalg import cho_factor, cho_solve
from jaxtyping import Array
import chex

from .utils import info_repr, norm_loading, lbfgs_solve


@dataclass
class Params:
    C: Array = field(default=None)
    d: Array = field(default=None)
    R: Array = field(default=None)
    M: Array = field(default=None)
    
    def initialize(self, n_obs, n_factors, *, random_state):
        key = jrandom.key(random_state)
        Ckey, dkey = jrandom.split(key)
        self.C = jrandom.normal(Ckey, shape=(n_obs, n_factors)) / n_obs
        self.d = jrandom.normal(dkey, shape=(n_obs,))
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
    def init_info(cls, *args, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def initialize_params(cls, *args, **kwargs):
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
    
    d = jnp.squeeze(d)
    C = C.T
    assert d.shape == (y_dim,)
    assert C.shape == (y_dim, z_dim)

    return C, d, R


class Gaussian(CVI):
    @staticmethod
    def update_pseudo(params, y):
        C = params.C()
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
    d, C = jnp.split(w, [1], axis=1)
    return jnp.squeeze(d), C


def poisson_nell(params, y, m, V, reg=100.):
    C, d = params
    n = y.shape[0]
    
    def _nell(y_t, m_t, V_t):
        eta = C @ m_t + d
        quad = jnp.einsum("ni,in->n", C, V_t @ C.T)  # (Y, Z) (T, Z, Z) (Z, Y) -> (T, Y, Y)
        lam = jnp.exp(eta + 0.5 * quad)
        return jnp.sum(lam - eta * y_t, axis=-1)
    
    C_reg = reg * jnp.linalg.norm(C) / n
    return jnp.mean(vmap(_nell)(y, m, V)) + C_reg


def poisson_cvi_stats(z, Z, y, H, d):
    """
    z = V^-1 m
    Z = -0.5 V^-1
    <=>
    m = Vz = -0.5 * Z^-1 z
    V = -0.5Z^-1
    """
    # chex.assert_tree_all_finite(z)
    # chex.assert_tree_all_finite(Z)
    
    n = jnp.size(z)
    tau = 1e-3
    # Zcho = cho_factor(Z + tau * jnp.eye(n))
    Z = Z + tau * jnp.eye(n)
    m = -0.5 * solve(Z, z)  # Vj
    # chex.assert_tree_all_finite(m)

    lin = H @ m + d
    quad = jnp.einsum("nl, ln -> n", H, -0.5 * solve(Z, H.mT))  # CVC'
    eta =  jnp.maximum(lin + 0.5 * quad, 5)
    lam = jnp.exp(eta)

    # chex.assert_tree_all_finite(lam)
    # chex.assert_tree_all_finite(quad)

    grad_m = (y - lam) @ H
    grad_V = -0.5 * jnp.einsum("ni, n, nj -> ij", H, lam, H)

    # chex.assert_tree_all_finite(grad_m)
    # chex.assert_tree_all_finite(grad_V)

    J_update = -2 * grad_V
    j_update = grad_m - 2 * grad_V @ m

    # chex.assert_tree_all_finite(J_update)
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
        
        d_z = Q.shape[0]
        z0 = jnp.zeros(d_z)
        Z0 = P = inv(Q)

        A = A + 1e-3 * jnp.eye(d_z)  # ill-condition

        def forward(carry, yt):
            ztm1, Ztm1 = carry

            # predict
            M = solve(A.T, solve(A.T, Ztm1.T).T)
            C = solve((M + P).T, M.T).T
            eye = jnp.eye(C.shape[0])
            L = eye - C
            Zp = multi_dot((L, M, L.T)) + multi_dot((C, P, C.T))
            zp = L @ solve(A.T, ztm1)

            j, J = poisson_cvi_stats(zp, Zp, yt, H, d)

            zt = zp + j
            Zt = Zp + J
            Zt = 0.5 * (Zt + Zt.mT)

            return (zt, Zt), (j, J)
        
        ztm1 = z0
        Ztm1 = Z0
        j = []
        J = []
        for yt in y:
            (ztm1, Ztm1), (jt, Jt) = forward((ztm1, Ztm1), yt)
            j.append(jt)
            J.append(Jt)
        # _, (j, J) = lax.scan(forward, (z0, Z0), y)
        j = jnp.stack(j)
        J = jnp.stack(J)
        return j, J

    @classmethod
    def update_readout(cls, params, y, m, V):
        C = params.C
        d = params.d
        M = params.M
        R = params.R
        
        y = jnp.concatenate(y, axis=0)
        m = jnp.concatenate(m, axis=0)
        V = jnp.concatenate(V, axis=0)
        # selected_m = m @ M.T  # (T, Z) (Z, sZ) -> (T, sZ)
        # selected_V = vmap(lambda v: M @ v @ M.T)(V)  # (T, Z, Z) -> (T, sZ, sZ)

        # w = jnp.column_stack([d, C])
        # w_shape = w.shape
        # w = w.flatten()

        # opt = minimize(poisson_nell, w, args=(w_shape, y, m, V), method="BFGS")
        # w_opt = opt.x

        # d, C = v2w(w_opt, w_shape)

        (C, d) , _ = lbfgs_solve((C, d), partial(poisson_nell, y=y, m=m, V=V))
        
        params = Params()
        params.C = C
        params.d = d
        params.R = R
        params.M = M
        return params
    
    @classmethod
    def update_pseudo(cls, params, y, z, Z, j, J, lr):
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

        chex.assert_tree_all_finite(dj)
        chex.assert_tree_all_finite(dJ)

        j = (1 - lr) * j + lr * dj
        J = (1 - lr) * J + lr * dJ
        
        return j, J
    
    @classmethod
    def initialize_params(cls, y, n_factors, *, random_state):
        key: Array = jrandom.key(random_state)
        Ckey, dkey = jrandom.split(key)
        
        m = jnp.mean(y, axis=0)
        d = jnp.log(m)
        r = y - m[None, ...]

        u, s, vh = jnp.linalg.svd(r, full_matrices=False)
        x = u[:, :n_factors]

        n, n_obs = y.shape
        V = jnp.tile(jnp.eye(n_factors), (n, 1, 1))  # dummpy variance
        C = jrandom.normal(Ckey, shape=(n_obs, n_factors)) / n_obs

        (C, d), _ = lbfgs_solve((C, d), partial(poisson_nell, y=y, m=x, V=V), max_iter=15000)

        return Params(C=C, d=d)
    
    @classmethod
    def cvi(cls, params, jJ, y, smooth_fun, smooth_args, cvi_iter, lr):
        for cv_it in range(cvi_iter):
            # print(f"\n{cv_it=}")
            zZ = [
                smooth_fun(jk, Jk, *smooth_args) for (jk, Jk) in jJ
            ]

            # for z, Z in zZ:
            #     print("zZ")
            #     print(jnp.mean(z, axis=0))
            #     print(jnp.mean(Z, axis=0))

            jJ = [cls.update_pseudo(params, yk, zk, Zk, jk, Jk, lr) for (zk, Zk), yk, (jk, Jk) in zip(zZ, y, jJ)]

            # for j, J in jJ:
            #     print("jJ")
            #     print(jnp.mean(j, axis=0))
            #     print(jnp.mean(J, axis=0))
        
        return zZ, jJ
