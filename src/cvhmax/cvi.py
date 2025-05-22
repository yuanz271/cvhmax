from abc import abstractmethod
from functools import partial

import numpy as np
from sklearn.decomposition import FactorAnalysis

from jax import numpy as jnp, vmap
from jax.numpy.linalg import inv, solve, multi_dot
from jax.scipy.linalg import cho_factor, cho_solve
from jaxtyping import Array
from equinox import Module, field

from .utils import info_repr, norm_loading, lbfgs_solve


TAU = 1e-6
MAX_LOGRATE = 7.0


def fa_init(ys, n_components, random_state):
    fa = FactorAnalysis(n_components=n_components, random_state=random_state)
    Y = np.vstack(ys)
    fa.fit(Y)
    ms = [jnp.array(fa.transform(y)) for y in ys]
    C = jnp.array(fa.components_.T)
    d = jnp.array(fa.mean_)

    return ms, C, d


class Params(Module):
    C: Array
    d: Array
    R: Array | None
    M: Array = field(static=True)

    # def initialize(self, n_obs, n_factors, *, random_state):
    #     key = jrandom.key(random_state)
    #     Ckey, dkey = jrandom.split(key)
    #     self.C = jrandom.normal(Ckey, shape=(n_obs, n_factors)) / n_obs
    #     self.d = jrandom.normal(dkey, shape=(n_obs,))
    #     self.R = jnp.eye(n_obs)

    def nC(self) -> Array:
        return norm_loading(self.C)


class CVI:
    @classmethod
    @abstractmethod
    def cvi(cls, *args, **kwargs):
        pass

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


def ridge_estimate(y, m, V, lam=0.1):
    """
    Ridge regression
    w = (z'z + lamI)^-1 z'y
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
    @classmethod
    def update_pseudo(cls, params, y, z, Z, j, J, lr):
        return j, J

    @classmethod
    def init_info(cls, params, y, A, Q):
        C = params.nC()
        d = params.d
        R = params.R
        M = params.M

        H = C @ M

        return info_repr(y, H, d, R)

    @classmethod
    def update_readout(cls, params, y, m, P):
        C, d, R = ridge_estimate(y, m, P)
        params = Params(C=C, d=d, R=R, M=params.M)
        return params

    @classmethod
    def cvi(cls, params, jJ, y, zZ0, smooth_fun, smooth_args, cvi_iter, lr):
        # observation updates are state independent
        zZ = [
            smooth_fun(jk, Jk, z0, Z0, *smooth_args)
            for (jk, Jk), (z0, Z0) in zip(jJ, zZ0)
        ]

        return zZ, jJ

    @classmethod
    def initialize_params(cls, ys, n_factors, mask, *, random_state):
        # key: Array = jrandom.key(random_state)
        # Ckey, dkey = jrandom.split(key)

        ms, C, d = fa_init(ys, n_factors, random_state)
        # zZ0 = [(m[0], jnp.eye(n_factors)) for m in ms]

        return Params(C=C, d=d, R=None, M=mask)


def poisson_nell(params, y, m, V, gamma=10.0):
    """
    :param gamma: regularization
    """
    C, d = params
    n = y.shape[0]

    def _nell(y_t, m_t, V_t):
        lin = C @ m_t + d
        quad = jnp.einsum(
            "ni,in->n", C, V_t @ C.T
        )  # (Y, Z) (T, Z, Z) (Z, Y) -> (T, Y, Y)
        eta = jnp.minimum(lin + 0.5 * quad, MAX_LOGRATE)
        lam = jnp.exp(eta)
        return jnp.sum(lam - eta * y_t, axis=-1)

    C_reg = gamma * jnp.linalg.norm(C) / n
    return jnp.mean(vmap(_nell)(y, m, V)) + C_reg


def poisson_cvi_stats(z, Z, y, H, d):
    """
    z = V^-1 m
    Z = -0.5 V^-1
    <=>
    m = Vz = -0.5 * Z^-1 z
    V = -0.5Z^-1
    """
    U, s, V = jnp.linalg.svd(Z)
    Z = multi_dot((U, jnp.diag(s + TAU), U.T))

    Zcho = cho_factor(Z)
    m = -0.5 * cho_solve(Zcho, z)  # Vj

    lin = H @ m + d
    quad = jnp.einsum("nl, ln -> n", H, -0.5 * cho_solve(Zcho, H.mT))  # CVC'
    eta = jnp.minimum(lin + 0.5 * quad, MAX_LOGRATE)
    lam = jnp.exp(eta)

    grad_m = (y - lam) @ H
    grad_V = -0.5 * jnp.einsum("ni, n, nj -> ij", H, lam, H)

    J_update = -2 * grad_V
    j_update = grad_m - 2 * grad_V @ m

    return j_update, J_update


class Poisson(CVI):
    @classmethod
    def init_info(cls, params: Params, y, A, Q):
        """Initialize pseudo observation"""
        C = params.nC()
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
            G = solve((M + P).T, M.T).T
            eye = jnp.eye(G.shape[0])
            L = eye - G
            Zp = multi_dot((L, M, L.T)) + multi_dot((G, P, G.T))
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
    def update_readout(cls, params: Params, y, m, V):
        C = params.nC()
        d = params.d
        M = params.M
        R = params.R
        
        (C, d), _ = lbfgs_solve((C, d), partial(poisson_nell, y=y, m=m, V=V))  # type: ignore

        nell = poisson_nell((C, d), y=y, m=m, V=V, gamma=0.0)
        return Params(C=C, d=d, R=R, M=M), nell  # type: ignore

    @classmethod
    def update_pseudo(cls, params: Params, y, z, Z, j, J, lr):
        """
        :param params: readout
        :param z: 1st posterior natural param of latent
        :param Z: 2nd posterior natural param of latent
        :param j: 1st natural param of posterior pseudo observation
        :param J: 2nd natural param of posterior pseudo observation
        :param y: observation
        :param lr: learning rate
        """
        C = params.nC()
        M = params.M
        H = C @ M
        d = params.d
        k, K = vmap(partial(poisson_cvi_stats, H=H, d=d))(z, Z, y)

        j = (1 - lr) * j + lr * k
        J = (1 - lr) * J + lr * K

        return j, J

    @classmethod
    def initialize_params(cls, ys, n_factors, mask, *, random_state):
        ms, C, d = fa_init(ys, n_factors, random_state)
        Y = jnp.vstack(ys)
        M = jnp.vstack(ms)
        n, n_obs = Y.shape
        V = jnp.tile(jnp.zeros((n_factors, n_factors)), (n, 1, 1))  # dummpy variance

        (C, d), _ = lbfgs_solve((C, d), partial(poisson_nell, y=Y, m=M, V=V))  # type: ignore

        return Params(C=C, d=d, R=None, M=mask)  # type: ignore

    @classmethod
    def cvi(cls, params: Params, jJ, y, zZ0, smooth_fun, smooth_args, cvi_iter, lr):
        for cv_it in range(cvi_iter):
            # print(f"\n{cv_it=}")
            zZ = [
                smooth_fun(jk, Jk, z0, Z0, *smooth_args)
                for (jk, Jk), (z0, Z0) in zip(jJ, zZ0)
            ]

            jJ = [
                cls.update_pseudo(params, yk, zk, Zk, jk, Jk, lr)
                for (zk, Zk), yk, (jk, Jk) in zip(zZ, y, jJ)
            ]

        return zZ, jJ
