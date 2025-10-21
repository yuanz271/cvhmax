from abc import abstractmethod
from collections.abc import Callable
from functools import partial
from typing import ClassVar, override
import numpy as np
from sklearn.decomposition import FactorAnalysis

from jax import Array, lax, numpy as jnp, vmap
from jax.numpy.linalg import inv, solve, multi_dot
from jax.scipy.linalg import cho_factor, cho_solve
from equinox import Module

from cvhmax.utils import ridge_estimate

from .utils import filter_array, trial_info_repr, norm_loading, lbfgs_solve


TAU = 1e-6
MAX_LOGRATE = 7.0


def fa_init(ys, n_components, random_state):
    fa = FactorAnalysis(n_components=n_components, random_state=random_state)
    fa.fit(ys)
    ms = jnp.array(fa.transform(ys))
    C = jnp.array(fa.components_.T)
    d = jnp.array(fa.mean_)

    return ms, C, d


class Params(Module):
    C: Array
    d: Array
    R: Array
    M: Array

    def loading(self) -> Array:
        return norm_loading(self.C)

    def lmask(self) -> Array:
        return lax.stop_gradient(self.M)


class CVI:
    registry: ClassVar[dict] = dict()

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        CVI.registry[cls.__name__] = cls

    @classmethod
    def infer(
        cls,
        params: Params,
        j: Array,
        J: Array,
        y: Array,
        ymask: Array,
        z0: Array,
        Z0: Array,
        smooth_fun: Callable,
        smooth_args: tuple,
        cvi_iter: int,
        lr: float,
    ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
        smooth_batch = vmap(
            lambda jk, Jk, zk0, Zk0: smooth_fun(jk, Jk, zk0, Zk0, *smooth_args)
        )

        def step(i, carry) -> tuple[Array, Array]:
            j, J = carry
            z, Z = smooth_batch(j, J, z0, Z0)
            j, J = cls.update_pseudo(params, y, ymask, z, Z, j, J, lr)
            return j, J

        j, J = lax.fori_loop(0, cvi_iter, step, (j, J))
        z, Z = vmap(
            lambda jk, Jk, zk0, Zk0: smooth_fun(jk, Jk, zk0, Zk0, *smooth_args)
        )(j, J, z0, Z0)

        return (z, Z), (j, J)

    @classmethod
    @abstractmethod
    def update_readout(cls, *args, **kwargs) -> tuple[Params, float]: ...

    @classmethod
    @abstractmethod
    def update_pseudo(
        cls,
        params: Params,
        y: Array,
        ymask: Array,
        z: Array,
        Z: Array,
        j: Array,
        J: Array,
        lr: float,
    ) -> tuple[Array, Array]: ...

    @classmethod
    @abstractmethod
    def initialize_info(
        cls, params: Params, y: Array, ymask: Array, A: Array, Q: Array
    ) -> tuple[Array, Array]: ...

    @classmethod
    @abstractmethod
    def initialize_params(cls, *args, **kwargs) -> Params: ...


class Gaussian(CVI):
    @classmethod
    @override
    def infer(
        cls,
        params: Params,
        j: Array,
        J: Array,
        y: Array,
        ymask: Array,
        z0: Array,
        Z0: Array,
        smooth_fun: Callable,
        smooth_args: tuple,
        cvi_iter: int,
        lr: float,
    ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
        return CVI.infer(params, j, J, y, ymask, z0, Z0, smooth_fun, smooth_args, 1, lr)

    @classmethod
    @override
    def update_pseudo(
        cls,
        params: Params,
        y: Array,
        ymask: Array,
        z: Array,
        Z: Array,
        j: Array,
        J: Array,
        lr: float,
    ) -> tuple[Array, Array]:
        return j, J

    @classmethod
    @override
    def initialize_info(
        cls, params: Params, y: Array, ymask: Array, A: Array, Q: Array
    ) -> tuple[Array, Array]:
        C = params.loading()
        d = params.d
        R: Array = params.R
        M = params.lmask()

        H = C @ M

        return vmap(partial(trial_info_repr, C=H, d=d, R=R))(y, ymask)

    @classmethod
    @override
    def update_readout(
        cls, params: Params, y: Array, ymask: Array, m: Array, P: Array
    ) -> tuple[Params, float]:
        y = filter_array(y, ymask)
        m = filter_array(m, ymask)
        C, d, R = ridge_estimate(y, m, P)
        params = Params(C=C, d=d, R=R, M=params.M)
        return params, jnp.nan

    @classmethod
    @override
    def initialize_params(
        cls,
        y: Array,
        ymask: Array,
        n_factors: int,
        lmask: Array,
        *,
        random_state: int,
    ) -> Params:
        y = filter_array(y, ymask)
        _, C, d = fa_init(y, n_factors, random_state)

        return Params(C=C, d=d, R=jnp.zeros(y.shape[-1]), M=lmask)


def poisson_trial_nell(
    params: tuple[Array, Array],
    y: Array,
    m: Array,
    V: Array,
    gamma: float = 10.0,
) -> float:
    """
    :param gamma: regularization
    """
    C, d = params

    def bin_nell(y_t, m_t, V_t):
        lin = C @ m_t + d
        quad = jnp.einsum(
            "ni,in->n", C, V_t @ C.T
        )  # (Y, Z) (T, Z, Z) (Z, Y) -> (T, Y, Y)
        eta = jnp.minimum(lin + 0.5 * quad, MAX_LOGRATE)
        lam = jnp.exp(eta)
        return jnp.sum(lam - eta * y_t, axis=-1)

    C_reg = gamma * jnp.linalg.norm(C) / y.shape[0]
    bin_nells = vmap(bin_nell)(y, m, V)

    return jnp.mean(bin_nells) + C_reg


def poisson_cvi_bin_stats(
    z: Array,
    Z: Array,
    y: Array,
    ymask: Array,
    H: Array,
    d: Array,
) -> tuple[Array, Array]:
    """
    H = CM
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

    K = -2 * grad_V
    k = grad_m - 2 * grad_V @ m

    k = jnp.where(jnp.expand_dims(ymask, -1), k, 0)
    K = jnp.where(jnp.expand_dims(ymask, (-2, -1)), K, 0)

    return k, K


class Poisson(CVI):
    @classmethod
    def initialize_info(
        cls,
        params: Params,
        y: Array,
        ymask: Array,
        A: Array,
        Q: Array,
    ) -> tuple[Array, Array]:
        """Initialize pseudo observation"""
        C = params.loading()
        M = params.lmask()
        H = C @ M
        d = params.d

        d_z = Q.shape[0]
        z0 = jnp.zeros(d_z)
        Z0 = P = inv(Q)

        A = A + 1e-3 * jnp.eye(d_z)  # ill-condition

        def forward(
            carry: tuple[Array, Array], ys: tuple[Array, Array]
        ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
            ztm1, Ztm1 = carry
            yt, ytmask = ys

            # predict
            M = solve(A.T, solve(A.T, Ztm1.T).T)
            G = solve((M + P).T, M.T).T
            eye = jnp.eye(G.shape[0])
            L = eye - G
            Zp = multi_dot((L, M, L.T)) + multi_dot((G, P, G.T))
            zp = L @ solve(A.T, ztm1)

            j, J = poisson_cvi_bin_stats(zp, Zp, yt, ytmask, H, d)

            zt = zp + j
            Zt = Zp + J
            Zt = 0.5 * (Zt + Zt.mT)

            return (zt, Zt), (j, J)

        _, (j, J) = lax.scan(forward, (z0, Z0), (y, ymask))

        return j, J

    @classmethod
    def update_readout(
        cls,
        params: Params,
        y: Array,
        ymask: Array,
        m: Array,
        V: Array,
    ) -> tuple[Params, float]:
        C = params.loading()
        d = params.d
        R = params.R

        y = filter_array(y, ymask)
        m = filter_array(m, ymask)
        V = filter_array(V, ymask)

        C, d = lbfgs_solve((C, d), partial(poisson_trial_nell, y=y, m=m, V=V))  # type: ignore

        nell = poisson_trial_nell((C, d), y=y, m=m, V=V, gamma=0.0)  # type: ignore
        return Params(C=C, d=d, R=R, M=params.M), nell  # type: ignore

    @classmethod
    def update_pseudo(
        cls,
        params: Params,
        y: Array,
        ymask: Array,
        z: Array,
        Z: Array,
        j: Array,
        J: Array,
        lr: float,
    ) -> tuple[Array, Array]:
        """
        :param params: readout
        :param y: observation
        :param ymask: missing mask (0: missing, 1: normal)
        :param z: latent's 1st posterior natural param
        :param Z: latent's 2nd posterior natural param
        :param j: update to 1st natural param
        :param J: update to 2nd natural param
        :param lr: learning rate
        """
        C = params.loading()
        M = params.lmask()
        H = C @ M
        d = params.d
        # print(f"{z.shape=} {Z.shape=} {y.shape=}, {ymask.shape=}")
        k, K = vmap(vmap(partial(poisson_cvi_bin_stats, H=H, d=d)))(
            z, Z, y, ymask
        )  # session

        j = (1 - lr) * j + lr * k
        J = (1 - lr) * J + lr * K

        return j, J

    @classmethod
    def initialize_params(
        cls,
        y: Array,
        ymask: Array,
        n_factors: int,
        lmask: Array,
        *,
        random_state: int,
    ) -> Params:
        y = filter_array(y, ymask)
        m, C, d = fa_init(y, n_factors, random_state)

        n_bins, n_obs = y.shape
        V = jnp.zeros((n_bins, n_factors, n_factors))  # dummpy variance

        C, d = lbfgs_solve((C, d), partial(poisson_trial_nell, y=y, m=m, V=V))  # type: ignore

        return Params(C=C, d=d, R=None, M=lmask)  # type: ignore
