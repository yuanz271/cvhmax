"""
- SSM
y[k] = H x[k] + N(0, R[k])
x[k] = F x[k-1] + B u[k] + N(0, Q[k])

- Information vector and matrix
Z[k] = P[k]^-1
z[k] = P[k]^-1 x[k]

- Information filtering
Predict:
M[k] = F[k]^-' Z[k-1] F[k]^-1
C[k] = M[k] (M[k] + Q[k]^-1)^-1
L[k] = I - C[k]
Z[k|k-1] = L[k] M[k] L[k]' + C[k] Q[k]^-1 C[k]'
z[k|k-1] = L[k] F[k]^-' z[k-1]

Update:
J[k] = H[k]' R[k]^-1 H[k]
j[k] = H[k]' R[k]^-1 y[k]
Z[k] = Z[k|k-1] + J[k]
z[k] = z[k|k-1] + j[k]

"""

import functools

import jax
import jax.numpy as jnp
from jax.numpy.linalg import multi_dot, solve
from jaxtyping import Float, Array


def predict(z, Z, F, P):
    M = solve(F.T, solve(F.T, Z.T).T)
    C = solve((M + P).T, M.T).T
    L = jnp.eye(C.shape[0]) - C
    Zp = multi_dot((L, M, L.T)) + multi_dot((C, P, C.T))
    zp = L @ solve(F.T, z)

    return zp, Zp


def information_filter_step(
    state: tuple[Float[Array, " state"], Float[Array, " state state"]],
    measure: tuple[Float[Array, " state"], Float[Array, " state state"]],
    F: Float[Array, " state state"],
    P: Float[Array, " state state"],
):
    """Single information filtering step
    :param state: previous state, tuple (z, Z)
    :param measure: current measure (i, I)
    :param F: transition matrix
    :param P: state noise precision
    """
    # carries
    z, Z = state
    j, J = measure

    # predict
    zp, Zp = predict(z, Z, F, P)
    # M = solve(F.T, solve(F.T, Z.T).T)
    # C = solve((M + P).T, M.T).T
    # eye = jnp.eye(C.shape[0])
    # L = eye - C
    # Zp = multi_dot((L, M, L.T)) + multi_dot((C, P, C.T))
    # zp = L @ solve(F.T, z)

    # update
    Z = Zp + J
    z = zp + j

    return (z, Z), (zp, Zp, z, Z)


def information_filter(
    init: tuple[Float[Array, " state"], Float[Array, " state state"]],
    measure: tuple[Float[Array, " time state"], Float[Array, " time state state"]],
    F: Float[Array, " state state"],
    P: Float[Array, " state state"],
    reverse: bool = False,
):
    """Information filter
    :param init: initial distribution
    :param measure: observation information time series
    :param F: transition matrix
    :param P: state noise precision matrix
    """
    _, ret = jax.lax.scan(
        functools.partial(information_filter_step, F=F, P=P),
        init=init,
        xs=measure,
        reverse=reverse,
    )

    return ret


@jax.jit
def bifilter(j, J, z0, Z0, Af, Pf, Ab, Pb):
    """Bidirectional filtering"""
    # Forward
    zpf, Zpf, zf, Zf = information_filter((z0, Z0), (j, J), Af, Pf)
    # zTp1, ZTp1 = predict(zf[-1], Zf[-1], Af, Pf)
    # Backward
    zpb, Zpb, zb, Zb = information_filter((z0, Z0), (j, J), Ab, Pb, reverse=True)

    z = zf + zpb  # assume 0 mean
    Z = Zf + Zpb - Z0  # stationary Q
    return z, Z
