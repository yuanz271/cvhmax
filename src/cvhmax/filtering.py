# - SSM
# y[k] = H x[k] + N(0, R[k])
# x[k] = F x[k-1] + B u[k] + N(0, Q[k])

# - Information vector and matrix
# Z[k] = P[k]^-1
# z[k] = P[k]^-1 x[k]

# - Information filtering
# Predict:
# M[k] = F[k]^-' Z[k-1] F[k]^-1
# C[k] = M[k] (M[k] + Q[k]^-1)^-1
# L[k] = I - C[k]
# Z[k|k-1] = L[k] M[k] L[k]' + C[k] Q[k]^-1 C[k]'
# z[k|k-1] = L[k] F[k]^-' z[k-1]

# Update:
# J[k] = H[k]' R[k]^-1 H[k]
# j[k] = H[k]' R[k]^-1 y[k]
# Z[k] = Z[k|k-1] + J[k]
# z[k] = z[k|k-1] + j[k]

from functools import partial

from jax import Array, lax, numpy as jnp
from jax.numpy.linalg import multi_dot, solve


def predict(z, Z, F, P):
    """Run the information filter prediction step.

    Parameters
    ----------
    z : Array
        Information vector at the previous time step.
    Z : Array
        Information matrix at the previous time step.
    F : Array
        State transition matrix.
    P : Array
        State noise precision matrix.

    Returns
    -------
    tuple[Array, Array]
        Predicted information vector and matrix.
    """
    M = solve(F.T, solve(F.T, Z.T).T)
    C = solve((M + P).T, M.T).T
    L = jnp.eye(C.shape[0]) - C
    Zp = multi_dot((L, M, L.T)) + multi_dot((C, P, C.T))
    zp = L @ solve(F.T, z)

    return zp, Zp


def information_filter_step(
    state: tuple[Array, Array],
    measure: tuple[Array, Array],
    F: Array,
    P: Array,
):
    """Advance the information filter by one observation.

    Parameters
    ----------
    state : tuple[Array, Array]
        Previous posterior information vector and matrix `(z, Z)`.
    measure : tuple[Array, Array]
        Observation information vector and matrix `(j, J)` for the current step.
    F : Array
        State transition matrix.
    P : Array
        State noise precision matrix.

    Returns
    -------
    tuple[tuple[Array, Array], tuple[Array, Array, Array, Array]]
        Updated posterior `(z, Z)` together with the predicted and updated
        information tuples `(zp, Zp, z, Z)` emitted for downstream use.
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
    init: tuple[Array, Array],
    measure: tuple[Array, Array],
    F: Array,
    P: Array,
    reverse: bool = False,
):
    """Filter an information-state sequence given observation information.

    Parameters
    ----------
    init : tuple[Array, Array]
        Initial information vector and matrix `(z0, Z0)`.
    measure : tuple[Array, Array]
        Observation information sequences `(j, J)`.
    F : Array
        State transition matrix.
    P : Array
        State noise precision matrix.
    reverse : bool, default=False
        Whether to scan the observation sequence in reverse order.

    Returns
    -------
    tuple[Array, Array, Array, Array]
        Tuple of predicted and posterior information vectors and matrices from
        each filtering step.
    """
    _, ret = lax.scan(
        partial(information_filter_step, F=F, P=P),
        init=init,
        xs=measure,
        reverse=reverse,
    )

    return ret


def bifilter(j, J, z0, Z0, Af, Pf, Ab, Pb) -> tuple[Array, Array]:
    """Run forward and backward information filters and merge the results.

    Parameters
    ----------
    j : Array
        Observation information vectors.
    J : Array
        Observation information matrices.
    z0 : Array
        Initial information vector.
    Z0 : Array
        Initial information matrix.
    Af : Array
        Forward transition matrix.
    Pf : Array
        Forward state noise precision matrix.
    Ab : Array
        Backward transition matrix.
    Pb : Array
        Backward state noise precision matrix.

    Returns
    -------
    tuple[Array, Array]
        Smoothed information vector and matrix.
    """
    # Forward
    zpf, Zpf, zf, Zf = information_filter((z0, Z0), (j, J), Af, Pf)
    # zTp1, ZTp1 = predict(zf[-1], Zf[-1], Af, Pf)
    # Backward
    zpb, Zpb, zb, Zb = information_filter((z0, Z0), (j, J), Ab, Pb, reverse=True)

    z = zf + zpb  # assume 0 mean
    Z = Zf + Zpb - Z0  # stationary Q
    return z, Z
