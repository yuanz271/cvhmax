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
I[k] = H[k]' R[k]^-1 H[k]
i[k] = H[k]' R[k]^-1 y[k]
Z[k] = Z[k|k-1] + I[k]
z[k] = z[k|k-1] + i[k]

"""
import functools

import jax
import jax.numpy as jnp
from jax.numpy.linalg import multi_dot, solve


def information_filter_step(state, measure, F, P):
    """
    Single information filtering step 
    given previous state, current measurement and model parameters

    param state: previous state, tuple (z, Z)
    param measure: (i, I)
        i: for mean, (p,)
        I: for precision, (p, p)
    param F: transition matrix
    param P: state noise precision
    """
    # carries
    z, Z = state
    i, I = measure
        
    # predict
    M = solve(F.T, solve(F.T, Z.T).T)
    C = solve((M + P).T, M.T).T
    eye = jnp.eye(C.shape[0])
    L = eye - C
    Zp = multi_dot((L, M, L.T)) + multi_dot((C, P, C.T))
    zp = L @ solve(F.T, z)

    # update
    # I = multi_dot(H.T, Rinv, H)
    # i = multi_dot(H.T, Rinv, y)

    Z = Zp + I
    z = zp + i
    
    return (z, Z), (zp, Zp, z, Z)


def information_filter(init, measure, F, P):
    """
    Information filter

    param init: initial state, tuple
    param measure: (i, I)
        i: array (T, ...), i[k], (p,)
        I: matrix or matrices (T, ...), I[k], (p, p)
    param F: transition matrix
    param P: state noise precision matrix
    """

    step = functools.partial(information_filter_step, F=F, P=P)
    _, z = jax.lax.scan(step, init=init, xs=measure)
    
    return z


def bifilter(i, I, z0, Z0, Af, Pf, Ab, Pb):
    zpf, Zpf, zf, Zf = information_filter((z0, Z0), (i, I), Af, Pf)
    zpb, Zpb, zb, Zb = information_filter((z0, Z0), (jnp.flip(i, axis=0), jnp.flip(I, axis=0)), Ab, Pb)

    z = zf + zpb
    Z = Zf + Zpb - Z0
    return z, Z
