"""Symbolic construction of the Hida-Matern kernel.

Builds a SymPy expression for the complex-valued Hida-Matern covariance

    k(tau) = sigma^2 * matern(p, tau, rho) * exp(i * omega * tau)

where ``p = order - 1`` is the Matern smoothness index (nu = p + 1/2).
"""

import sympy as sym

# Canonical symbol names used throughout the symbolic pipeline.
tau = sym.Symbol("tau", positive=True)
sigma = sym.Symbol("sigma", positive=True)
rho = sym.Symbol("rho", positive=True)
omega = sym.Symbol("omega", real=True)


def matern_poly(p: int, tau: sym.Symbol, rho: sym.Symbol) -> sym.Expr:
    """Construct the Matern(nu = p + 1/2) kernel symbolically.

    Parameters
    ----------
    p : int
        Smoothness index. ``p = 0`` gives Matern-1/2 (exponential),
        ``p = 1`` gives Matern-3/2, etc.
    tau : sym.Symbol
        Lag variable (assumed positive).
    rho : sym.Symbol
        Length-scale parameter.

    Returns
    -------
    sym.Expr
        Symbolic Matern kernel expression.
    """
    if p < 0:
        raise ValueError(f"Matern order p must be >= 0, got {p}")

    mult = sym.Rational(sym.factorial(p), sym.factorial(2 * p)) * sym.exp(
        -sym.sqrt(2 * p + 1) * tau / rho
    )

    sum_factor = sym.Integer(0)
    for i in range(p + 1):
        coeff = sym.Rational(
            sym.factorial(p + i),
            sym.factorial(i) * sym.factorial(p - i),
        )
        sum_factor += coeff * (sym.sqrt(8 * p + 4) * tau / rho) ** (p - i)

    return sym.simplify(mult * sum_factor)


def hida_matern_kernel(order: int) -> tuple[sym.Expr, tuple[sym.Symbol, ...]]:
    """Build the full Hida-Matern kernel for the given SSM order.

    Parameters
    ----------
    order : int
        State-space model order (``M``). The Matern smoothness index is
        ``p = order - 1``.

    Returns
    -------
    tuple[sym.Expr, tuple[sym.Symbol, ...]]
        ``(kernel_expr, (tau, sigma, rho, omega))`` where *kernel_expr* is
        the symbolic Hida-Matern covariance and the tuple contains the
        symbols used in the expression.

    Raises
    ------
    ValueError
        If *order* is less than 1.
    """
    if order < 1:
        raise ValueError(f"SSM order must be >= 1, got {order}")

    p = order - 1
    k = sigma**2 * matern_poly(p, tau, rho) * sym.exp(sym.I * omega * tau)
    return k, (tau, sigma, rho, omega)
