"""Runtime Hida-Matern kernel generator using SymPy + sympy2jax.

Symbolically differentiates the Hida-Matern covariance function and
converts the resulting expressions into JIT-compatible JAX functions
via ``sympy2jax.SymbolicModule``.
"""

from __future__ import annotations

from functools import lru_cache

import jax.numpy as jnp
from jax import Array
import sympy as sym
import sympy2jax

from .matern import hida_matern_kernel, tau as _tau


class HidaMaternKernelGenerator:
    """Generate JAX-callable kernel functions for arbitrary HM order.

    Given an SSM order ``M``, this class symbolically differentiates the
    Hida-Matern kernel and produces three callable outputs:

    * ``create_K_hat(tau, sigma, rho, omega)`` — the M x M complex
      state-space covariance matrix.
    * ``get_moments(sigma, rho, omega)`` — the 2M spectral moments.
    * ``get_base_kernel(tau, sigma, rho, omega)`` — the scalar base
      kernel evaluated at ``|tau|``.

    All returned functions are compatible with ``jax.jit``, ``jax.vmap``,
    and ``jax.grad``.

    Parameters
    ----------
    order : int
        State-space model order (M >= 1).

    Examples
    --------
    >>> gen = HidaMaternKernelGenerator(2)
    >>> K = gen.create_K_hat(jnp.array(0.5), jnp.array(1.0),
    ...                      jnp.array(1.0), jnp.array(0.0))
    >>> K.shape
    (2, 2)
    """

    def __init__(self, order: int):
        if order < 1:
            raise ValueError(f"SSM order must be >= 1, got {order}")

        self.order = order

        # Build symbolic kernel and compute derivatives
        kernel_expr, symbols = hida_matern_kernel(order)
        self._symbols = symbols  # (tau, sigma, rho, omega)

        # Compute successive derivatives and their limits at tau=0
        derivs, limits = _compute_partials(kernel_expr, _tau, 2 * order)

        # Build the K_hat matrix entries symbolically
        K_hat_entries, K_hat_limit_entries = _build_K_hat_symbolic(
            derivs, limits, order
        )

        # Convert symbolic expressions to JAX functions via sympy2jax
        self._K_hat_module = _build_K_hat_module(
            K_hat_entries, K_hat_limit_entries, order
        )

        # Build moments (limits at tau=0)
        self._moments_values = _build_moments_symbolic(limits, order)
        self._moments_module = sympy2jax.SymbolicModule(
            expressions=self._moments_values
        )

        # Build base kernel module
        self._base_kernel_module = sympy2jax.SymbolicModule(expressions=[kernel_expr])

    def create_K_hat(self, tau: Array, sigma: Array, rho: Array, omega: Array) -> Array:
        """Evaluate the M x M complex state-space covariance at lag *tau*.

        Parameters
        ----------
        tau : Array
            Time lag (scalar). The absolute value is used internally.
        sigma : Array
            Kernel amplitude.
        rho : Array
            Length-scale parameter.
        omega : Array
            Oscillation frequency.

        Returns
        -------
        Array
            Complex-valued (M, M) covariance matrix.
        """
        M = self.order
        tau_abs = jnp.abs(tau)

        # Evaluate all K_hat entries (outer + inner entries, general case)
        general_vals = self._K_hat_module["general"](
            tau=tau_abs, sigma=sigma, rho=rho, omega=omega
        )
        limit_vals = self._K_hat_module["limit"](sigma=sigma, rho=rho, omega=omega)

        # Use limit values when tau == 0
        is_zero = jnp.equal(tau_abs, 0.0)

        K_hat = jnp.zeros((M, M), dtype=jnp.complex128)

        idx = 0
        # Fill outer entries (row 0 and last column)
        for r in range(M):
            for c in range(M):
                if r == 0 or c == M - 1:
                    val = jnp.where(is_zero, limit_vals[idx], general_vals[idx])
                    K_hat = K_hat.at[r, c].set(val)
                    idx += 1

        # Fill inner entries via antisymmetry: K[r, c] = -K[r-1, c+1]
        for r in range(1, M):
            for c in range(M - 1):
                K_hat = K_hat.at[r, c].set(-K_hat[r - 1, c + 1])

        return K_hat

    def get_moments(self, sigma: Array, rho: Array, omega: Array) -> Array:
        """Return the 2M spectral moments.

        Parameters
        ----------
        sigma : Array
            Kernel amplitude.
        rho : Array
            Length-scale parameter.
        omega : Array
            Oscillation frequency.

        Returns
        -------
        Array
            Real-valued vector of length 2M. Odd-indexed entries are zero.
        """
        vals = self._moments_module(sigma=sigma, rho=rho, omega=omega)
        moments = jnp.stack([v for v in vals])
        return moments.real

    def get_base_kernel(
        self, tau: Array, sigma: Array, rho: Array, omega: Array
    ) -> Array:
        """Evaluate the scalar base kernel at ``|tau|``.

        Parameters
        ----------
        tau : Array
            Time lag (scalar or vector).
        sigma : Array
            Kernel amplitude.
        rho : Array
            Length-scale parameter.
        omega : Array
            Oscillation frequency.

        Returns
        -------
        Array
            Real part of the base kernel.
        """
        tau_abs = jnp.abs(tau)
        result = self._base_kernel_module(
            tau=tau_abs, sigma=sigma, rho=rho, omega=omega
        )
        return result[0].real


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_partials(
    kernel: sym.Expr, dx: sym.Symbol, n: int
) -> tuple[list[sym.Expr], list[sym.Expr]]:
    """Compute successive derivatives and their limits at dx -> 0+.

    Parameters
    ----------
    kernel : sym.Expr
        Symbolic kernel expression.
    dx : sym.Symbol
        Differentiation variable.
    n : int
        Number of derivatives to compute (0-th through (n-1)-th).

    Returns
    -------
    tuple[list[sym.Expr], list[sym.Expr]]
        Lists of derivative expressions and their limits at ``dx = 0``.
    """
    derivs = [None] * n
    limits = [None] * n

    derivs[0] = sym.simplify(kernel)
    limits[0] = sym.simplify(kernel.limit(dx, 0, "+"))

    for p in range(1, n):
        derivs[p] = sym.simplify(derivs[p - 1].diff(dx))
        limits[p] = sym.simplify(derivs[p].limit(dx, 0, "+"))

    return derivs, limits


def _build_K_hat_symbolic(
    derivs: list[sym.Expr],
    limits: list[sym.Expr],
    order: int,
) -> tuple[list[sym.Expr], list[sym.Expr]]:
    """Build the symbolic entries for the K_hat matrix (outer positions only).

    The outer positions are row 0 and column M-1. Inner entries are derived
    from the antisymmetry relation K[r, c] = -K[r-1, c+1] at evaluation time.

    Parameters
    ----------
    derivs : list[sym.Expr]
        Derivative expressions from :func:`_compute_partials`.
    limits : list[sym.Expr]
        Limit expressions from :func:`_compute_partials`.
    order : int
        SSM order M.

    Returns
    -------
    tuple[list[sym.Expr], list[sym.Expr]]
        ``(general_entries, limit_entries)`` for the outer positions.
    """
    general_entries = []
    limit_entries = []

    for r in range(order):
        for c in range(order):
            if r == 0 or c == order - 1:
                sign = sym.Integer(-1) ** c
                general_entries.append(sign * derivs[r + c])
                limit_entries.append(sign * limits[r + c])

    return general_entries, limit_entries


def _build_K_hat_module(
    general_entries: list[sym.Expr],
    limit_entries: list[sym.Expr],
    order: int,
) -> dict[str, sympy2jax.SymbolicModule]:
    """Convert symbolic K_hat entries into sympy2jax modules.

    Parameters
    ----------
    general_entries : list[sym.Expr]
        Symbolic expressions for the general (tau > 0) case.
    limit_entries : list[sym.Expr]
        Symbolic expressions for the tau = 0 limit case.
    order : int
        SSM order M.

    Returns
    -------
    dict[str, sympy2jax.SymbolicModule]
        Dictionary with ``"general"`` and ``"limit"`` modules.
    """
    # The limit entries don't depend on tau — substitute tau=0 symbolically
    # to remove it from the expression (sympy2jax needs all symbols present
    # or absent consistently). Since limits are already evaluated at tau->0,
    # we just need to make sure tau isn't a free symbol.
    # For the general module, tau is present.
    general_mod = sympy2jax.SymbolicModule(expressions=general_entries)

    # Limit entries should not contain tau (they're limits), but let's be safe
    # and substitute in case sympy left residual tau references
    clean_limits = [expr.subs(_tau, 0) for expr in limit_entries]
    limit_mod = sympy2jax.SymbolicModule(expressions=clean_limits)

    return {"general": general_mod, "limit": limit_mod}


def _build_moments_symbolic(limits: list[sym.Expr], order: int) -> list[sym.Expr]:
    """Build the 2M spectral moment expressions.

    Even-indexed moments are the absolute value of the even-order derivative
    at tau=0. Odd-indexed moments are zero.

    Parameters
    ----------
    limits : list[sym.Expr]
        Limit expressions from :func:`_compute_partials`.
    order : int
        SSM order M.

    Returns
    -------
    list[sym.Expr]
        List of 2M symbolic moment expressions.
    """
    moments = []
    for m in range(2 * order):
        if m % 2 == 0:
            # Absolute value of the limit; strip leading minus sign
            val = limits[m]
            moments.append(sym.Abs(val))
        else:
            moments.append(sym.Float(0.0))

    return moments


@lru_cache(maxsize=32)
def make_kernel(order: int) -> HidaMaternKernelGenerator:
    """Return a cached kernel generator for the given HM order.

    Parameters
    ----------
    order : int
        State-space model order (M >= 1). Corresponds to Matern smoothness
        nu = (order - 1) + 0.5, i.e. order 1 -> Matern-1/2, order 2 ->
        Matern-3/2, etc.

    Returns
    -------
    HidaMaternKernelGenerator
        Cached generator instance.
    """
    return HidaMaternKernelGenerator(order)
