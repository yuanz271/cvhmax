"""Hida-Matern kernel generator subpackage.

Provides runtime symbolic construction of Hida-Matern state-space
kernel matrices for arbitrary smoothness orders using SymPy and
sympy2jax.
"""

from .generator import HidaMaternKernelGenerator, make_kernel

__all__ = ["HidaMaternKernelGenerator", "make_kernel"]
