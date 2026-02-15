"""Hida-Matern kernel generator subpackage.

Provides runtime symbolic construction of Hida-Matern state-space
kernel matrices for arbitrary smoothness orders using SymPy and
sympy2jax.

This subpackage requires the ``kergen`` extra::

    pip install cvhmax[kergen]
"""

try:
    from .generator import HidaMaternKernelGenerator, make_kernel
except ImportError as _e:
    raise ImportError(
        "kernel_generator requires extra dependencies (sympy, sympy2jax). "
        "Install with:  pip install cvhmax[kergen]"
    ) from _e

__all__ = ["HidaMaternKernelGenerator", "make_kernel"]
