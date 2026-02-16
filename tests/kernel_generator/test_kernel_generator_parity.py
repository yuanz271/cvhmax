"""Parity tests against the PyTorch reference implementation.

Requires torch and the reference code under hida_matern_gp_lvms/.
"""

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from cvhmax.kernel_generator import make_kernel


_REF_ROOT = Path(__file__).resolve().parents[2] / "hida_matern_gp_lvms"
if _REF_ROOT.is_dir() and str(_REF_ROOT) not in sys.path:
    sys.path.insert(0, str(_REF_ROOT))


def _torch_and_ref_available():
    try:
        import importlib
        import torch  # noqa: F401

        importlib.import_module(
            "hida_matern_kernel_generator.hm_ss_kernels.hida_M_1.hida_M_1_K_hat"
        )
        return True
    except ImportError:
        return False


requires_ref = pytest.mark.skipif(
    not _torch_and_ref_available(),
    reason="torch or hida_matern_gp_lvms reference code not available",
)


@requires_ref
class TestParityPyTorchReference:
    """Cross-implementation parity tests against the PyTorch reference."""

    @pytest.mark.parametrize("order", [1, 2, 3])
    @pytest.mark.parametrize("tau_val", [0.0, 0.5, 1.0])
    def test_K_hat_parity(self, order, tau_val):
        """Generated K_hat should match PyTorch reference."""
        import importlib
        import torch

        mod = importlib.import_module(
            f"hida_matern_kernel_generator.hm_ss_kernels."
            f"hida_M_{order}.hida_M_{order}_K_hat"
        )

        log_sigma = torch.tensor(0.0, dtype=torch.float64)
        log_ls = torch.tensor(0.0, dtype=torch.float64)
        log_b = torch.tensor(0.0, dtype=torch.float64)

        sigma_val = float(torch.nn.functional.softplus(log_sigma))
        rho_val = float(torch.nn.functional.softplus(log_ls))
        omega_val = float(torch.nn.functional.softplus(log_b))

        K_ref = mod.create_K_hat(
            tau_val, log_sigma, log_ls, log_b, dtype=torch.complex128
        )
        K_ref = K_ref.numpy()

        gen = make_kernel(order)
        K_jax = gen.create_K_hat(
            jnp.array(tau_val),
            jnp.array(sigma_val),
            jnp.array(rho_val),
            jnp.array(omega_val),
        )
        K_jax = np.asarray(K_jax)

        npt.assert_allclose(K_jax, K_ref, atol=1e-8)
