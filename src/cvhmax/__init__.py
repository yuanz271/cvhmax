from .cvhm import CVHM
from .cvi import CVI, Gaussian, Poisson, Params
from .hm import HidaMatern

__all__ = [
    "CVHM",
    "CVI",
    "Gaussian",
    "Poisson",
    "Params",
    "HidaMatern",
]

# Optional: kernel_generator requires the `kergen` extra.
try:
    from .kernel_generator import HidaMaternKernelGenerator, make_kernel

    __all__ += ["HidaMaternKernelGenerator", "make_kernel"]
except ImportError:
    pass
