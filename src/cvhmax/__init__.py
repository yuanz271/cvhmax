from ._version import __version__, __version_tuple__  # noqa: F401
from .cvhm import CVHM, lift, project
from .cvi import CVI, Gaussian, Poisson, Params
from .hm import HidaMatern
from .utils import pad_trials, unpad_trials

__all__ = [
    "CVHM",
    "CVI",
    "Gaussian",
    "Poisson",
    "Params",
    "HidaMatern",
    "lift",
    "project",
    "pad_trials",
    "unpad_trials",
]

# Optional: kernel_generator requires the `kergen` extra.
try:
    from .kernel_generator import HidaMaternKernelGenerator, make_kernel

    __all__ += ["HidaMaternKernelGenerator", "make_kernel"]
except ImportError:
    pass
