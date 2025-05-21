from jax import tree_util

# import equinox as eqx
import numpy as np
import pytest

from cvhmax import hm


def test_HidaMatern():
    kernel = hm.HidaMatern(1.0, 1.0, 0.0, 0)

    K0 = kernel.K(0.0)
    assert K0.shape == (1, 1)

    dt = 1.0

    kernel.Af(dt)
    kernel.Qf(dt)
    kernel.Ab(dt)
    kernel.Qb(dt)


# def test_composite():
#     # 2 latents
#     # L1: 1 kernel
#     # L2: 2 kernels
#     hyperparams = [[{'sigma': 1., 'rho': 1., 'omega': 0., 'order': 1}], [{'sigma': 1., 'rho': 1., 'omega': 0., 'order': 0}, {'sigma': 1., 'rho': 1., 'omega': 1., 'order': 1}]]
#     hyperspec = [[{'sigma': True, 'rho': True, 'omega': True, 'order': False}], [{'sigma': True, 'rho': True, 'omega': True, 'order': False}, {'sigma': True, 'rho': True, 'omega': True, 'order': False}]]
#     # print(tree_util.tree_structure(hyperparams))
#     hyperdef, hyperflat = tree_util.tree_flatten(hyperparams)
#     # print(hyperflat)
#     # https://docs.kidger.site/equinox/all-of-equinox/
#     # eqx.partition

#     params, static = eqx.partition(hyperparams, hyperspec)
#     # eqx.tree_pprint(params)
#     # eqx.tree_pprint(static)
#     paramflat, paramdef = tree_util.tree_flatten(params)
#     # print(paramdef)
#     # print(paramflat)


def test_Ks():
    kernelparam = {"sigma": 1.0, "rho": 1.0, "omega": 0.0, "order": 0}
    hm.Ks(kernelparam, 1.0)

    kernelparam = {"sigma": 1.0, "rho": 1.0, "omega": 0.0, "order": 1}
    hm.Ks(kernelparam, 1.0)

    kernelparam = {"sigma": 1.0, "rho": 1.0, "omega": 0.0, "order": 2}
    with pytest.raises(NotImplementedError):
        hm.Ks(kernelparam, 1.0)


def test_ssm_repr():
    dt = 1.0
    kernelparams = [
        [{"sigma": 1.0, "rho": 1.0, "omega": 0.0, "order": 1}],
        [
            {"sigma": 1.0, "rho": 1.0, "omega": 0.0, "order": 0},
            {"sigma": 1.0, "rho": 1.0, "omega": 1.0, "order": 1},
        ],
    ]
    Af, Qf, Ab, Qb = hm.ssm_repr(kernelparams, dt)
    # eqx.tree_pprint(Af)
    paramflat, paramdef = tree_util.tree_flatten(Af)
    assert len(paramflat) == 3


def test_mask():
    kernelparams = [
        [{"sigma": 1.0, "rho": 1.0, "omega": 0.0, "order": 1}],
        [
            {"sigma": 1.0, "rho": 1.0, "omega": 0.0, "order": 0},
            {"sigma": 1.0, "rho": 1.0, "omega": 1.0, "order": 1},
        ],
    ]
    M = hm.latent_mask(kernelparams)
    print(M)


def make_mask():
    gp_dimensions = [[1, 2], [3, 4, 5]]
    dx = 0
    lat_dim = len(gp_dimensions)
    ssm_order = sum(sum(gp_dimensions, []))
    H = np.zeros((lat_dim, 2 * ssm_order))

    for i, ldims in enumerate(gp_dimensions):
        for j, gp_dim in enumerate(ldims):
            H[i, dx] = 1
            dx += gp_dim

    print(H)
