"""Positive model serialization and convergence tests."""

import json
import subprocess
import sys
import zipfile

import numpy as np
from jax import numpy as jnp

from cvhmax.cvhm import CVHM, readout_change
from cvhmax.cvi import Params
from cvhmax.hm import HidaMatern


def _model(observation="Gaussian", **kwargs):
    config = dict(
        n_components=1,
        dt=0.5,
        kernels=[HidaMatern(sigma=1.2, rho=2.0, omega=0.3, order=1, s=1e-5)],
        observation=observation,
        lr=0.2,
        max_iter=10,
        cvi_iter=1,
    )
    config.update(kwargs)
    return CVHM(**config)


def _data(n_components=1, n_features=2, n_time=16):
    return jnp.asarray(
        np.random.default_rng(0).normal(size=(1, n_time, n_features))
    )


def test_unfitted_model_round_trip(tmp_path):
    model = _model()
    path = tmp_path / "model.cvhmax"
    model.save(path)
    restored = CVHM.load(path)

    assert restored.params is None
    assert restored.get_config() == model.get_config()
    assert restored.kernels[0] == model.kernels[0]


def test_fitted_gaussian_round_trip_and_infer(tmp_path):
    data = _data()
    model = _model().fit(data, random_state=0)
    path = tmp_path / "model.cvhmax"
    model.save(path)
    restored = CVHM.load(path)

    np.testing.assert_array_equal(np.asarray(restored.params.C), np.asarray(model.params.C))
    np.testing.assert_array_equal(np.asarray(restored.params.d), np.asarray(model.params.d))
    np.testing.assert_array_equal(np.asarray(restored.params.R), np.asarray(model.params.R))

    means, covariances = restored.infer(data)
    assert means.shape == (1, data.shape[1], 1)
    assert covariances.shape == (1, data.shape[1], 1, 1)
    assert np.isfinite(np.asarray(means)).all()
    assert np.isfinite(np.asarray(covariances)).all()


def test_fitted_poisson_round_trip_with_sentinel_r(tmp_path):
    data = jnp.asarray(np.random.default_rng(1).poisson(2.0, size=(1, 8, 2)))
    model = _model("Poisson").fit(data, random_state=0)
    path = tmp_path / "poisson.cvhmax"
    model.save(path)
    restored = CVHM.load(path)

    assert np.asarray(restored.params.R).shape == ()
    assert float(restored.params.R) == 0.0
    np.testing.assert_array_equal(np.asarray(restored.params.C), np.asarray(model.params.C))
    means, covariances = restored.infer(data)
    assert np.isfinite(np.asarray(means)).all()
    assert np.isfinite(np.asarray(covariances)).all()


def test_model_config_round_trip():
    model = _model(tol=0.01, min_iter=3, convergence_patience=4)
    restored = CVHM.from_config(model.get_config())
    assert restored.get_config() == model.get_config()


def test_hida_matern_config_round_trip():
    for order in (0, 1, 2):
        kernel = HidaMatern(
            sigma=1.5, rho=10.0, omega=0.4, order=order, s=1e-5
        )
        assert HidaMatern.from_config(kernel.get_config()) == kernel


def test_multiple_components_and_supported_orders_round_trip(tmp_path):
    orders = (0, 1, 2)
    model = CVHM(
        n_components=3,
        dt=0.5,
        kernels=[HidaMatern(order=order) for order in orders],
        observation="Gaussian",
        max_iter=2,
        min_iter=2,
    )
    data = _data(n_components=3, n_features=3)
    model.fit(data, random_state=0)
    path = tmp_path / "multi.cvhmax"
    model.save(path)
    restored = CVHM.load(path)

    assert tuple(kernel.order for kernel in restored.kernels) == orders
    means, covariances = restored.infer(data)
    assert means.shape[-1] == 3
    assert covariances.shape[-2:] == (3, 3)


def test_archive_is_single_model_artifact(tmp_path):
    path = tmp_path / "model.cvhmax"
    _model().fit(_data(), random_state=0).save(path)
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        members = set(archive.namelist())
    assert manifest["format"] == "cvhmax.CVHM"
    assert members == {"manifest.json", "params.eqx"}


def test_round_trip_in_fresh_process(tmp_path):
    path = tmp_path / "model.cvhmax"
    _model().fit(_data(), random_state=0).save(path)
    script = """
import sys
import numpy as np
import jax.numpy as jnp
from cvhmax.cvhm import CVHM
model = CVHM.load(sys.argv[1])
y = jnp.asarray(np.random.default_rng(0).normal(size=(1, 16, 2)))
means, covariances = model.infer(y)
assert means.shape == (1, 16, 1)
assert covariances.shape == (1, 16, 1, 1)
print(model.observation, model.n_components)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "Gaussian 1"


def test_readout_change_is_invariant_to_glm_rotation():
    C = jnp.asarray([[1.0, 0.0], [0.0, 2.0], [1.0, -1.0]])
    theta = 0.37
    rotation = jnp.asarray(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    old = Params(C=C, d=jnp.zeros(3), R=jnp.eye(3))
    new = Params(C=C @ rotation, d=old.d, R=old.R)
    assert float(readout_change(new, old)) < 1e-6


def test_convergence_stops_a_fit_and_reports_diagnostics():
    model = _model().fit(_data(), random_state=0)
    assert model.converged_ is True
    assert model.min_iter <= model.n_iter_ < model.max_iter


def test_convergence_configuration_round_trip(tmp_path):
    model = _model(tol=0.01, min_iter=3, convergence_patience=4)
    path = tmp_path / "model.cvhmax"
    model.save(path)
    restored = CVHM.load(path)
    assert restored.tol == model.tol
    assert restored.min_iter == model.min_iter
    assert restored.convergence_patience == model.convergence_patience
