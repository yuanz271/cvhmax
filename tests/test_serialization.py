"""Model serialization round-trip tests."""

import json
import subprocess
import sys
import zipfile

import numpy as np
import pytest
from jax import numpy as jnp

from cvhmax.cvhm import CVHM
from cvhmax.estimator import Estimator
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
    np.testing.assert_array_equal(np.asarray(restored.params.alpha), np.asarray(model.params.alpha))

    means, covariances = restored.infer(data)
    assert means.shape == (1, data.shape[1], 1)
    assert covariances.shape == (1, data.shape[1], 1, 1)
    assert np.isfinite(np.asarray(means)).all()
    assert np.isfinite(np.asarray(covariances)).all()


def test_ard_estimator_and_precisions_round_trip(tmp_path):
    data = _data(n_features=4, n_time=24)
    model = _model(
        n_components=1,
        estimator="ARDEstimator",
        estimator_config={"alpha_rate": 1e-3, "iterations": 20},
        max_iter=3,
    ).fit(data, random_state=0)
    path = tmp_path / "ard.cvhmax"
    model.save(path)
    restored = CVHM.load(path)

    assert restored.estimator == "ARDEstimator"
    assert restored.estimator_config == {"alpha_rate": 1e-3, "iterations": 20}
    assert restored.estimator_impl.alpha_rate == 1e-3
    assert restored.estimator_impl.iterations == 20
    np.testing.assert_array_equal(np.asarray(restored.params.alpha), np.asarray(model.params.alpha))
    np.testing.assert_array_equal(np.asarray(restored.params.C), np.asarray(model.params.C))


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


def test_convergence_configuration_round_trip(tmp_path):
    model = _model(tol=0.01, min_iter=3, convergence_patience=4)
    path = tmp_path / "model.cvhmax"
    model.save(path)
    restored = CVHM.load(path)
    assert restored.tol == model.tol
    assert restored.min_iter == model.min_iter
    assert restored.convergence_patience == model.convergence_patience


def test_numpy_and_jax_alpha_config_round_trip(tmp_path):
    """Numpy/JAX alpha values persist as portable JSON without changing shape."""
    kernels = [HidaMatern(order=0) for _ in range(2)]
    cases = [
        (np.asarray([0.25, 0.75]), [0.25, 0.75], [0.25, 0.75]),
        (jnp.asarray([0.25, 0.75]), [0.25, 0.75], [0.25, 0.75]),
        (np.float64(0.5), 0.5, [0.5]),
    ]
    for index, (alpha, portable, stored) in enumerate(cases):
        model = CVHM(
            n_components=2,
            dt=0.5,
            kernels=kernels,
            estimator="RidgeEstimator",
            estimator_config={"alpha": alpha},
        )
        # get_config() is the JSON-compatible configuration protocol. A scalar
        # stays a scalar and an array stays an array, with the same values.
        assert model.get_config()["estimator_config"] == {"alpha": portable}
        json.dumps(model.get_config())

        path = tmp_path / f"alpha_{index}.cvhmax"
        model.save(path)
        restored = CVHM.load(path)
        assert restored.estimator_config == {"alpha": portable}
        np.testing.assert_allclose(
            np.asarray(restored.estimator_impl.alpha), stored
        )


def test_custom_estimator_is_rejected_on_save(tmp_path):
    """A registered user estimator is not part of the serialization format."""

    class ProbeEstimator(Estimator):
        def estimate(self, likelihood, params, y, valid_y, m, V):
            return params, jnp.nan

    try:
        model = _model(estimator="ProbeEstimator")
        assert type(model.estimator_impl) is ProbeEstimator
        with pytest.raises(ValueError, match="Unsupported loading estimator"):
            model.save(tmp_path / "probe.cvhmax")
    finally:
        Estimator.registry.pop("ProbeEstimator", None)


def test_same_named_custom_estimator_is_rejected_on_save(tmp_path):
    """A user subclass registered under a built-in name must not be saved."""
    script = """
import sys
import jax.numpy as jnp
from cvhmax.cvhm import CVHM
from cvhmax.estimator import Estimator
from cvhmax.hm import HidaMatern

class RidgeEstimator(Estimator):
    def estimate(self, likelihood, params, y, valid_y, m, V):
        return params, jnp.nan

model = CVHM(n_components=1, dt=1.0, kernels=[HidaMatern(order=0)],
             estimator="RidgeEstimator")
assert Estimator.registry["RidgeEstimator"] is RidgeEstimator
try:
    model.save(sys.argv[1])
except ValueError:
    sys.exit(0)
sys.exit(1)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "shadow.cvhmax")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_shadowed_builtin_estimator_cannot_be_loaded(tmp_path):
    """Loading must not silently substitute a same-named user estimator."""
    path = tmp_path / "model.cvhmax"
    _model().save(path)
    script = """
import sys
import jax.numpy as jnp
from cvhmax.estimator import Estimator

class RidgeEstimator(Estimator):
    def estimate(self, likelihood, params, y, valid_y, m, V):
        return params, jnp.nan

from cvhmax.cvhm import CVHM

try:
    CVHM.load(sys.argv[1])
except ValueError:
    sys.exit(0)
sys.exit(1)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
