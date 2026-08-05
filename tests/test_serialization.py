"""Model serialization round-trip tests."""

import json
import subprocess
import sys
import zipfile

import numpy as np
import pytest
from jax import numpy as jnp

from cvhmax.cvhm import CVHM
from cvhmax.hm import HidaMatern


def _model(observation="Gaussian"):
    return CVHM(
        n_components=1,
        dt=0.5,
        kernels=[HidaMatern(sigma=1.2, rho=2.0, omega=0.3, order=1, s=1e-5)],
        observation=observation,
        lr=0.2,
        max_iter=2,
        cvi_iter=1,
    )


def _data():
    return jnp.asarray(np.random.default_rng(0).normal(size=(1, 8, 2)))


def test_unfitted_model_round_trip(tmp_path):
    path = tmp_path / "model.cvhmax"
    model = _model()
    model.save(path)
    restored = CVHM.load(path)

    assert restored.params is None
    assert not hasattr(restored, "posterior")
    assert restored.observation == model.observation
    assert restored.kernels[0] == model.kernels[0]
    assert restored.n_components == model.n_components
    assert restored.dt == model.dt


def test_fitted_gaussian_round_trip_and_infer(tmp_path):
    path = tmp_path / "model.cvhmax"
    model = _model().fit(_data(), random_state=0)
    params_before = tuple(np.asarray(value) for value in (model.params.C, model.params.d, model.params.R))
    model.save(path)
    restored = CVHM.load(path)

    assert not hasattr(restored, "posterior")
    assert not hasattr(restored, "latent")
    for actual, expected in zip(
        (restored.params.C, restored.params.d, restored.params.R), params_before
    ):
        np.testing.assert_array_equal(np.asarray(actual), expected)

    posterior = restored.infer(_data())
    assert posterior[0].shape == (1, 8, 1)
    assert all(np.isfinite(np.asarray(value)).all() for value in posterior)
    np.testing.assert_array_equal(
        np.asarray(restored.params.C), np.asarray(model.params.C)
    )


def test_fitted_poisson_round_trip_with_none_covariance(tmp_path):
    y = jnp.asarray(np.random.default_rng(1).poisson(2.0, size=(1, 5, 2)))
    model = _model("Poisson").fit(y, random_state=0)
    path = tmp_path / "poisson.cvhmax"
    model.save(path)
    restored = CVHM.load(path)

    assert restored.params.R is None
    np.testing.assert_array_equal(
        np.asarray(restored.params.C), np.asarray(model.params.C)
    )


def test_archive_is_self_contained_and_data_only(tmp_path):
    path = tmp_path / "model.cvhmax"
    _model().fit(_data(), random_state=0).save(path)
    with zipfile.ZipFile(path) as archive:
        assert set(archive.namelist()) == {"manifest.json", "params.eqx"}
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["format"] == "cvhmax.CVHM"
        assert manifest["kernels"][0]["class"] == "cvhmax.hm.HidaMatern"
        assert b"pickle" not in archive.read("params.eqx").lower()


def test_invalid_manifest_version_fails(tmp_path):
    path = tmp_path / "model.cvhmax"
    _model().fit(_data(), random_state=0).save(path)
    broken = tmp_path / "broken.cvhmax"
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(broken, "w") as target:
        manifest = json.loads(source.read("manifest.json"))
        manifest["version"] = 99
        target.writestr("manifest.json", json.dumps(manifest))
        target.writestr("params.eqx", source.read("params.eqx"))
    with pytest.raises(ValueError, match="Unsupported serialization version"):
        CVHM.load(broken)


def test_custom_kernel_is_rejected(tmp_path):
    class CustomKernel:
        pass

    model = CVHM(n_components=1, dt=1.0, kernels=[CustomKernel()])
    with pytest.raises(TypeError, match="HidaMatern"):
        model.save(tmp_path / "model.cvhmax")


def test_round_trip_in_fresh_process(tmp_path):
    path = tmp_path / "model.cvhmax"
    _model().fit(_data(), random_state=0).save(path)
    script = """
import sys
import numpy as np
import jax.numpy as jnp
from cvhmax.cvhm import CVHM
model = CVHM.load(sys.argv[1])
assert model.params is not None
assert not hasattr(model, 'posterior')
y = jnp.asarray(np.random.default_rng(0).normal(size=(1, 8, 2)))
posterior = model.infer(y)
assert posterior[0].shape == (1, 8, 1)
print(model.observation, model.n_components)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "Gaussian 1"
