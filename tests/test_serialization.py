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


def _multi_component_model(orders):
    return CVHM(
        n_components=len(orders),
        dt=0.5,
        kernels=[HidaMatern(order=order) for order in orders],
        observation="Gaussian",
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
        # Kernels are HidaMatern by contract; no kernel type field is stored.
        assert "class" not in manifest["kernels"][0]
        assert set(manifest["kernels"][0]) == {"sigma", "rho", "omega", "order", "s"}
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
    with pytest.raises((AttributeError, TypeError)):
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


# ---------------------------------------------------------------------------
# get_config / from_config protocol tests
# ---------------------------------------------------------------------------


class TestGetConfigProtocol:
    """Verify the get_config/from_config protocol on CVHM and HidaMatern."""

    def test_hida_matern_get_config_round_trip(self):
        kernel = HidaMatern(sigma=1.5, rho=50.0, omega=0.0, order=1, s=1e-5)
        config = kernel.get_config()
        assert config == {"sigma": 1.5, "rho": 50.0, "omega": 0.0, "order": 1, "s": 1e-5}
        restored = HidaMatern.from_config(config)
        assert restored == kernel

    def test_hida_matern_from_config_missing_key(self):
        with pytest.raises(ValueError, match="missing required key"):
            HidaMatern.from_config({"sigma": 1.0, "rho": 2.0})

    @pytest.mark.parametrize("order", [0, 1, 2])
    def test_hida_matern_all_supported_orders(self, order):
        kernel = HidaMatern(sigma=1.0, rho=10.0, omega=0.0, order=order, s=1e-5)
        config = kernel.get_config()
        restored = HidaMatern.from_config(config)
        assert restored == kernel

    def test_cvhm_get_config_round_trip(self):
        model = _model()
        config = model.get_config()
        assert config["n_components"] == 1
        assert config["dt"] == 0.5
        assert config["observation"] == "Gaussian"
        assert len(config["kernels"]) == 1
        assert set(config["kernels"][0]) == {"sigma", "rho", "omega", "order", "s"}

        restored = CVHM.from_config(config)
        assert restored.n_components == model.n_components
        assert restored.dt == model.dt
        assert restored.observation == model.observation
        assert restored.lr == model.lr
        assert restored.max_iter == model.max_iter
        assert restored.cvi_iter == model.cvi_iter
        assert restored.params is None
        assert not hasattr(restored, "posterior")
        assert restored.kernels[0] == model.kernels[0]

    def test_cvhm_from_config_missing_key(self):
        with pytest.raises(ValueError, match="missing required key"):
            CVHM.from_config({"n_components": 1})


# ---------------------------------------------------------------------------
# Multiple components and orders
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("orders", [[0], [1], [2], [0, 1], [1, 2]])
def test_round_trip_multi_component_and_orders(tmp_path, orders):
    """Round-trip with multiple components and various HidaMatern orders."""
    n_components = len(orders)
    obs_dim = max(n_components, 2)
    path = tmp_path / "model.cvhmax"
    model = _multi_component_model(orders)
    y = jnp.asarray(np.random.default_rng(0).normal(size=(1, 32, obs_dim)))
    model.fit(y, random_state=0)
    model.save(path)
    restored = CVHM.load(path)

    assert restored.n_components == n_components
    assert len(restored.kernels) == n_components
    for rk, ok in zip(restored.kernels, model.kernels):
        assert rk.order == ok.order
        np.testing.assert_array_equal(np.asarray(rk.sigma), np.asarray(ok.sigma))
    np.testing.assert_array_equal(
        np.asarray(restored.params.C), np.asarray(model.params.C)
    )
    # Infer after load
    m, V = restored.infer(y)
    assert m.shape == (1, 32, n_components)


# ---------------------------------------------------------------------------
# HidaMatern subclass round-trip (restored as base HidaMatern)
# ---------------------------------------------------------------------------


def test_hida_matern_subclass_is_restored_as_base(tmp_path):
    class CustomHidaMatern(HidaMatern):
        pass

    kernel = CustomHidaMatern(sigma=1.0, rho=10.0, omega=0.0, order=0, s=1e-5)
    model = CVHM(n_components=1, dt=0.5, kernels=[kernel], observation="Gaussian")
    y = jnp.asarray(np.random.default_rng(0).normal(size=(1, 8, 2)))
    model.fit(y, random_state=0)

    path = tmp_path / "custom.cvhmax"
    model.save(path)
    restored = CVHM.load(path)

    assert isinstance(restored.kernels[0], HidaMatern)
    assert not isinstance(restored.kernels[0], CustomHidaMatern)
    assert restored.kernels[0] == HidaMatern(sigma=1.0, rho=10.0, omega=0.0, order=0, s=1e-5)


# ---------------------------------------------------------------------------
# Malformed archive error cases
# ---------------------------------------------------------------------------


def test_unknown_observation_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unsupported observation"):
        CVHM(
            n_components=1,
            dt=1.0,
            kernels=[HidaMatern()],
            observation="UnknownObs",
        )


def test_unsupported_kernel_order_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unsupported Mat\u00e9rn order"):
        HidaMatern(order=3)


def test_missing_manifest_member_rejected(tmp_path):
    path = tmp_path / "missing.cvhmax"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("params.eqx", b"garbage")
    with pytest.raises(ValueError, match="missing manifest.json"):
        CVHM.load(path)


def test_unsafe_archive_path_rejected(tmp_path):
    path = tmp_path / "unsafe.cvhmax"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../manifest.json", json.dumps({"format": "cvhmax.CVHM"}))
    with pytest.raises(ValueError, match="Unsafe archive member"):
        CVHM.load(path)


def test_duplicate_archive_member_rejected(tmp_path):
    path = tmp_path / "dup.cvhmax"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps({"format": "cvhmax.CVHM"}))
        archive.writestr("manifest.json", json.dumps({"format": "cvhmax.CVHM"}))
    with pytest.raises(ValueError, match="duplicate"):
        CVHM.load(path)


def test_missing_params_eqx_when_present_is_true(tmp_path):
    path = tmp_path / "mismatch.cvhmax"
    manifest = {
        "format": "cvhmax.CVHM",
        "version": 1,
        "model": {
            "n_components": 1,
            "dt": 1.0,
            "observation": "Gaussian",
            "lr": 0.1,
            "max_iter": 10,
            "cvi_iter": 5,
        },
        "kernels": [{"sigma": 1.0, "rho": 10.0, "omega": 0.0, "order": 0, "s": 1e-5}],
        "params": {"present": True, "type": "cvhmax.cvi.Params", "R_is_none": False, "arrays": {"C": {"shape": [2, 1], "dtype": "float64"}, "d": {"shape": [2], "dtype": "float64"}, "R": {"shape": [2, 2], "dtype": "float64"}}},
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
    with pytest.raises(ValueError, match="params.eqx presence does not match"):
        CVHM.load(path)


def _valid_manifest_with_params() -> dict:
    return {
        "format": "cvhmax.CVHM",
        "version": 1,
        "model": {
            "n_components": 1,
            "dt": 1.0,
            "observation": "Gaussian",
            "lr": 0.1,
            "max_iter": 10,
            "cvi_iter": 5,
        },
        "kernels": [{"sigma": 1.0, "rho": 10.0, "omega": 0.0, "order": 0, "s": 1e-5}],
        "params": {"present": True, "type": "cvhmax.cvi.Params", "R_is_none": False, "arrays": {"C": {"shape": [2, 1], "dtype": "float64"}, "d": {"shape": [2], "dtype": "float64"}, "R": {"shape": [2, 2], "dtype": "float64"}}},
    }


def test_malformed_param_dtype_rejected(tmp_path):
    path = tmp_path / "bad_dtype.cvhmax"
    manifest = _valid_manifest_with_params()
    manifest["params"]["arrays"]["C"]["dtype"] = "not-a-real-dtype"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
    with pytest.raises(ValueError, match="Invalid parameter dtype"):
        CVHM.load(path)


def test_malformed_param_shape_rejected(tmp_path):
    path = tmp_path / "bad_shape.cvhmax"
    manifest = _valid_manifest_with_params()
    manifest["params"]["arrays"]["d"]["shape"] = [-1]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
    with pytest.raises(ValueError, match="must be at least 0"):
        CVHM.load(path)


def test_params_r_is_none_disagreement_rejected(tmp_path):
    path = tmp_path / "r_mismatch.cvhmax"
    manifest = _valid_manifest_with_params()
    manifest["params"]["R_is_none"] = True  # arrays.R is not None
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
    with pytest.raises(ValueError, match="disagrees with params.arrays.R"):
        CVHM.load(path)


def test_manifest_size_limit_rejected(tmp_path, monkeypatch):
    import cvhmax.serialization as serialization_module

    monkeypatch.setattr(serialization_module, "_MAX_MANIFEST_BYTES", 16)
    path = tmp_path / "model.cvhmax"
    _model().save(path)  # manifest.json is much larger than 16 bytes
    with pytest.raises(ValueError, match="manifest.json is too large"):
        CVHM.load(path)


def test_params_payload_size_limit_rejected(tmp_path, monkeypatch):
    import cvhmax.serialization as serialization_module

    monkeypatch.setattr(serialization_module, "_MAX_PARAMS_BYTES", 16)
    path = tmp_path / "model.cvhmax"
    model = _model().fit(_data(), random_state=0)
    with pytest.raises(ValueError, match="Serialized parameters are too large"):
        model.save(path)
    assert not path.exists()


# ---------------------------------------------------------------------------
# Overwrite and cleanup
# ---------------------------------------------------------------------------


def test_save_overwrite_works(tmp_path):
    path = tmp_path / "model.cvhmax"
    model = _model().fit(_data(), random_state=0)
    model.save(path)
    assert path.exists()
    # Save again to the same path (should overwrite cleanly)
    model.save(path)
    assert path.exists()
    restored = CVHM.load(path)
    assert restored.params is not None


def test_save_to_existing_unfitted_then_fitted(tmp_path):
    """Unfitted save, then fitted save to same path should work."""
    path = tmp_path / "model.cvhmax"
    _model().save(path)
    assert path.exists()
    _model().fit(_data(), random_state=0).save(path)
    restored = CVHM.load(path)
    assert restored.params is not None


def test_failed_save_does_not_overwrite_destination(tmp_path):
    """A failed save must not corrupt or overwrite an existing file."""
    path = tmp_path / "model.cvhmax"
    path.write_text("original content")
    model = CVHM(n_components=1, dt=1.0, kernels=[HidaMatern()])
    model.observation = "UnknownObs"
    with pytest.raises(ValueError, match="Unsupported observation"):
        model.save(path)
    assert path.read_text() == "original content"


# ---------------------------------------------------------------------------
# Inference does not mutate params
# ---------------------------------------------------------------------------


def test_infer_does_not_mutate_params(tmp_path):
    path = tmp_path / "model.cvhmax"
    model = _model().fit(_data(), random_state=0)
    C_before = np.asarray(model.params.C).copy()
    model.save(path)
    restored = CVHM.load(path)
    m, V = restored.infer(_data())
    C_after = np.asarray(restored.params.C)
    np.testing.assert_array_equal(C_after, C_before)
