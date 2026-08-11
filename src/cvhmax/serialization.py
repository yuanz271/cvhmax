"""Data-only serialization for fitted CVHM models."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any
import zipfile

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from .cvi import Params

FORMAT = "cvhmax.CVHM"
VERSION = 1
_MANIFEST = "manifest.json"
_PARAMS = "params.eqx"
_MAX_MANIFEST_BYTES = 1 << 20
_MAX_PARAMS_BYTES = 1 << 30
_SUPPORTED_OBSERVATIONS = {"Gaussian", "Poisson"}


def _array_metadata(array: Any) -> dict[str, Any]:
    value = np.asarray(array)
    return {"shape": list(value.shape), "dtype": str(value.dtype)}


def _finite_float(value: Any, name: str) -> float:
    array = np.asarray(value)
    if array.ndim != 0 or not np.isfinite(array):
        raise ValueError(f"{name} must be a finite scalar")
    return float(array)


def _params_manifest(params: Params | None) -> dict[str, Any]:
    if params is None:
        return {"present": False}
    if not isinstance(params, Params):
        raise TypeError("Only cvhmax.cvi.Params can be serialized")
    return {
        "present": True,
        "type": "cvhmax.cvi.Params",
        "arrays": {
            "C": _array_metadata(params.C),
            "d": _array_metadata(params.d),
            "R": _array_metadata(params.R),
        },
    }


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def build_manifest(model: Any) -> dict[str, Any]:
    """Build the serialization manifest using the get_config() protocol.

    Parameters
    ----------
    model : CVHM
        The model to serialize.

    Returns
    -------
    dict
        Manifest dictionary ready for JSON serialization.
    """
    if model.observation not in _SUPPORTED_OBSERVATIONS:
        raise ValueError(
            f"Unsupported observation model {model.observation!r}; "
            "only Gaussian and Poisson can be serialized"
        )
    config = model.get_config()
    if len(config["kernels"]) != config["n_components"]:
        raise ValueError("CVHM requires one kernel per component")
    return {
        "format": FORMAT,
        "version": VERSION,
        "model": {
            "n_components": _integer(config["n_components"], "model.n_components", minimum=1),
            "dt": _finite_float(config["dt"], "model.dt"),
            "observation": config["observation"],
            "lr": _finite_float(config["lr"], "model.lr"),
            "max_iter": _integer(config["max_iter"], "model.max_iter"),
            "cvi_iter": _integer(config["cvi_iter"], "model.cvi_iter"),
            "tol": _finite_float(config["tol"], "model.tol"),
            "min_iter": _integer(config["min_iter"], "model.min_iter"),
            "convergence_patience": _integer(
                config["convergence_patience"], "model.convergence_patience"
            ),
        },
        "kernels": config["kernels"],
        "params": _params_manifest(model.params),
    }


def _serialize_params(params: Params) -> bytes:
    buffer = io.BytesIO()
    eqx.tree_serialise_leaves(buffer, params)
    return buffer.getvalue()


def _validate_params_shapes(params: Params, n_components: int) -> None:
    C = np.asarray(params.C)
    d = np.asarray(params.d)
    if C.ndim != 2 or C.shape[1] != n_components:
        raise ValueError("Params.C shape is incompatible with n_components")
    if d.ndim != 1 or d.shape[0] != C.shape[0]:
        raise ValueError("Params.d shape is incompatible with Params.C")
    R = np.asarray(params.R)
    if R.ndim > 0:
        if R.shape != (C.shape[0], C.shape[0]):
            raise ValueError("Params.R shape is incompatible with Params.C")


def _deserialize_params(payload: bytes, metadata: dict[str, Any]) -> Params:
    arrays = metadata.get("arrays")
    if not isinstance(arrays, dict):
        raise ValueError("params.arrays must be an object")

    def skeleton(name: str):
        item = arrays.get(name)
        if not isinstance(item, dict) or not isinstance(item.get("shape"), list):
            raise ValueError(f"Invalid parameter metadata for {name}")
        try:
            shape = tuple(int(axis) for axis in item["shape"])
            dtype = np.dtype(item["dtype"])
        except (TypeError, ValueError, KeyError) as error:
            raise ValueError(f"Invalid parameter metadata for {name}") from error
        if any(axis < 0 for axis in shape):
            raise ValueError(f"Invalid parameter shape for {name}")
        return jnp.zeros(shape, dtype=dtype)

    params = Params(C=skeleton("C"), d=skeleton("d"), R=skeleton("R"))
    try:
        return eqx.tree_deserialise_leaves(io.BytesIO(payload), params)
    except Exception as error:
        raise ValueError("Could not restore params.eqx") from error


def _validate_archive(archive: zipfile.ZipFile) -> None:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise ValueError("Archive contains duplicate member names")
    for info in infos:
        path = Path(info.filename)
        if path.is_absolute() or ".." in path.parts or "\\" in info.filename:
            raise ValueError(f"Unsafe archive member: {info.filename!r}")
        if info.flag_bits & 1:
            raise ValueError(f"Encrypted archive member is not supported: {info.filename!r}")
        if info.file_size > _MAX_PARAMS_BYTES:
            raise ValueError(f"Archive member is too large: {info.filename!r}")
    if _MANIFEST not in names:
        raise ValueError("Archive is missing manifest.json")


def _read_json(archive: zipfile.ZipFile) -> dict[str, Any]:
    info = archive.getinfo(_MANIFEST)
    if info.file_size > _MAX_MANIFEST_BYTES:
        raise ValueError("manifest.json is too large")
    try:
        manifest = json.loads(archive.read(_MANIFEST))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Invalid manifest.json") from error
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must contain an object")
    return manifest


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("format") != FORMAT:
        raise ValueError("Invalid serialization format")
    if manifest.get("version") != VERSION:
        raise ValueError(f"Unsupported serialization version: {manifest.get('version')!r}")
    model = manifest.get("model")
    kernels = manifest.get("kernels")
    params = manifest.get("params")
    if not isinstance(model, dict) or not isinstance(kernels, list) or not isinstance(params, dict):
        raise ValueError("Manifest has invalid model, kernels, or params metadata")
    for key in (
        "n_components",
        "dt",
        "observation",
        "lr",
        "max_iter",
        "cvi_iter",
        "tol",
        "min_iter",
        "convergence_patience",
    ):
        if key not in model:
            raise ValueError(f"Manifest is missing model.{key}")
    _integer(model["n_components"], "model.n_components", minimum=1)
    _finite_float(model["dt"], "model.dt")
    _finite_float(model["lr"], "model.lr")
    _integer(model["max_iter"], "model.max_iter")
    _integer(model["cvi_iter"], "model.cvi_iter")
    _finite_float(model["tol"], "model.tol")
    _integer(model["min_iter"], "model.min_iter", minimum=1)
    _integer(model["convergence_patience"], "model.convergence_patience", minimum=1)
    if model["min_iter"] > model["max_iter"]:
        raise ValueError("model.min_iter must not exceed model.max_iter")
    if model["observation"] not in _SUPPORTED_OBSERVATIONS:
        raise ValueError(f"Unsupported observation model: {model['observation']!r}")
    if len(kernels) != model["n_components"]:
        raise ValueError("Manifest kernel count does not match n_components")
    for index, kernel in enumerate(kernels):
        if not isinstance(kernel, dict):
            raise ValueError(f"Manifest kernel {index} must be an object")
        required = ("sigma", "rho", "omega", "order", "s")
        for key in required:
            if key not in kernel:
                raise ValueError(f"Manifest is missing kernels[{index}].{key}")
        _finite_float(kernel["sigma"], f"kernels[{index}].sigma")
        _finite_float(kernel["rho"], f"kernels[{index}].rho")
        _finite_float(kernel["omega"], f"kernels[{index}].omega")
        _finite_float(kernel["s"], f"kernels[{index}].s")
        _integer(kernel["order"], f"kernels[{index}].order")
        if kernel["order"] not in (0, 1, 2):
            raise ValueError("Only HidaMatern orders 0, 1, and 2 can be serialized")
    if not isinstance(params.get("present"), bool):
        raise ValueError("params.present must be a boolean")
    if not params["present"]:
        if set(params) != {"present"}:
            raise ValueError("Unfitted params metadata has unexpected fields")
        return
    if params.get("type") != "cvhmax.cvi.Params":
        raise ValueError(f"Unsupported parameter type: {params.get('type')!r}")
    arrays = params.get("arrays")
    if not isinstance(arrays, dict) or set(arrays) != {"C", "d", "R"}:
        raise ValueError("params.arrays must contain exactly C, d, and R")
    for name, metadata in arrays.items():
        if not isinstance(metadata, dict) or not isinstance(metadata.get("shape"), list):
            raise ValueError(f"Invalid parameter metadata for {name}")
        _dtype = metadata.get("dtype")
        if not isinstance(_dtype, str):
            raise ValueError(f"Invalid parameter dtype for {name}")
        try:
            dtype = np.dtype(_dtype)
        except TypeError as error:
            raise ValueError(f"Invalid parameter dtype for {name}") from error
        if dtype.hasobject:
            raise ValueError(f"Object dtype is not supported for parameter {name}")
        for axis in metadata["shape"]:
            _integer(axis, f"params.{name}.shape", minimum=0)


def _validate_builtin_observation(observation: str) -> None:
    if observation not in _SUPPORTED_OBSERVATIONS:
        raise ValueError(
            f"Unsupported observation model {observation!r}; "
            "only Gaussian and Poisson can be serialized"
        )


def save(model: Any, path: str | os.PathLike[str]) -> None:
    """Save a CVHM model to a ZIP archive.

    Parameters
    ----------
    model : CVHM
        Model to save. Must have a ``get_config()`` method and built-in
        observation model.
    path : str or os.PathLike
        Destination path for the archive.
    """
    _validate_builtin_observation(model.observation)
    manifest = build_manifest(model)
    params_payload = None if model.params is None else _serialize_params(model.params)
    if params_payload is not None and len(params_payload) > _MAX_PARAMS_BYTES:
        raise ValueError("Serialized parameters are too large")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False
        ) as file:
            temporary = Path(file.name)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(_MANIFEST, json.dumps(manifest, allow_nan=False, indent=2))
            if params_payload is not None:
                archive.writestr(_PARAMS, params_payload)
        os.replace(temporary, destination)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def load(path: str | os.PathLike[str]) -> Any:
    """Load a CVHM model from a ZIP archive.

    Parameters
    ----------
    path : str or os.PathLike
        Path to the archive created by :func:`save`.

    Returns
    -------
    CVHM
        Reconstructed model with fitted readout parameters (if present).
        Posterior and latent caches are not restored.
    """
    from .cvhm import CVHM

    try:
        archive = zipfile.ZipFile(path, "r")
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError(f"Invalid CVHM archive: {path}") from error
    with archive:
        _validate_archive(archive)
        manifest = _read_json(archive)
        _validate_manifest(manifest)
        params_metadata = manifest["params"]
        has_params = bool(params_metadata.get("present"))
        has_payload = _PARAMS in archive.namelist()
        if has_params != has_payload:
            raise ValueError("params.eqx presence does not match manifest metadata")

        model_config = manifest["model"]
        model_config["kernels"] = manifest["kernels"]
        model = CVHM.from_config(model_config)
        if has_params:
            model.params = _deserialize_params(archive.read(_PARAMS), params_metadata)
            _validate_params_shapes(model.params, model.n_components)
        return model