"""Backend-free validation of portable checkpoint roles and preprocessing contracts."""

from pathlib import Path
from .data.registry import SOURCE_DEVICES, registry_hash
from .preprocessing import CLASSIFIER_TRANSFORM
from .provenance import file_hash, read_json


def checkpoint_metadata(directory, kind=None, device=None):
    directory = Path(directory).resolve()
    meta = read_json(directory / "metadata.json")
    required = {
        "kind",
        "schema_version",
        "config",
        "architecture_id",
        "source_device",
        "input_layout",
        "input_length",
        "class_registry_sha256",
        "parameters",
        "weights",
        "weights_sha256",
        "training",
    }
    if not required.issubset(meta):
        raise ValueError("Checkpoint metadata is incomplete")
    if meta["schema_version"] != 1 or meta["kind"] not in ("classification", "translation"):
        raise ValueError("Unsupported checkpoint schema or kind")
    if kind is not None and meta["kind"] != kind:
        raise ValueError("Checkpoint role mismatch")
    if device is not None and meta["source_device"] != device:
        raise ValueError("Checkpoint source device mismatch")
    if meta["input_length"] != 1400 or meta["class_registry_sha256"] != registry_hash():
        raise ValueError("Checkpoint spectral length/class mapping mismatch")
    if type(meta["parameters"]) is not int or meta["parameters"] < 1:
        raise ValueError("Invalid checkpoint parameter count")
    if not isinstance(meta["config"], dict) or not isinstance(meta["training"], dict):
        raise ValueError("Checkpoint configuration/training must be objects")
    if meta["training"].get("status") not in (
        "trained",
        "untrained_initialization",
        "converted_historical_checkpoint",
    ):
        raise ValueError("Unknown checkpoint training status")
    if meta["kind"] == "classification":
        if (
            meta["source_device"] != "Renishaw"
            or meta["input_layout"] != "N,1,L"
            or meta.get("input_transform") != CLASSIFIER_TRANSFORM
        ):
            raise ValueError("Classifier preprocessing/device contract mismatch")
    elif (
        meta["source_device"] not in SOURCE_DEVICES
        or meta.get("target_device") != "Renishaw"
        or meta["input_layout"] != "N,L,1"
        or meta.get("input_range") != [0, 1]
        or meta.get("output_head") != "out"
        or meta.get("output_range_policy") != "linear_unclipped"
    ):
        raise ValueError("Translator input/target/output contract mismatch")
    name = meta["weights"]
    if not isinstance(name, str) or Path(name).name != name or name in ("", ".", ".."):
        raise ValueError("Checkpoint weight name must be a simple filename")
    path = (directory / name).resolve()
    if not path.is_relative_to(directory) or file_hash(path) != meta["weights_sha256"]:
        raise ValueError("Checkpoint weight path/hash mismatch")
    return meta
