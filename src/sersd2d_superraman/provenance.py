"""Content identities and strict JSON output used by every stage."""

from pathlib import Path
import hashlib
import json
import os
import platform
import tempfile
import numpy as np
from . import __version__


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_hash(values):
    a = np.ascontiguousarray(values)
    digest = hashlib.sha256(f"{a.dtype.str}:{a.shape}".encode())
    digest.update(a.tobytes())
    return digest.hexdigest()


def json_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Refuse invalid scientific values rather than emitting nonstandard NaN JSON.
    payload = json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, name = tempfile.mkstemp(prefix=".json-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)
    return path


def runtime(backend=None):
    out = {
        "package": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    if backend == "tensorflow":
        import tensorflow as tf

        out.update(
            tensorflow=tf.__version__,
            keras=tf.keras.__version__,
            gpus=[x.name for x in tf.config.list_physical_devices("GPU")],
        )
    elif backend == "torch":
        import torch

        out.update(torch=str(torch.__version__), cuda_available=torch.cuda.is_available())
    return out


def fresh_directory(path):
    """Do not mix a new experiment with a previous run's artifacts."""
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output directory is nonempty: {path}. Choose a new run directory.")
    path.mkdir(parents=True, exist_ok=True)
    return path
