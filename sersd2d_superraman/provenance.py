"""Content identities and strict JSON output used by every stage."""

from pathlib import Path
from contextlib import contextmanager
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


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def read_json(path):
    """Read standard JSON without ambiguous duplicate fields or NaN/Infinity."""
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_invalid_constant,
    )


@contextmanager
def atomic_output(path, overwrite=False):
    """Publish a closed file on the same filesystem; failures leave the target intact.

    A hard link provides an atomic no-clobber commit for new artifacts. Local
    NTFS/ext4 support this; filesystems without hard links fail without publishing.
    Existing targets are replaced only for explicitly mutable manifests/histories.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and path.exists():
        raise ValueError(f"Refusing to overwrite an existing artifact: {path}")
    descriptor, name = tempfile.mkstemp(prefix=".pending-", suffix=path.suffix, dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        yield temporary
        if overwrite:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise ValueError(f"Refusing to overwrite an existing artifact: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def write_npz(path, **arrays):
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Array artifacts require a .npz output")
    with atomic_output(path) as temporary:
        np.savez_compressed(temporary, **arrays)
    return path


def publish_checkpoint(directory, weight_name, writer, metadata):
    """Commit immutable weight bytes before the metadata pointer.

    Serialization or metadata failures preserve the previously published pair.
    Only the previous managed weight file is removed after a successful commit.
    Checkpoint directories have one writer; copy a bundle before concurrent use.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    old_path = directory / "metadata.json"
    old = read_json(old_path) if old_path.exists() else {}
    # Validate serializability before a potentially expensive framework save.
    json.dumps(metadata, allow_nan=False)
    with tempfile.TemporaryDirectory(prefix=".checkpoint-", dir=directory) as staging:
        temporary = Path(staging) / weight_name
        writer(temporary)
        digest = file_hash(temporary)
        suffix = ".weights.h5" if weight_name.endswith(".weights.h5") else ".pt"
        name = f"model-{digest}{suffix}"
        weights = directory / name
        published = {**metadata, "weights": name, "weights_sha256": digest}
        already_present = weights.exists()
        if already_present:
            if file_hash(weights) != digest:
                raise ValueError("Existing content-addressed weight file is damaged")
        else:
            os.replace(temporary, weights)
        try:
            write_json(old_path, published)
        except BaseException:
            # Cancellation can arrive just after the atomic metadata replacement.
            # Never remove bytes that the now-committed pointer already references.
            if not already_present:
                try:
                    committed = old_path.exists() and read_json(old_path).get("weights") == name
                except (OSError, ValueError):
                    committed = True  # Preserve bytes when commit status cannot be read.
                if not committed:
                    weights.unlink(missing_ok=True)
            raise
    previous = old.get("weights")
    if (
        isinstance(previous, str)
        and previous != name
        and Path(previous).name == previous
        and (previous.startswith("model-") or previous in ("model.pt", "model.weights.h5"))
    ):
        # Cleanup cannot turn a committed checkpoint into a failed training step.
        try:
            (directory / previous).unlink(missing_ok=True)
        except OSError:
            pass
    return published


def implementation_identity():
    """Bind cached work to the actual package source, including an editable checkout."""
    root = Path(__file__).parent
    return json_hash({p.relative_to(root).as_posix(): file_hash(p) for p in sorted(root.rglob("*.py"))})


@contextmanager
def exclusive_lock(path):
    path = Path(path)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise ValueError(
            "Another stage is running or interrupted; wait or select a new run directory"
        ) from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        path.unlink(missing_ok=True)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Refuse invalid scientific values rather than emitting nonstandard NaN JSON.
    payload = json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with atomic_output(path, overwrite=True) as temporary:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
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
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Output directory is nonempty: {path}. Choose a new run directory.")
    path.mkdir(parents=True, exist_ok=True)
    return path
