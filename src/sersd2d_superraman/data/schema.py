"""Canonical float32 [N,1400] spectra and pickle-free portable interchange."""

from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np
from .registry import canonical_device
from ..provenance import array_hash, file_hash

LENGTH = 1400


@dataclass
class SpectrumBatch:
    spectra: np.ndarray
    source_device: str
    sample_ids: np.ndarray | None = None
    labels: np.ndarray | None = None
    axis_cm1: np.ndarray | None = None
    groups: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def validate(self, normalized=True):
        x = np.asarray(self.spectra)
        if x.ndim != 2 or x.shape[1] != LENGTH or len(x) == 0:
            raise ValueError(f"spectra must be nonempty [N,{LENGTH}], got {x.shape}")
        if not np.issubdtype(x.dtype, np.number) or np.iscomplexobj(x):
            raise ValueError("Spectra must be real numeric values")
        x = np.ascontiguousarray(x, dtype=np.float32)
        if not np.isfinite(x).all():
            raise ValueError("Spectra contain NaN or infinity")
        if np.any(np.ptp(x, axis=1) <= 1e-8):
            raise ValueError("Constant or effectively constant spectra cannot be normalized")
        if normalized and (x.min() < -1e-6 or x.max() > 1 + 1e-6):
            raise ValueError("Preprocessed spectra must lie in [0,1]; declare another transform explicitly")
        self.spectra = x
        self.source_device = canonical_device(self.source_device)
        if self.sample_ids is None:
            self.sample_ids = np.asarray([f"{self.source_device}:{i}" for i in range(len(x))])
        self.sample_ids = np.asarray(self.sample_ids, dtype=str)
        if (
            self.sample_ids.shape != (len(x),)
            or len(np.unique(self.sample_ids)) != len(x)
            or np.any(np.char.strip(self.sample_ids) == "")
        ):
            raise ValueError("sample_ids must have one unique string per spectrum")
        if self.labels is not None:
            y = np.asarray(self.labels)
            if y.shape != (len(x),) or not np.issubdtype(y.dtype, np.number) or np.iscomplexobj(y):
                raise ValueError("labels must be numeric [N]")
            if not np.isfinite(y).all() or np.any(y != np.floor(y)) or np.any((y < 0) | (y > 19)):
                raise ValueError("Labels must be integers 0 through 19")
            self.labels = y.astype(np.int64)
        if self.axis_cm1 is not None:
            if np.iscomplexobj(self.axis_cm1):
                raise ValueError("Raman coordinates must be real values")
            a = np.asarray(self.axis_cm1, dtype=np.float64)
            if a.shape != (LENGTH,) or not np.isfinite(a).all() or not np.all(np.diff(a) > 0):
                raise ValueError("axis_cm1 must contain 1400 finite, increasing wavenumbers")
            self.axis_cm1 = a
        if self.groups is not None:
            self.groups = np.asarray(self.groups, dtype=str)
            if self.groups.shape != (len(x),) or np.any(np.char.strip(self.groups) == ""):
                raise ValueError("groups must have one acquisition-group ID per spectrum")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a JSON object")
        json.dumps(self.metadata, allow_nan=False)
        return self

    def subset(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        return SpectrumBatch(
            self.spectra[rows],
            self.source_device,
            self.sample_ids[rows],
            None if self.labels is None else self.labels[rows],
            self.axis_cm1,
            None if self.groups is None else self.groups[rows],
            dict(self.metadata),
        )

    def identity(self):
        return {
            "spectra_sha256": array_hash(self.spectra),
            "sample_ids_sha256": array_hash(self.sample_ids),
            "labels_sha256": None if self.labels is None else array_hash(self.labels),
            "axis_cm1_sha256": None if self.axis_cm1 is None else array_hash(self.axis_cm1),
            "groups_sha256": None if self.groups is None else array_hash(self.groups),
            "source_device": self.source_device,
            "rows": len(self.spectra),
            "metadata": self.metadata,
        }


def save_batch(path, batch, normalized=True):
    batch.validate(normalized=normalized)
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Canonical batches use the .npz extension")
    if path.exists():
        raise ValueError(f"Refusing to overwrite an existing batch: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "spectra": batch.spectra,
        "sample_ids": batch.sample_ids,
        "source_device": np.asarray(batch.source_device),
        "metadata_json": np.asarray(json.dumps(batch.metadata, allow_nan=False)),
    }
    for key in ("labels", "axis_cm1", "groups"):
        value = getattr(batch, key)
        if value is not None:
            arrays[key] = value
    np.savez_compressed(path, **arrays)
    return path


def load_batch(path, normalized=True):
    with np.load(path, allow_pickle=False) as values:
        if not {"spectra", "source_device", "sample_ids"}.issubset(values.files):
            raise ValueError("Batch is missing canonical fields")
        batch = SpectrumBatch(
            values["spectra"],
            str(values["source_device"].item()),
            values["sample_ids"],
            **{k: values[k] if k in values else None for k in ("labels", "axis_cm1", "groups")},
            metadata=json.loads(str(values["metadata_json"].item())) if "metadata_json" in values else {},
        )
    return batch.validate(normalized=normalized)


def load_device(path, device):
    """Read canonical NPZ or the original root-key SERS HDF5; never alter inputs."""
    path = Path(path)
    device = canonical_device(device)
    if path.suffix == ".npz":
        batch = load_batch(path)
        if batch.source_device != device:
            raise ValueError(f"Batch device {batch.source_device} differs from requested {device}")
        return batch
    import h5py

    with h5py.File(path, "r") as handle:
        keys = {k.lower(): k for k in handle.keys()}
        try:
            x = np.asarray(handle[keys[f"{device}_GT".lower()]])
            y = np.asarray(handle[keys[f"{device}_GT_Labels".lower()]]).reshape(-1)
        except KeyError as error:
            raise ValueError(f"Missing {device}_GT and/or {device}_GT_Labels in {path.name}") from error
    # Historical files use either [N,L] or [N,L,1], not arbitrary axis inference.
    if x.ndim == 3 and x.shape[1:] == (LENGTH, 1):
        x = x[:, :, 0]
    digest = file_hash(path)
    return SpectrumBatch(
        x,
        device,
        np.asarray([f"{digest[:12]}:{device}:{i}" for i in range(len(x))]),
        y,
        metadata={
            "source_file_sha256": digest,
            "preprocessing": "legacy_preprocessed_01",
            "axis_provenance": "not stored; spectral bin index only",
        },
    ).validate()
