"""Dataset discovery and local-input guidance; no credentials or downloads required."""

from pathlib import Path
from .registry import DEVICES

DATASET_TITLE = "Multi-Instrument SERS Dataset"
DATASET_URL = "https://zenodo.org/records/23372983"
DATASET_DOI = "10.5281/zenodo.23372983"
DATASET_VERSION = "1.0.0"


def dataset_info():
    """Return offline guidance for the versioned dataset's restricted-access record."""
    return {
        "title": DATASET_TITLE,
        "record_url": DATASET_URL,
        "doi": DATASET_DOI,
        "version": DATASET_VERSION,
        "creator": "Sakib Mahmud",
        "access": "restricted",
        "request_policy": "Available following curator approval of a reasonable request.",
        "license_listed_by_record": "CC-BY-4.0",
        "steps": [
            "Open the Zenodo record and sign in to request access.",
            "Briefly explain the intended use; include affiliation if applicable.",
            "After approval, download the files and extract any archive locally.",
            "Set dataset to the local preprocessed HDF5 file or five-device NPZ directory, "
            "or set prepared_data to an existing prepared study; use exactly one input setting.",
        ],
        "local_input_formats": ["preprocessed five-device HDF5", "five-device canonical NPZ directory"],
        "documentation": "docs/DATA.md",
        "files_included_in_git": False,
        "trained_weights_included_in_software": False,
    }


def _local_path(value):
    if "://" in str(value):
        raise ValueError(
            f"Use a local downloaded path, not a record or download URL. "
            f"Request dataset access at {DATASET_URL}, then configure dataset or prepared_data."
        )
    return Path(value).expanduser()


def require_dataset(value):
    """Check local inputs before creating preparation outputs or launching workers."""
    path = _local_path(value)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Request access at {DATASET_URL}; after approval, "
            "download/extract the data and set dataset to its local preprocessed HDF5 file "
            "or five-device NPZ directory. See docs/DATA.md or run sers-pipeline dataset-info."
        )
    if path.is_dir():
        missing = [f"{device}.npz" for device in DEVICES if not (path / f"{device}.npz").is_file()]
        if missing:
            raise FileNotFoundError(
                f"Dataset directory is missing canonical device batches: {', '.join(missing)}. "
                "Point dataset to the downloaded preprocessed HDF5 file, or provide all five "
                "device NPZ files as described in docs/DATA.md."
            )
    elif not path.is_file():
        raise ValueError(f"Dataset must be a local file or directory: {path}")
    elif path.suffix.lower() in (".zip", ".tar", ".gz", ".tgz", ".7z", ".rar"):
        raise ValueError(
            "Extract the approved dataset archive locally first, then point dataset to the "
            "preprocessed HDF5 file or five-device NPZ directory. See docs/DATA.md."
        )
    elif path.suffix.lower() == ".npz":
        raise ValueError(
            "Preparation requires all five devices. Set dataset to the directory containing "
            "Renishaw.npz, Enwave.npz, Tec5.npz, Fdefender.npz, and RID.npz, or to the HDF5 file."
        )
    return path


def require_prepared(value):
    """Explain the distinction between a downloaded dataset and prepared split roles."""
    path = _local_path(value)
    if not path.is_dir():
        raise FileNotFoundError(
            f"Prepared study not found: {path}. Request dataset access at {DATASET_URL}. "
            "After downloading/extracting it, set dataset to the local preprocessed HDF5 file "
            "instead of prepared_data to create the split roles, or run sers-pipeline prepare. "
            "See docs/DATA.md."
        )
    required = [Path("study.json")]
    required += [
        Path(d) / name for d in DEVICES for name in ("split.json", "train.npz", "validation.npz", "test.npz")
    ]
    missing = [p.as_posix() for p in required if not (path / p).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Prepared study is incomplete: {path}; first missing artifact: {missing[0]}. "
            "prepared_data must name the output of sers-pipeline prepare, not a downloaded "
            "archive or an unprepared dataset directory. See docs/DATA.md."
        )
    return path
