"""Inspect the selected interpreter without importing or initializing ML backends."""

from importlib.metadata import PackageNotFoundError, version
import platform
import sys
from . import __version__
from .data.access import dataset_info


def environment_report():
    packages = {}
    for name in ("numpy", "h5py", "tensorflow", "torch", "matplotlib", "jupyterlab", "ipykernel"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "software_version": __version__,
        "interpreter": sys.executable,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "python_supported": (3, 11) <= sys.version_info[:2] < (3, 14),
        "packages": packages,
        "missing_notebook_packages": [k for k, v in packages.items() if v is None],
        "historically_verified_frameworks": {"tensorflow": "2.21.0", "torch": "2.6.0"},
        "dataset": dataset_info(),
        "scope": "Installed metadata only; GPU availability and binary compatibility are not probed.",
    }
