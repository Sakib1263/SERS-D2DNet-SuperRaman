"""Run installed-wheel import and cascade from outside the source checkout."""

from pathlib import Path
import sys
import sersd2d_superraman
from sersd2d_superraman.provenance import write_json


def main():
    assert "site-packages" in Path(sersd2d_superraman.__file__).parts
    assert "tensorflow" not in sys.modules and "torch" not in sys.modules
    assert "prismsers" not in sys.modules and "sersd2d_ablation" not in sys.modules
    from sersd2d_superraman.pipeline import run_pipeline

    root = Path(__file__).resolve().parents[1]
    report = run_pipeline(
        root / "tmp/synthetic_demo/bundle/bundle.json",
        root / "tmp/synthetic_demo/prepared/Enwave/test.npz",
        root / "tmp/installed_wheel_prediction",
    )
    assert report["rows"] == 20
    write_json(
        root / "tmp/installed_wheel_verification.json",
        {
            "installed_from_wheel": True,
            "imported_from_site_packages": True,
            "framework_free_initial_import": True,
            "neighbor_project_imports": False,
            "cascade_rows": 20,
            "scope": "synthetic installed-artifact verification",
        },
    )
    print("Installed wheel imports independently and completes the 20-row synthetic cascade")


if __name__ == "__main__":
    main()
