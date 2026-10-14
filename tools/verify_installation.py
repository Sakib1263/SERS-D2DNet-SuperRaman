"""Verify a wheel installation from outside the checkout using a selected portable bundle."""

import argparse
from pathlib import Path
import sys
import sersd2d_superraman
from sersd2d_superraman.provenance import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if "site-packages" not in Path(sersd2d_superraman.__file__).parts:
        raise ValueError("Expected an installed wheel, not an editable/source import")
    framework_free = "torch" not in sys.modules and "tensorflow" not in sys.modules
    if not framework_free:
        raise ValueError("Initial package import unexpectedly loaded an ML backend")
    from sersd2d_superraman.pipeline import run_pipeline

    report = run_pipeline(args.bundle, args.input, args.output)
    result = {
        "version": sersd2d_superraman.__version__,
        "imported_from_site_packages": True,
        "framework_free_initial_import": framework_free,
        "neighbor_project_imports": any(k in sys.modules for k in ("prismsers", "sersd2d_ablation")),
        "cascade_rows": report["rows"],
        "contains_synthetic_models": report["contains_synthetic_models"],
        "scope": "installed-artifact software verification; not scientific efficacy",
    }
    write_json(Path(args.output) / "installation_check.json", result)
    print(result)


if __name__ == "__main__":
    main()
