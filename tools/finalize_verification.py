"""Collect completed local evidence and verify that supplied files are unchanged."""

from pathlib import Path
import hashlib
import json
import platform
import xml.etree.ElementTree as ET
import numpy as np
from sersd2d_superraman.provenance import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    tree = ET.parse(ROOT / "tmp/verification_tests.xml")
    suites = list(tree.getroot().iter("testsuite"))
    totals = {
        k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")
    }
    if totals["failures"] or totals["errors"]:
        raise ValueError("Cannot finalize a failing contract suite")
    manifest = read_json(ROOT / "docs/audit/source_inventory_2026-10-14.json")
    changed = [
        entry["path"]
        for entry in manifest["files"]
        if hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest() != entry["sha256"]
    ]
    if changed:
        raise ValueError(f"Supplied sources changed: {changed}")
    notebooks = read_json(ROOT / "tmp/notebook_validation/validation.json")
    if len(notebooks) != 5 or not all(n["executed"] for n in notebooks):
        raise ValueError("All five notebooks must execute")
    cascade = json.loads((ROOT / "tmp/notebook_validation/04_translate_and_classify.ipynb").read_text())
    text = "".join("".join(o.get("text", [])) for c in cascade["cells"] for o in c.get("outputs", []))
    if "'rows': 20" not in text:
        raise ValueError("Integration notebook did not execute its actual cascade")
    original = read_json(ROOT / "docs/audit/original_reconciliation.json")
    real = read_json(ROOT / "docs/audit/real_input_verification.json")
    conversion = read_json(ROOT / "docs/audit/conversion_verification.json")
    cuda = read_json(ROOT / "docs/audit/cuda_verification.json")
    demo = read_json(ROOT / "tmp/synthetic_demo/prediction/report.json")
    report = {
        "date": "2026-10-14",
        "software_version": "0.1.0",
        "scope": "software verification; not published-score reproduction",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "tensorflow": original["tensorflow"],
        "torch": cuda["torch"],
        "contract_tests": totals,
        "notebooks": notebooks,
        "supplied_files_verified_unchanged": len(manifest["files"]),
        "default_translator_parameters": original["parameters"],
        "default_classifier_parameters": 448572,
        "original_classifier_fixture": read_json(ROOT / "tests/fixtures/legacy_classifier_reference.json"),
        "real_input_checks": real,
        "legacy_conversion": conversion,
        "pytorch_cuda": cuda,
        "tensorflow_gpu_verified": False,
        "synthetic_cascade": {
            "rows": demo["rows"],
            "source_device": demo["source_device"],
            "sample_identity_preserved": True,
            "width": 2,
            "epochs_per_stage": 1,
        },
        "research_dataset_sha256": original["dataset_sha256"],
        "remaining": [
            "Original paper-trained checkpoints were not found",
            "Published results are not reproduced",
            "Full research training/evaluation is not run",
            "Original raw preprocessing/grid needs recovery",
            "GitHub publication and remote CI execution have not occurred",
        ],
    }
    write_json(ROOT / "docs/VERIFICATION.json", report)
    print({"tests": totals, "notebooks": len(notebooks), "original_files_unchanged": len(manifest["files"])})


if __name__ == "__main__":
    main()
