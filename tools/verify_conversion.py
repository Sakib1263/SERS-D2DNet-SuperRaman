"""Exercise full-object migration using the trusted original class and synthetic weights."""

from pathlib import Path
import importlib
import os
import sys
import torch
from sersd2d_superraman.data import load_batch
from sersd2d_superraman.classification.conversion import convert_legacy
from sersd2d_superraman.provenance import write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = os.environ["SERS_ORIGINAL_CLASSIFICATION"]
    sys.path.insert(0, directory)
    module = importlib.import_module("models_utils")
    torch.manual_seed(37)
    model = module.SuperRAMAN_V2(input_ch=1, class_num=20).eval()
    target = ROOT / "tmp/legacy_conversion_test"
    target.mkdir(parents=True, exist_ok=True)
    old = target / "synthetic_original_object.pt"
    torch.save({"model": model, "scope": "synthetic initialization"}, old)
    batch = load_batch(ROOT / "tmp/synthetic_demo/prepared/Enwave/test.npz")
    report = convert_legacy(
        old,
        target / "converted",
        legacy_source_directory=directory,
        parity_spectra=batch.spectra,
        trust_pickle=True,
    )
    report["verification_scope"] = "synthetic original-class object, not paper weights"
    write_json(ROOT / "docs/audit/conversion_verification.json", report)
    print(report)


if __name__ == "__main__":
    main()
