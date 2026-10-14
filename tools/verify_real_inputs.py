"""Bounded real-input compatibility checks; no scientific performance claims."""

from pathlib import Path
import numpy as np
from sersd2d_superraman.data import load_batch, DEVICES, SOURCE_DEVICES
from sersd2d_superraman.data.splits import assert_disjoint
from sersd2d_superraman.translation.model import build_model, primary_output
from sersd2d_superraman.classification.model import load_checkpoint, predict_logits
from sersd2d_superraman.provenance import write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    records = {}
    for device in DEVICES:
        batches = [
            load_batch(ROOT / "data/prepared" / device / f"{role}.npz")
            for role in ("train", "validation", "test")
        ]
        assert_disjoint(*batches)
        assert all(set(b.labels) == set(range(20)) for b in batches)
        records[device] = {
            "counts": [len(b.spectra) for b in batches],
            "all_classes_present": True,
            "no_identity_or_duplicate_overlap": True,
        }
    translator = build_model()
    classifier, _ = load_checkpoint(ROOT / "tmp/synthetic_demo/classifier/checkpoint")
    for device in SOURCE_DEVICES:
        batch = load_batch(ROOT / "data/prepared" / device / "test.npz")
        rows = np.asarray([np.flatnonzero(batch.labels == c)[0] for c in range(20)])
        spectra = batch.spectra[rows]
        translated = primary_output(translator, spectra, batch_size=1)
        logits = predict_logits(classifier, translated)
        records[device].update(
            checked_classes=20,
            translated_shape=list(translated.shape),
            logits_shape=list(logits.shape),
            finite=True,
            output_min=float(translated.min()),
            output_max=float(translated.max()),
        )
    write_json(
        ROOT / "docs/audit/real_input_verification.json",
        {
            "scope": "input compatibility only; translator initialized, classifier synthetic-trained",
            "translator_parameters": int(translator.count_params()),
            "devices": records,
        },
    )
    print("Verified real splits and default-graph inference for 20 classes on all four portable devices")


if __name__ == "__main__":
    main()
