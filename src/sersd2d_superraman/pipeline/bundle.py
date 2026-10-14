"""Portable model bundles and separate-process inference orchestration."""

from pathlib import Path
import os
import shutil
import subprocess
import sys
import time
import numpy as np
from ..data import load_batch, SOURCE_DEVICES, registry_hash
from ..provenance import file_hash, fresh_directory, read_json, write_json
from ..artifacts import checkpoint_metadata


def create_bundle(output, classifier, translators, allow_partial=False, allow_untrained=False):
    if not allow_partial and set(translators) != set(SOURCE_DEVICES):
        raise ValueError(
            "A complete bundle requires all four translators; use allow_partial for a selected device"
        )
    if set(translators) - set(SOURCE_DEVICES):
        raise ValueError("Bundle contains an unknown portable device")
    sources = {
        "classifier": (Path(classifier), "classification", "Renishaw"),
        **{d: (Path(p), "translation", d) for d, p in translators.items()},
    }
    # Validate every artifact before creating the output directory.
    metadata = {}
    for key, (directory, kind, device) in sources.items():
        obj = checkpoint_metadata(directory, kind, device)
        weights = directory / obj["weights"]
        if Path(obj["weights"]).name != obj["weights"]:
            raise ValueError("Checkpoint weight name must be a simple filename")
        if obj.get("schema_version") != 1 or obj.get("kind") != kind or obj.get("source_device") != device:
            raise ValueError(f"Checkpoint role/device mismatch: {key}")
        if obj.get("class_registry_sha256") != registry_hash() or file_hash(weights) != obj["weights_sha256"]:
            raise ValueError(f"Checkpoint hash/class mapping mismatch: {key}")
        if not allow_untrained and obj.get("training", {}).get("status") == "untrained_initialization":
            raise ValueError("Untrained models require explicit allow_untrained for a demonstration")
        metadata[key] = obj
    output = fresh_directory(output)
    entries = {}
    for key, (directory, _, _) in sources.items():
        dest = output / "models" / key
        dest.mkdir(parents=True)
        for name in ("metadata.json", metadata[key]["weights"]):
            shutil.copy2(directory / name, dest / name)
        entries[key] = {
            "directory": dest.relative_to(output).as_posix(),
            "metadata_sha256": file_hash(dest / "metadata.json"),
            "weights_sha256": metadata[key]["weights_sha256"],
        }
    bundle = {
        "schema_version": 1,
        "class_registry_sha256": registry_hash(),
        "classifier": entries["classifier"],
        "translators": {d: entries[d] for d in translators},
        "partial": set(translators) != set(SOURCE_DEVICES),
        "contains_synthetic_models": any(m["training"].get("synthetic", False) for m in metadata.values()),
        "contains_untrained_models": any(
            m["training"]["status"] == "untrained_initialization" for m in metadata.values()
        ),
        "paper_reproduction_certified": False,
    }
    write_json(output / "bundle.json", bundle)
    return output / "bundle.json"


def verify_bundle(path):
    path = Path(path).resolve()
    bundle = read_json(path)
    if bundle.get("schema_version") != 1 or bundle.get("class_registry_sha256") != registry_hash():
        raise ValueError("Unsupported bundle or class registry")
    if set(bundle["translators"]) - set(SOURCE_DEVICES):
        raise ValueError("Unknown translator device in bundle")
    directories = {}
    synthetic_models = False
    untrained_models = False
    for key, entry in {"classifier": bundle["classifier"], **bundle["translators"]}.items():
        directory = (path.parent / entry["directory"]).resolve()
        if not directory.is_relative_to(path.parent):
            raise ValueError("Bundle artifacts must be within the bundle directory")
        if file_hash(directory / "metadata.json") != entry["metadata_sha256"]:
            raise ValueError(f"Bundle metadata changed: {key}")
        meta = checkpoint_metadata(
            directory,
            "classification" if key == "classifier" else "translation",
            "Renishaw" if key == "classifier" else key,
        )
        synthetic_models |= bool(meta["training"].get("synthetic", False))
        untrained_models |= meta["training"]["status"] == "untrained_initialization"
        weights = directory / meta["weights"]
        if not weights.resolve().is_relative_to(directory) or file_hash(weights) != entry["weights_sha256"]:
            raise ValueError(f"Bundle weight hash mismatch: {key}")
        directories[key] = directory
    if "contains_untrained_models" in bundle and bundle["contains_untrained_models"] != untrained_models:
        raise ValueError("Bundle training status mismatch")
    bundle["contains_synthetic_models"] = synthetic_models
    bundle["contains_untrained_models"] = untrained_models
    return bundle, directories


def run_pipeline(bundle_path, input_path, output, batch_size=16, device="cpu", translation_device="cpu"):
    """Translate once, classify both paths, and compare identical input identities."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if translation_device not in ("cpu", "cuda"):
        raise ValueError("translation_device must be cpu or cuda")
    batch = load_batch(input_path)
    if batch.metadata.get("translation") is not None:
        raise ValueError("Input is already translated; use classify directly")
    bundle, directories = verify_bundle(bundle_path)
    if batch.source_device != "Renishaw" and batch.source_device not in directories:
        raise ValueError(f"No translator for {batch.source_device} in this bundle")
    output = fresh_directory(output)
    started = time.perf_counter()

    def launch(command, tensorflow=False):
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        if tensorflow and translation_device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = "-1"
        result = subprocess.run(
            [sys.executable, "-m", "sersd2d_superraman", *command], env=env, capture_output=True, text=True
        )
        if result.returncode:
            raise RuntimeError(f"Stage {command[0]} failed:\n{result.stderr[-6000:]}")

    translated = output / "translated.npz"
    if batch.source_device == "Renishaw":
        shutil.copy2(input_path, translated)
    else:
        launch(
            [
                "translate",
                "--checkpoint",
                str(directories[batch.source_device]),
                "--input",
                str(input_path),
                "--output",
                str(translated),
                "--batch-size",
                str(batch_size),
                "--device",
                translation_device,
            ],
            tensorflow=True,
        )
    for role, path in (("original", input_path), ("translated", translated)):
        launch(
            [
                "classify",
                "--checkpoint",
                str(directories["classifier"]),
                "--input",
                str(path),
                "--output",
                str(output / f"{role}_predictions.npz"),
                "--batch-size",
                str(batch_size),
                "--device",
                device,
            ]
        )
    with (
        np.load(output / "original_predictions.npz", allow_pickle=False) as before,
        np.load(output / "translated_predictions.npz", allow_pickle=False) as after,
    ):
        if not np.array_equal(before["sample_ids"], after["sample_ids"]) or not np.array_equal(
            before["sample_ids"], batch.sample_ids
        ):
            raise ValueError("Prediction row identities changed across stages")
        report = {
            "schema_version": 1,
            "input": batch.identity(),
            "bundle_sha256": file_hash(bundle_path),
            "source_device": batch.source_device,
            "rows": len(batch.spectra),
            "wall_seconds_including_subprocess_startup_and_io": time.perf_counter() - started,
            "contains_untrained_models": bundle["contains_untrained_models"],
            "contains_synthetic_models": bundle["contains_synthetic_models"],
            "device": device,
            "translation_device": translation_device,
            "translation_bypassed": batch.source_device == "Renishaw",
            "synthetic": bool(batch.metadata.get("synthetic", False)),
            "paper_reproduction_certified": False,
            "translated_file_sha256": file_hash(translated),
            "original_predictions_sha256": file_hash(output / "original_predictions.npz"),
            "translated_predictions_sha256": file_hash(output / "translated_predictions.npz"),
        }
        if batch.labels is not None:
            from ..evaluation import classification_metrics

            report["classification"] = {
                "original": classification_metrics(batch.labels, before["logits"]),
                "translated": classification_metrics(batch.labels, after["logits"]),
            }
    write_json(output / "report.json", report)
    return report
