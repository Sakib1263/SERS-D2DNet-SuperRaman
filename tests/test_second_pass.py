import json
import subprocess
import numpy as np
import pytest
from sersd2d_superraman.cli import prepare, synthetic, _parts
from sersd2d_superraman.workflow import Workflow, validate_prepared
from sersd2d_superraman.provenance import read_json, write_json, file_hash
from sersd2d_superraman.data import load_batch, save_batch
from test_contracts import batch


def test_identity_covers_axis_and_acquisition_groups():
    b = batch()
    before = b.identity()
    b.axis_cm1 = np.arange(1400, dtype=float)
    b.groups = b.sample_ids.copy()
    assert b.identity()["axis_cm1_sha256"] != before["axis_cm1_sha256"]
    assert b.identity()["groups_sha256"] != before["groups_sha256"]


def test_empty_identifiers_rejected():
    b = batch()
    b.sample_ids[0] = ""
    with pytest.raises(ValueError):
        b.validate()
    b = batch()
    b.groups = np.full(len(b.spectra), "")
    with pytest.raises(ValueError):
        b.validate()


def test_prepared_device_mismatch_rejected(tmp_path):
    for role in ("train", "validation", "test"):
        save_batch(tmp_path / "Renishaw" / f"{role}.npz", batch("Enwave"))
    with pytest.raises(ValueError, match="device"):
        _parts(tmp_path, "Renishaw")


def test_sealed_preparation_detects_changed_spectral_content(tmp_path):
    synthetic(tmp_path / "source", per_class=5)
    prepare(tmp_path / "source", tmp_path / "prepared")
    assert validate_prepared(tmp_path / "prepared")["Enwave"]["test"] == 20
    path = tmp_path / "prepared/Enwave/test.npz"
    b = load_batch(path)
    b.spectra[0, 10] = 0.72
    path.unlink()
    save_batch(path, b)
    with pytest.raises(ValueError, match="spectral content"):
        validate_prepared(tmp_path / "prepared")


def demo_config():
    from pathlib import Path

    return read_json(Path(__file__).parents[1] / "configs/workflow_demo.json")


def test_workflow_refuses_changed_configuration(tmp_path):
    config = demo_config()
    config["run_directory"] = "run"
    Workflow(tmp_path, config)
    config["translation_device"] = "cuda"
    with pytest.raises(ValueError, match="Configuration changed"):
        Workflow(tmp_path, config)


def test_workflow_never_substitutes_demo_inputs_for_research(tmp_path):
    config = demo_config()
    config.update(mode="research", run_directory="run")
    with pytest.raises(ValueError, match="exactly one"):
        Workflow(tmp_path, config)


def test_research_rejects_synthetic_prepared_data(tmp_path):
    synthetic(tmp_path / "synthetic", per_class=5)
    prepare(tmp_path / "synthetic", tmp_path / "prepared")
    config = demo_config()
    config.update(mode="research", run_directory="research", prepared_data="prepared")
    w = Workflow(tmp_path, config)
    with pytest.raises(ValueError, match="refuses synthetic"):
        w.prepare()


def test_controller_reuse_checks_inputs_and_outputs(tmp_path, monkeypatch):
    config = demo_config()
    config["run_directory"] = "run"
    w = Workflow(tmp_path, config)
    artifact, source = tmp_path / "artifact.txt", tmp_path / "input.txt"
    source.write_text("original")
    calls = []

    def worker(*args, **kwargs):
        calls.append(args)
        artifact.write_text("produced")
        return subprocess.CompletedProcess(args[0], 0, "", "")

    monkeypatch.setattr(subprocess, "run", worker)
    w._step("probe", ["inspect"], [artifact], [source])
    w._step("probe", ["inspect"], [artifact], [source])
    assert len(calls) == 1
    source.write_text("changed")
    with pytest.raises(ValueError, match="Inputs or parameters"):
        w._step("probe", ["inspect"], [artifact], [source])
    source.write_text("original")
    artifact.write_text("changed")
    with pytest.raises(ValueError, match="artifact changed"):
        w._step("probe", ["inspect"], [artifact], [source])


def test_pipeline_rejects_invalid_batch_before_creating_output(tmp_path):
    from sersd2d_superraman.pipeline import run_pipeline

    with pytest.raises(ValueError, match="batch_size"):
        run_pipeline("missing", "missing", tmp_path / "run", batch_size=0)
    assert not (tmp_path / "run").exists()


def test_bundle_checks_semantic_role_even_when_manifest_hash_matches(tmp_path):
    pytest.importorskip("torch")
    from sersd2d_superraman.classification.model import SuperRaman, save_checkpoint
    from sersd2d_superraman.pipeline import create_bundle, verify_bundle

    save_checkpoint(tmp_path / "checkpoint", SuperRaman())
    manifest = create_bundle(tmp_path / "bundle", tmp_path / "checkpoint", {}, True, True)
    metadata_path = tmp_path / "bundle/models/classifier/metadata.json"
    metadata = read_json(metadata_path)
    metadata["source_device"] = "Enwave"
    write_json(metadata_path, metadata)
    bundle = read_json(manifest)
    bundle["classifier"]["metadata_sha256"] = file_hash(metadata_path)
    write_json(manifest, bundle)
    with pytest.raises(ValueError, match="source device"):
        verify_bundle(manifest)


def test_classifier_rejects_incomplete_validation_and_nan_decay(tmp_path):
    pytest.importorskip("torch")
    from sersd2d_superraman.classification.training import train_classifier, ClassifierTrainingConfig

    train, val = batch(seed=8), batch(seed=99).subset([0])
    val.sample_ids = np.asarray(["independent-validation-row"])
    with pytest.raises(ValueError, match="20 classes"):
        train_classifier(train, val, tmp_path / "run")
    with pytest.raises(ValueError, match="optimizer"):
        ClassifierTrainingConfig(weight_decay=float("nan")).validate()


def test_smoothgrad_rejects_nan_class_and_noise():
    pytest.importorskip("torch")
    from sersd2d_superraman.classification.model import SuperRaman
    from sersd2d_superraman.explainability import explain

    model = SuperRaman()
    with pytest.raises(ValueError, match="target_classes"):
        explain(model, batch().spectra[:1], target_classes=np.array([np.nan]))
    with pytest.raises(ValueError, match="settings"):
        explain(model, batch().spectra[:1], noise_std=np.nan)


def test_cli_tensorflow_device_is_explicit():
    from sersd2d_superraman.cli import parser

    args = parser().parse_args(
        [
            "translate",
            "--input",
            "in.npz",
            "--output",
            "out.npz",
            "--checkpoint",
            "checkpoint",
            "--device",
            "cpu",
        ]
    )
    assert args.device == "cpu"


def test_prediction_evaluation_rejects_foreign_input_metadata(tmp_path):
    from sersd2d_superraman.cli import main
    from sersd2d_superraman.data.registry import registry_hash

    b = batch()
    save_batch(tmp_path / "input.npz", b)
    metadata = {
        "input": {**b.identity(), "spectra_sha256": "foreign"},
        "class_registry_sha256": registry_hash(),
    }
    np.savez(
        tmp_path / "predictions.npz",
        sample_ids=b.sample_ids,
        logits=np.zeros((len(b.spectra), 20)),
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    with pytest.raises(SystemExit, match="different input"):
        main(
            [
                "evaluate-classification",
                "--input",
                str(tmp_path / "input.npz"),
                "--predictions",
                str(tmp_path / "predictions.npz"),
                "--output",
                str(tmp_path / "metrics.json"),
            ]
        )


def test_controller_refuses_concurrent_notebook_stage(tmp_path):
    config = demo_config()
    config["run_directory"] = "run"
    workflow = Workflow(tmp_path, config)
    (workflow.output / "workflow.lock").write_text("another process")
    with pytest.raises(ValueError, match="Another stage"):
        workflow._step("probe", ["inspect"], [])


def test_controller_rejects_modified_snapshot(tmp_path):
    config = demo_config()
    config["run_directory"] = "run"
    workflow = Workflow(tmp_path, config)
    snapshot = read_json(workflow.output / "classifier_config.json")
    snapshot["training"]["epochs"] = 123
    write_json(workflow.output / "classifier_config.json", snapshot)
    with pytest.raises(ValueError, match="snapshot changed"):
        Workflow(tmp_path, config)


def test_class_mean_metrics_reject_different_physical_grids():
    from sersd2d_superraman.evaluation import translation_metrics

    source, reference = batch("Enwave"), batch()
    source.axis_cm1 = np.arange(1400, dtype=float)
    reference.axis_cm1 = source.axis_cm1 + 1
    with pytest.raises(ValueError, match="axes differ"):
        translation_metrics(source, source.spectra, reference)


def test_complex_labels_and_axis_are_rejected():
    b = batch()
    b.labels = b.labels.astype(complex)
    with pytest.raises(ValueError, match="labels"):
        b.validate()
    b = batch()
    b.axis_cm1 = np.arange(1400).astype(complex)
    with pytest.raises(ValueError, match="real"):
        b.validate()


def test_already_translated_inputs_cannot_be_translated_twice(tmp_path):
    from sersd2d_superraman.cli import main
    from sersd2d_superraman.pipeline import run_pipeline

    translated = batch("Enwave")
    translated.metadata["translation"] = {"target_device": "Renishaw"}
    save_batch(tmp_path / "translated.npz", translated)
    with pytest.raises(ValueError, match="already translated"):
        run_pipeline("missing", tmp_path / "translated.npz", tmp_path / "cascade")
    with pytest.raises(SystemExit, match="already translated"):
        main(
            [
                "translate",
                "--input",
                str(tmp_path / "translated.npz"),
                "--checkpoint",
                "missing",
                "--output",
                str(tmp_path / "twice.npz"),
            ]
        )
