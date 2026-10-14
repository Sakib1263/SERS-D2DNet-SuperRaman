"""Failure, cancellation, editable-source, and efficiency cases first reviewed in v0.3.

These checks do not train models, execute notebooks, or replay the earlier numerical
regressions. Framework writers are represented by bytes to isolate publication faults.
"""

from pathlib import Path
import sys
import numpy as np
import pytest
from sersd2d_superraman import provenance as p
from sersd2d_superraman.data import SpectrumBatch, save_batch
from sersd2d_superraman.workflow import Workflow


def config():
    value = p.read_json(Path(__file__).parents[1] / "configs/workflow_demo.json")
    value["run_directory"] = "run"
    return value


def test_failed_npz_publication_leaves_no_final_or_partial_file(tmp_path, monkeypatch):
    def fail(path, **arrays):
        Path(path).write_bytes(b"partial zip")
        raise OSError("simulated disk full")

    monkeypatch.setattr(np, "savez_compressed", fail)
    with pytest.raises(OSError, match="disk full"):
        p.write_npz(tmp_path / "batch.npz", spectra=np.ones((1, 1400)))
    assert list(tmp_path.iterdir()) == []


def test_atomic_no_clobber_when_another_writer_wins(tmp_path):
    target = tmp_path / "winner.npz"
    with pytest.raises(ValueError, match="overwrite"):
        with p.atomic_output(target) as temporary:
            temporary.write_bytes(b"loser")
            target.write_bytes(b"winner")
    assert target.read_bytes() == b"winner"
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("suffix", ["model.pt", "model.weights.h5"])
@pytest.mark.parametrize("fault", ["weights", "metadata"])
def test_failed_checkpoint_update_preserves_previous_committed_pair(tmp_path, monkeypatch, suffix, fault):
    original = p.publish_checkpoint(tmp_path, suffix, lambda path: path.write_bytes(b"valid"), {"epoch": 1})
    before = (tmp_path / "metadata.json").read_bytes()

    def fail(*args, **kwargs):
        raise OSError("simulated publication failure")

    def writer(path):
        path.write_bytes(b"new weights")

    if fault == "weights":
        writer = fail
    else:
        monkeypatch.setattr(p, "write_json", fail)
    with pytest.raises(OSError, match="publication failure"):
        p.publish_checkpoint(tmp_path, suffix, writer, {"epoch": 2})
    assert (tmp_path / "metadata.json").read_bytes() == before
    assert (tmp_path / original["weights"]).read_bytes() == b"valid"
    assert len(list(tmp_path.iterdir())) == 2


def test_checkpoint_success_retires_previous_managed_weights(tmp_path):
    old = p.publish_checkpoint(tmp_path, "model.pt", lambda path: path.write_bytes(b"old"), {"epoch": 1})
    new = p.publish_checkpoint(tmp_path, "model.pt", lambda path: path.write_bytes(b"new"), {"epoch": 2})
    assert not (tmp_path / old["weights"]).exists()
    assert p.read_json(tmp_path / "metadata.json") == new
    assert p.file_hash(tmp_path / new["weights"]) == new["weights_sha256"]


def test_interrupted_controller_marks_failure_and_releases_lock(tmp_path, monkeypatch):
    workflow = Workflow(tmp_path, config())

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt("cancelled by user")

    monkeypatch.setattr("sersd2d_superraman.workflow.run_worker", interrupt)
    with pytest.raises(KeyboardInterrupt):
        workflow._step("interrupted", ["inspect"], [])
    assert workflow.status() == {"interrupted": "failed"}
    assert not (workflow.output / "workflow.lock").exists()


def test_live_controller_rechecks_configuration_snapshot(tmp_path):
    workflow = Workflow(tmp_path, config())
    snapshot = workflow.output / "classifier_config.json"
    p.write_json(snapshot, {"model": {}, "training": {}})
    with pytest.raises(ValueError, match="snapshot changed"):
        workflow._step("must-not-start", ["inspect"], [])
    assert workflow.status() == {}


def test_editable_source_change_invalidates_cached_stage_without_version_bump(tmp_path, monkeypatch):
    workflow = Workflow(tmp_path, config())
    identity = ["first implementation"]
    monkeypatch.setattr("sersd2d_superraman.workflow.implementation_identity", lambda: identity[0])
    monkeypatch.setattr("sersd2d_superraman.workflow.run_worker", lambda *args, **kwargs: None)
    workflow._step("cached", ["inspect"], [])
    identity[0] = "changed implementation"
    with pytest.raises(ValueError, match="Inputs or parameters changed"):
        workflow._step("cached", ["inspect"], [])


def test_source_changes_during_a_worker_fail_the_stage(tmp_path, monkeypatch):
    workflow = Workflow(tmp_path, config())
    identities = iter(["before", "after"])
    monkeypatch.setattr("sersd2d_superraman.workflow.implementation_identity", lambda: next(identities))
    monkeypatch.setattr("sersd2d_superraman.workflow.run_worker", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="source changed while"):
        workflow._step("changing", ["inspect"], [])
    assert workflow.status() == {"changing": "failed"}


def test_incomplete_workflow_initialization_has_no_committed_state(tmp_path, monkeypatch):
    import sersd2d_superraman.workflow as module

    original = module.write_json

    def fail_translation(path, value):
        if Path(path).name == "translation_config.json":
            raise OSError("simulated initialization failure")
        return original(path, value)

    monkeypatch.setattr(module, "write_json", fail_translation)
    with pytest.raises(OSError):
        Workflow(tmp_path, config())
    assert not (tmp_path / "run/workflow.json").exists()
    assert not (tmp_path / "run/workflow.lock").exists()
    monkeypatch.setattr(module, "write_json", original)
    with pytest.raises(ValueError, match="nonempty"):
        Workflow(tmp_path, config())


def test_file_backed_worker_keeps_full_log_and_bounded_error_tail(tmp_path):
    from sersd2d_superraman.workers import run_worker

    log = tmp_path / "worker.log"
    command = [sys.executable, "-c", "import sys; print('x' * 100000); print('failure marker'); sys.exit(7)"]
    with pytest.raises(RuntimeError, match="failure marker") as captured:
        run_worker(command, log)
    assert log.stat().st_size > 100000
    assert len(str(captured.value)) < 6400
    assert "status 7" in str(captured.value)


def test_worker_interrupt_terminates_its_posix_process_group(tmp_path, monkeypatch):
    import sersd2d_superraman.workers as module

    if module.os.name != "posix":
        pytest.skip("POSIX group cancellation contract")
    signals, starts = [], []

    class Process:
        pid = 123456

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def wait(self, timeout=None):
            if timeout is None:
                raise KeyboardInterrupt()
            return -15

    def launch(*args, **kwargs):
        starts.append(kwargs)
        return Process()

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    monkeypatch.setattr(module.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    with pytest.raises(KeyboardInterrupt):
        module.run_worker(["mock-worker"], tmp_path / "cancel.log")
    assert starts[0]["start_new_session"] is True
    assert signals == [(Process.pid, module.signal.SIGTERM)]


def test_translation_evaluation_rejects_same_ids_from_a_different_input_file(tmp_path):
    from sersd2d_superraman.cli import main

    values = np.random.default_rng(123).uniform(0, 1, (2, 1400)).astype(np.float32)
    source = SpectrumBatch(values, "Enwave", labels=np.array([0, 1])).validate()
    source_path = save_batch(tmp_path / "source.npz", source)
    translated = SpectrumBatch(
        values.copy(),
        "Enwave",
        source.sample_ids.copy(),
        source.labels.copy(),
        metadata={
            "translation": {
                "source_device": "Enwave",
                "target_device": "Renishaw",
                "input_file_sha256": "foreign input",
            }
        },
    )
    save_batch(tmp_path / "translated.npz", translated)
    save_batch(tmp_path / "reference.npz", SpectrumBatch(values, "Renishaw", labels=source.labels))
    with pytest.raises(SystemExit, match="Translation provenance"):
        main(
            [
                "evaluate-translation",
                "--input",
                str(source_path),
                "--translated",
                str(tmp_path / "translated.npz"),
                "--reference",
                str(tmp_path / "reference.npz"),
                "--output",
                str(tmp_path / "metrics.json"),
            ]
        )
    assert not (tmp_path / "metrics.json").exists()


def test_disjointness_hashes_each_role_once(tmp_path, monkeypatch):
    from sersd2d_superraman.data import splits

    batches = [
        SpectrumBatch(np.full((2, 1400), i), "Enwave", sample_ids=np.array([f"{i}:a", f"{i}:b"]))
        for i in range(3)
    ]
    calls = []

    def fingerprints(values):
        calls.append(id(values))
        return np.array([str(id(values))])

    monkeypatch.setattr(splits, "fingerprints", fingerprints)
    splits.assert_disjoint(*batches)
    assert len(calls) == len(set(calls)) == 3


@pytest.mark.parametrize(
    "payload", ['{"mode":"demo","mode":"research"}', '{"learning_rate":NaN}', '{"value":Infinity}']
)
def test_json_refuses_ambiguous_or_nonfinite_configuration(tmp_path, payload):
    path = tmp_path / "config.json"
    path.write_text(payload)
    with pytest.raises(ValueError):
        p.read_json(path)


def test_bundle_rejects_false_complete_device_declaration(tmp_path):
    from sersd2d_superraman.pipeline import verify_bundle
    from sersd2d_superraman.data.registry import registry_hash

    path = tmp_path / "bundle.json"
    p.write_json(
        path,
        {
            "schema_version": 1,
            "class_registry_sha256": registry_hash(),
            "classifier": {},
            "translators": {},
            "partial": False,
        },
    )
    with pytest.raises(ValueError, match="partial-device"):
        verify_bundle(path)


def test_doctor_reports_missing_notebook_packages_without_backend_imports(monkeypatch):
    import sersd2d_superraman.diagnostics as module

    before = set(sys.modules)

    def installed(name):
        if name == "torch":
            raise module.PackageNotFoundError(name)
        return "test-version"

    monkeypatch.setattr(module, "version", installed)
    report = module.environment_report()
    assert report["packages"]["torch"] is None
    assert report["missing_notebook_packages"] == ["torch"]
    assert not ({"torch", "tensorflow"} & (set(sys.modules) - before))


def test_missing_backend_message_identifies_extra_and_interpreter(monkeypatch):
    import sersd2d_superraman.cli as module

    def missing(args):
        raise ModuleNotFoundError("No module named tensorflow", name="tensorflow")

    monkeypatch.setattr(module, "_run", missing)
    with pytest.raises(SystemExit) as captured:
        module.main(["doctor"])
    assert 'python -m pip install -e ".[translation]"' in str(captured.value)
    assert sys.executable in str(captured.value)


def test_interrupt_after_metadata_commit_keeps_new_checkpoint_readable(tmp_path, monkeypatch):
    p.publish_checkpoint(tmp_path, "model.pt", lambda path: path.write_bytes(b"old"), {"epoch": 1})
    commit = p.write_json

    def interrupt_after_commit(path, obj):
        commit(path, obj)
        raise KeyboardInterrupt("interrupt at publication boundary")

    monkeypatch.setattr(p, "write_json", interrupt_after_commit)
    with pytest.raises(KeyboardInterrupt):
        p.publish_checkpoint(tmp_path, "model.pt", lambda path: path.write_bytes(b"new"), {"epoch": 2})
    metadata = p.read_json(tmp_path / "metadata.json")
    assert metadata["epoch"] == 2
    assert p.file_hash(tmp_path / metadata["weights"]) == metadata["weights_sha256"]


def test_controller_records_the_metadata_selected_checkpoint_weight(tmp_path, monkeypatch):
    from sersd2d_superraman.data.registry import registry_hash
    from sersd2d_superraman.preprocessing import CLASSIFIER_TRANSFORM

    workflow = Workflow(tmp_path, config())
    checkpoint = workflow.output / "classification/checkpoint"
    metadata = {
        "schema_version": 1,
        "kind": "classification",
        "config": {},
        "architecture_id": "test-publication-only",
        "source_device": "Renishaw",
        "input_layout": "N,1,L",
        "input_length": 1400,
        "class_registry_sha256": registry_hash(),
        "parameters": 1,
        "training": {"status": "untrained_initialization"},
        "input_transform": CLASSIFIER_TRANSFORM,
    }

    def worker(*args, **kwargs):
        p.publish_checkpoint(
            checkpoint, "model.pt", lambda path: path.write_bytes(b"stand-in bytes"), metadata
        )

    monkeypatch.setattr("sersd2d_superraman.workflow.run_worker", worker)
    workflow._step("classifier-publication", ["mock-checkpoint-writer"], [checkpoint / "metadata.json"])
    record = p.read_json(workflow.state_path)["steps"]["classifier-publication"]
    selected = p.read_json(checkpoint / "metadata.json")["weights"]
    assert str(checkpoint / selected) in record["outputs"]
    assert str(checkpoint / "model.pt") not in record["outputs"]


def test_nested_worker_cancellation_reaches_the_real_grandchild(tmp_path, monkeypatch):
    import os
    import time
    import sersd2d_superraman.workers as module

    if os.name != "posix":
        pytest.skip("POSIX nested worker process-group contract")
    monkeypatch.delenv("_SERS_WORKER_GROUP", raising=False)
    pid_file = tmp_path / "grandchild.pid"
    descendant = f"import os,time; from pathlib import Path; Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(30)"
    nested = (
        "import sys; from sersd2d_superraman.workers import run_worker; "
        f"run_worker([sys.executable, '-c', {descendant!r}], {str(tmp_path / 'nested.log')!r})"
    )
    original_wait = module.subprocess.Popen.wait
    interrupted = []

    def wait_then_interrupt(process, timeout=None):
        if timeout is None and not interrupted:
            for _ in range(150):
                if pid_file.exists():
                    break
                time.sleep(0.02)
            assert pid_file.exists(), "Grandchild did not start"
            interrupted.append(process.pid)
            raise KeyboardInterrupt("cancel nested workflow")
        return original_wait(process, timeout=timeout)

    monkeypatch.setattr(module.subprocess.Popen, "wait", wait_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        module.run_worker([sys.executable, "-c", nested], tmp_path / "outer.log")
    pid = int(pid_file.read_text())
    # A terminated child can briefly remain as a zombie awaiting init's reap.
    for _ in range(100):
        stat = Path(f"/proc/{pid}/stat")
        if not stat.exists() or stat.read_text().split()[2] == "Z":
            break
        time.sleep(0.01)
    else:
        os.kill(pid, module.signal.SIGKILL)
        pytest.fail("Cancelled nested worker left a live grandchild")
