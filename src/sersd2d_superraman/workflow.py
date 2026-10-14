"""Notebook control surface with persistent configuration, explicit stages, and verified artifact reuse.

Training still occurs independently in the two backends. The controller uses the same
CLI workers as terminal users and never substitutes demo data for missing research data.
"""

from pathlib import Path
import os
import sys
from .data import DEVICES, SOURCE_DEVICES, load_batch
from .data.splits import assert_disjoint
from .provenance import (
    array_hash,
    file_hash,
    json_hash,
    read_json,
    write_json,
    exclusive_lock,
    implementation_identity,
)
from .workers import run_worker
from .artifacts import checkpoint_metadata
import numpy as np
from . import __version__


def validate_prepared(directory):
    """Check all roles before training, including cross-device acquisition-group leakage."""
    directory = Path(directory)
    study = read_json(directory / "study.json")
    if study.get("protocol") != "maintained_independent_within_device":
        raise ValueError("Unsupported preparation protocol")
    roles = ("train", "validation", "test")
    batches, counts = {}, {}
    for device in DEVICES:
        split = read_json(directory / device / "split.json")
        body = {k: v for k, v in split.items() if k != "manifest_sha256"}
        if json_hash(body) != split.get("manifest_sha256"):
            raise ValueError(f"Changed split manifest: {device}")
        if study["devices"][device]["manifest_sha256"] != split["manifest_sha256"]:
            raise ValueError(f"Study/split manifest mismatch: {device}")
        counts[device] = {}
        for role in roles:
            batch = load_batch(directory / device / f"{role}.npz")
            if batch.source_device != device or batch.labels is None or set(batch.labels) != set(range(20)):
                raise ValueError(f"All 20 labeled classes are required in {device}/{role}")
            if batch.sample_ids.tolist() != split["parts"][role]["sample_ids"]:
                raise ValueError(f"Split row identities changed: {device}/{role}")
            class_counts = {str(c): int((batch.labels == c).sum()) for c in range(20)}
            if class_counts != split["parts"][role]["class_counts"]:
                raise ValueError(f"Split labels changed: {device}/{role}")
            batches[device, role] = batch
            for field in ("axis_cm1",):
                value = getattr(batch, field)
                digest = None if value is None else array_hash(value)
                if digest != split["input"].get(field + "_sha256"):
                    raise ValueError(f"Prepared Raman axis changed: {device}/{role}")
            counts[device][role] = len(batch.spectra)
        assert_disjoint(*(batches[device, role] for role in roles))
        # Reconstruct original row order to check spectral content against the sealed manifest.
        rows = [np.asarray(split["parts"][role]["rows"], dtype=np.int64) for role in roles]
        all_rows = np.concatenate(rows)
        if not np.array_equal(np.sort(all_rows), np.arange(len(all_rows))):
            raise ValueError(f"Split indices are incomplete or duplicated: {device}")
        restored = np.empty((len(all_rows), 1400), dtype=np.float32)
        restored_labels = np.empty(len(all_rows), dtype=np.int64)
        grouped = batches[device, "train"].groups is not None
        restored_groups = (
            np.empty(len(all_rows), dtype=batches[device, "train"].groups.dtype) if grouped else None
        )
        for role, indices in zip(roles, rows):
            restored[indices] = batches[device, role].spectra
            restored_labels[indices] = batches[device, role].labels
            if (batches[device, role].groups is not None) != grouped:
                raise ValueError(f"Prepared group availability changed: {device}/{role}")
            if grouped:
                restored_groups[indices] = batches[device, role].groups
            if batches[device, role].metadata != split["input"]["metadata"]:
                raise ValueError(f"Prepared metadata changed: {device}/{role}")
        if array_hash(restored) != split["input"]["spectra_sha256"]:
            raise ValueError(f"Prepared spectral content changed: {device}")
        if array_hash(restored_labels) != split["input"]["labels_sha256"]:
            raise ValueError(f"Prepared label assignment changed: {device}")
        group_digest = None if restored_groups is None else array_hash(restored_groups)
        if group_digest != split["input"].get("groups_sha256"):
            raise ValueError(f"Prepared acquisition groups changed: {device}")
    # A union per role detects the same cross-device group intersections without
    # rebuilding large ID/group sets for every device pair.
    role_groups = {
        role: set().union(
            *(set(batches[d, role].groups) if batches[d, role].groups is not None else set() for d in DEVICES)
        )
        for role in roles
    }
    for i, role in enumerate(roles):
        for other in roles[i + 1 :]:
            if role_groups[role] & role_groups[other]:
                raise ValueError("Acquisition group overlap between split roles")
    return counts


class Workflow:
    """Open a run shared by every notebook; reuse only completed, content-verified stages.

    A failed stage remains visible and its directory is preserved for inspection.
    Choose a new run directory to retry it. This is artifact reuse, not optimizer resume.
    """

    def __init__(self, project, configuration):
        self.project = Path(project).resolve()
        config = read_json(configuration) if isinstance(configuration, (str, Path)) else dict(configuration)
        if not isinstance(config, dict):
            raise ValueError("Workflow configuration must be a JSON object")
        if not isinstance(config.get("run_directory"), str) or not config["run_directory"].strip():
            raise ValueError("run_directory must be a nonempty path")
        required = {
            "schema_version",
            "mode",
            "run_directory",
            "sources",
            "classifier_config",
            "translation_config",
            "classifier_device",
            "translation_device",
        }
        if not required.issubset(config) or set(config) - (
            required | {"dataset", "prepared_data", "per_class"}
        ):
            raise ValueError("Incomplete or unknown workflow configuration fields")
        if config["schema_version"] != 1 or config["mode"] not in ("demo", "research"):
            raise ValueError("Workflow mode must be demo or research")
        if (
            not isinstance(config["sources"], list)
            or not config["sources"]
            or len(set(config["sources"])) != len(config["sources"])
            or set(config["sources"]) - set(SOURCE_DEVICES)
        ):
            raise ValueError("Select unique portable source devices")
        if config["translation_device"] not in ("cpu", "cuda"):
            raise ValueError("translation_device must be cpu or cuda")
        for key in ("classifier_config", "translation_config"):
            if not isinstance(config[key], dict) or set(config[key]) != {"model", "training"}:
                raise ValueError(f"{key} requires model and training objects")
        if config["mode"] == "research":
            if bool(config.get("dataset")) == bool(config.get("prepared_data")):
                raise ValueError("Research mode requires exactly one dataset or prepared_data path")
        if config["mode"] == "demo" and (config.get("dataset") or config.get("prepared_data")):
            raise ValueError("Demo mode uses generated synthetic inputs only")
        import re

        if not isinstance(config["classifier_device"], str) or not re.fullmatch(
            r"cpu|cuda(?::[0-9]+)?", config["classifier_device"]
        ):
            raise ValueError("classifier_device must be cpu, cuda, or cuda:<index>")
        if any(
            not isinstance(config[k][section], dict)
            for k in ("classifier_config", "translation_config")
            for section in ("model", "training")
        ):
            raise ValueError("Model and training configurations must be objects")
        self.config = config
        self.output = self.resolve(config["run_directory"])
        self.state_path = self.output / "workflow.json"
        self.output.mkdir(parents=True, exist_ok=True)
        with exclusive_lock(self.output / "workflow.lock"):
            if self.state_path.exists():
                self.state = read_json(self.state_path)
                self._check_configuration()
            else:
                if any(p.name != "workflow.lock" for p in self.output.iterdir()):
                    raise ValueError("Output directory is nonempty; select a new run_directory")
                self.state = {
                    "schema_version": 1,
                    "configuration": config,
                    "configuration_sha256": json_hash(config),
                    "steps": {},
                    "paper_reproduction_certified": False,
                }
                # Publish workflow.json last: incomplete initialization cannot be reopened.
                for role in ("classifier", "translation"):
                    write_json(self.output / f"{role}_config.json", config[f"{role}_config"])
                write_json(self.state_path, self.state)

    def _check_configuration(self):
        digest = json_hash(self.config)
        if (
            self.state.get("configuration_sha256") != digest
            or json_hash(self.state.get("configuration")) != digest
        ):
            raise ValueError("Configuration changed for this run. Select a new run_directory.")
        for role in ("classifier", "translation"):
            if read_json(self.output / f"{role}_config.json") != self.config[f"{role}_config"]:
                raise ValueError("Configuration snapshot changed; select a new run_directory")

    def resolve(self, value):
        path = Path(value).expanduser()
        return (self.project / path).resolve() if not path.is_absolute() else path.resolve()

    @property
    def prepared(self):
        return (
            self.resolve(self.config["prepared_data"])
            if self.config.get("prepared_data")
            else self.output / "prepared"
        )

    @property
    def bundle(self):
        return self.output / "bundle/bundle.json"

    def _step(self, key, arguments, outputs, inputs=()):
        with exclusive_lock(self.output / "workflow.lock"):
            self.state = read_json(self.state_path)
            self._check_configuration()
            return self._execute_step(key, arguments, outputs, inputs)

    def _execute_step(self, key, arguments, outputs, inputs=()):
        outputs = [Path(p).resolve() for p in outputs]
        implementation = implementation_identity()
        signature = json_hash(
            {
                "software_version": __version__,
                "implementation_sha256": implementation,
                "arguments": list(map(str, arguments)),
                "inputs": {str(Path(p).resolve()): file_hash(p) for p in inputs},
            }
        )
        existing = self.state["steps"].get(key)
        if existing:
            if existing["status"] != "completed":
                raise ValueError(
                    f"Stage {key} previously failed or was interrupted. Inspect it and start a new run."
                )
            if existing["signature"] != signature or set(
                existing.get("declared_outputs", existing["outputs"])
            ) != set(map(str, outputs)):
                raise ValueError(f"Inputs or parameters changed for completed stage {key}; start a new run")
            for name, digest in existing["outputs"].items():
                p = Path(name)
                if file_hash(p) != digest:
                    raise ValueError(f"Completed stage artifact changed: {p.name}")
            print(f"{key}: verified existing artifacts")
            return
        self.state["steps"][key] = {"status": "running", "signature": signature}
        write_json(self.state_path, self.state)
        env = os.environ.copy()
        env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONUTF8"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = str(self.project / "src") + os.pathsep + env.get("PYTHONPATH", "")
        if arguments[0] == "train-translation" and self.config["translation_device"] == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = "-1"
        try:
            log_path = self.output / "logs" / f"{key.replace(':', '_')}.log"
            run_worker(
                [sys.executable, "-m", "sersd2d_superraman", *map(str, arguments)],
                log_path,
                cwd=self.project,
                env=env,
            )
            if implementation_identity() != implementation:
                raise ValueError("Package source changed while the stage was running; start a new run")
            committed_outputs = list(outputs)
            for p in outputs:
                if p.name == "metadata.json" and p.parent.name == "checkpoint":
                    committed_outputs.append(p.parent / checkpoint_metadata(p.parent)["weights"])
            record = {
                "status": "completed",
                "signature": signature,
                "declared_outputs": list(map(str, outputs)),
                "implementation_sha256": implementation,
                "log": str(log_path),
                "outputs": {str(p): file_hash(p) for p in committed_outputs},
            }
            self.state["steps"][key] = record
            print(f"{key}: completed")
        except BaseException as error:
            self.state["steps"][key] = {"status": "failed", "signature": signature, "error": str(error)}
            raise
        finally:
            write_json(self.state_path, self.state)

    def prepare(self):
        """Prepare synthetic or authorized real data, then check every device and role."""
        if not self.config.get("prepared_data"):
            dataset = (
                self.resolve(self.config["dataset"])
                if self.config.get("dataset")
                else self.output / "synthetic"
            )
            if self.config["mode"] == "demo":
                self._step(
                    "synthetic",
                    ["make-synthetic", "--output", dataset, "--per-class", self.config.get("per_class", 5)],
                    [dataset / f"{device}.npz" for device in DEVICES],
                )
            inputs = [dataset / f"{d}.npz" for d in DEVICES] if dataset.is_dir() else [dataset]
            outputs = [self.prepared / "study.json"]
            outputs += [
                self.prepared / device / name
                for device in DEVICES
                for name in ("split.json", "train.npz", "validation.npz", "test.npz")
            ]
            self._step(
                "prepare", ["prepare", "--dataset", dataset, "--output", self.prepared], outputs, inputs
            )
        counts = validate_prepared(self.prepared)
        if self.config["mode"] == "research":
            for device in DEVICES:
                if load_batch(self.prepared / device / "train.npz").metadata.get("synthetic"):
                    raise ValueError("Research mode refuses synthetic prepared data")
        return counts

    def train_translations(self):
        self.prepare()
        for device in self.config["sources"]:
            output = self.output / "translation" / device
            config = self.output / "translation_config.json"
            inputs = [
                self.prepared / d / f"{role}.npz"
                for d in (device, "Renishaw")
                for role in ("train", "validation", "test")
            ]
            self._step(
                f"translation:{device}",
                [
                    "train-translation",
                    "--data",
                    self.prepared,
                    "--source",
                    device,
                    "--config",
                    config,
                    "--device",
                    self.config["translation_device"],
                    "--output",
                    output,
                ],
                [
                    output / "run.json",
                    output / "history.json",
                    output / "checkpoint/metadata.json",
                ],
                [config, *inputs],
            )
        return {
            device: read_json(self.output / "translation" / device / "run.json")
            for device in self.config["sources"]
        }

    def train_classifier(self):
        self.prepare()
        output = self.output / "classification"
        config = self.output / "classifier_config.json"
        inputs = [self.prepared / "Renishaw" / f"{r}.npz" for r in ("train", "validation", "test")]
        self._step(
            "classifier",
            [
                "train-classifier",
                "--data",
                self.prepared,
                "--config",
                config,
                "--device",
                self.config["classifier_device"],
                "--output",
                output,
            ],
            [
                output / "run.json",
                output / "history.json",
                output / "checkpoint/metadata.json",
            ],
            [config, *inputs],
        )
        return read_json(output / "run.json")

    def build_bundle(self):
        from .pipeline import verify_bundle

        classifier = self.output / "classification/checkpoint"
        args = ["bundle", "--classifier", classifier, "--output", self.output / "bundle"]
        inputs = [classifier / "metadata.json", classifier / checkpoint_metadata(classifier)["weights"]]
        for device in self.config["sources"]:
            checkpoint = self.output / "translation" / device / "checkpoint"
            args += ["--translator", f"{device}={checkpoint}"]
            inputs += [checkpoint / "metadata.json", checkpoint / checkpoint_metadata(checkpoint)["weights"]]
        if set(self.config["sources"]) != set(SOURCE_DEVICES):
            args += ["--allow-partial"]
        self._step("bundle", args, [self.bundle], inputs)
        verify_bundle(self.bundle)
        return self.bundle

    def evaluate(self):
        """Held-out portable-device cascade plus class-mean Renishaw comparisons."""
        from .pipeline import verify_bundle

        self.prepare()
        verify_bundle(self.bundle)
        reports = {}
        for device in self.config["sources"]:
            output = self.output / "evaluation" / device
            source = self.prepared / device / "test.npz"
            reference = self.prepared / "Renishaw/test.npz"
            self._step(
                f"predict:{device}",
                [
                    "predict",
                    "--bundle",
                    self.bundle,
                    "--input",
                    source,
                    "--output",
                    output,
                    "--device",
                    self.config["classifier_device"],
                    "--translation-device",
                    self.config["translation_device"],
                ],
                [
                    output / name
                    for name in (
                        "report.json",
                        "translated.npz",
                        "original_predictions.npz",
                        "translated_predictions.npz",
                    )
                ],
                [source, self.bundle],
            )
            self._step(
                f"metrics:{device}",
                [
                    "evaluate-translation",
                    "--input",
                    source,
                    "--translated",
                    output / "translated.npz",
                    "--reference",
                    reference,
                    "--output",
                    output / "translation_metrics.json",
                ],
                [output / "translation_metrics.json"],
                [source, reference, output / "translated.npz"],
            )
            reports[device] = {
                "classification": read_json(output / "report.json")["classification"],
                "translation": read_json(output / "translation_metrics.json")["macro_class_mean"],
            }
        write_json(
            self.output / "evaluation_summary.json",
            {"mode": self.config["mode"], "devices": reports, "paper_reproduction_certified": False},
        )
        return reports

    def status(self):
        self.state = read_json(self.state_path)
        return {key: value["status"] for key, value in self.state["steps"].items()}
