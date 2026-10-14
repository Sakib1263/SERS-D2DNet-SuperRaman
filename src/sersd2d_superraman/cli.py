"""Standalone command line for data, both models, and the cascaded pipeline."""

import argparse
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
from .data import SpectrumBatch, load_batch, save_batch, load_device, DEVICES, SOURCE_DEVICES
from .data.registry import canonical_device
from .data.splits import independent_split, assert_disjoint
from .provenance import fresh_directory, file_hash, read_json, write_json
from .devices import tensorflow_device


def synthetic(output, per_class=8, seed=37):
    if per_class < 5:
        raise ValueError("Synthetic split demonstration needs at least five samples per class")
    output = fresh_directory(output)
    rng = np.random.default_rng(seed)
    axis = np.arange(1400)
    labels = np.repeat(np.arange(20), per_class)
    for number, device in enumerate(DEVICES):
        spectra = []
        for label in labels:
            center = 50 + label * 60
            x = np.exp(-(((axis - center) / (12 + label % 5)) ** 2))
            x += 0.3 * np.exp(-(((axis - (1200 - label * 20)) / 35) ** 2))
            x += rng.uniform(0, 0.035, 1400)
            x = np.maximum(x, 0) ** (1 + number * 0.03)
            x = (x - x.min()) / np.ptp(x)
            spectra.append(x)
        batch = SpectrumBatch(
            np.asarray(spectra),
            device,
            np.asarray([f"synthetic:{device}:{i}" for i in range(len(labels))]),
            labels,
            metadata={
                "synthetic": True,
                "seed": seed,
                "preprocessing": "synthetic_01_v1",
                "axis_provenance": "synthetic spectral bin index; no physical Raman axis",
            },
        )
        save_batch(output / f"{device}.npz", batch)
    return {"output": str(output), "samples_per_device": len(labels), "synthetic": True}


def prepare(dataset, output, seed=37, validation_fraction=0.1, test_fraction=0.2):
    output = fresh_directory(output)
    dataset = Path(dataset)
    manifests = {}
    for device in DEVICES:
        path = dataset / f"{device}.npz" if dataset.is_dir() else dataset
        batch = load_device(path, device)
        parts, manifest = independent_split(batch, seed, validation_fraction, test_fraction)
        for role, rows in parts.items():
            save_batch(output / device / f"{role}.npz", batch.subset(rows))
        write_json(output / device / "split.json", manifest)
        manifests[device] = {
            "manifest_sha256": manifest["manifest_sha256"],
            "counts": {k: len(v) for k, v in parts.items()},
        }
    write_json(
        output / "study.json",
        {
            "schema_version": 1,
            "protocol": "maintained_independent_within_device",
            "devices": manifests,
            "test_used_for_selection": False,
        },
    )
    return manifests


def _config(path):
    obj = {} if path is None else read_json(path)
    if obj.get("executable") is False:
        raise ValueError("Historical evidence configuration is not executable; choose a maintained config")
    if set(obj) - {"model", "training"}:
        raise ValueError("Configuration contains unknown fields; expected model and training objects")
    if any(not isinstance(v, dict) for v in obj.values()):
        raise ValueError("Configuration model/training values must be objects")
    return obj


def _parts(directory, device):
    batches = tuple(
        load_batch(Path(directory) / device / f"{name}.npz") for name in ("train", "validation", "test")
    )
    if any(b.source_device != device for b in batches):
        raise ValueError("Prepared split device does not match its directory")
    return batches


def _run(args):
    if args.command in (
        "translate",
        "classify",
        "evaluate-classification",
        "evaluate-translation",
        "explain",
        "benchmark",
    ):
        if Path(args.output).exists():
            raise ValueError("Output file already exists; choose a new artifact path")
    if args.command == "make-synthetic":
        return synthetic(args.output, args.per_class, args.seed)
    if args.command == "prepare":
        return prepare(args.dataset, args.output, args.seed, args.validation_fraction, args.test_fraction)
    if args.command == "inspect":
        batch = load_batch(args.input, normalized=not args.translated)
        return {
            **batch.identity(),
            "range": [float(batch.spectra.min()), float(batch.spectra.max())],
            "classes": None if batch.labels is None else sorted(set(batch.labels.tolist())),
        }
    if args.command == "train-classifier":
        from .classification.model import ClassifierConfig
        from .classification.training import ClassifierTrainingConfig, train_classifier

        obj = _config(args.config)
        parameters = obj.get("training", {}).copy()
        if args.epochs is not None:
            parameters["epochs"] = args.epochs
        tr, val, test = _parts(args.data, "Renishaw")
        assert_disjoint(tr, val, test)
        _, report = train_classifier(
            tr,
            val,
            args.output,
            ClassifierConfig(**obj.get("model", {})),
            ClassifierTrainingConfig(**parameters),
            args.device,
        )
        return report
    if args.command == "train-translation":
        from .translation.model import TranslationConfig
        from .translation.training import TranslationTrainingConfig, train_translation

        obj = _config(args.config)
        parameters, architecture = obj.get("training", {}).copy(), obj.get("model", {}).copy()
        if args.epochs is not None:
            parameters["epochs"] = args.epochs
        if args.width is not None:
            architecture["width"] = args.width
        source = canonical_device(args.source)
        st, sv, sx = _parts(args.data, source)
        rt, rv, rx = _parts(args.data, "Renishaw")
        assert_disjoint(st, sv, sx)
        assert_disjoint(rt, rv, rx)
        with tensorflow_device(args.device):
            _, report = train_translation(
                st,
                rt,
                sv,
                rv,
                args.output,
                TranslationConfig(**architecture),
                TranslationTrainingConfig(**parameters),
            )
        return report
    if args.command == "translate":
        from .translation.model import load_checkpoint, primary_output

        batch = load_batch(args.input)
        if batch.metadata.get("translation") is not None:
            raise ValueError("Input is already translated; classify it directly instead of translating twice")
        with tensorflow_device(args.device):
            model, _, meta = load_checkpoint(args.checkpoint)
            if meta["source_device"] != batch.source_device:
                raise ValueError("Translator checkpoint is for a different source device")
            result = primary_output(model, batch.spectra, args.batch_size)
        out = SpectrumBatch(
            result,
            batch.source_device,
            batch.sample_ids,
            batch.labels,
            batch.axis_cm1,
            batch.groups,
            {
                **batch.metadata,
                "translation": {
                    "checkpoint_sha256": meta["weights_sha256"],
                    "source_device": batch.source_device,
                    "target_device": "Renishaw",
                    "output_head": "out",
                    "output_range_policy": "linear_unclipped",
                    "input_file_sha256": file_hash(args.input),
                },
            },
        )
        save_batch(args.output, out, normalized=False)
        return {
            "output": args.output,
            "rows": len(result),
            "range": [float(result.min()), float(result.max())],
        }
    if args.command == "classify":
        from .classification.model import load_checkpoint, predict_logits

        batch = load_batch(args.input, normalized=False)
        model, metadata = load_checkpoint(args.checkpoint, args.device)
        logits = predict_logits(model, batch.spectra, args.batch_size, args.device)
        probabilities = np.exp(logits - logits.max(1, keepdims=True))
        probabilities /= probabilities.sum(1, keepdims=True)
        out = Path(args.output)
        if out.suffix != ".npz":
            raise ValueError("Predictions require a .npz output")
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            logits=logits,
            probabilities=probabilities,
            predictions=logits.argmax(1),
            sample_ids=batch.sample_ids,
            metadata_json=np.asarray(
                json.dumps(
                    {
                        "checkpoint_sha256": metadata["weights_sha256"],
                        "input": batch.identity(),
                        "class_registry_sha256": metadata["class_registry_sha256"],
                        "probability_calibration": "not established",
                    },
                    allow_nan=False,
                )
            ),
        )
        return {"output": str(out), "rows": len(logits)}
    if args.command == "evaluate-classification":
        from .evaluation import classification_metrics

        batch = load_batch(args.input, normalized=False)
        if batch.labels is None:
            raise ValueError("Ground-truth labels are required for evaluation")
        with np.load(args.predictions, allow_pickle=False) as values:
            if not np.array_equal(values["sample_ids"], batch.sample_ids):
                raise ValueError("Prediction IDs do not match the ground-truth batch")
            metadata = json.loads(str(values["metadata_json"].item()))
            from .data.registry import registry_hash

            if (
                metadata.get("class_registry_sha256") != registry_hash()
                or metadata.get("input") != batch.identity()
            ):
                raise ValueError("Predictions belong to a different input or class registry")
            report = classification_metrics(batch.labels, values["logits"])
        write_json(args.output, report)
        return {k: report[k] for k in ("accuracy", "weighted_f1", "samples")}
    if args.command == "evaluate-translation":
        from .evaluation import translation_metrics

        source = load_batch(args.input)
        translated = load_batch(args.translated, normalized=False)
        reference = load_batch(args.reference)
        if (
            not np.array_equal(source.sample_ids, translated.sample_ids)
            or source.source_device != translated.source_device
        ):
            raise ValueError("Translated rows do not match original input")
        report = translation_metrics(source, translated.spectra, reference)
        write_json(args.output, report)
        return report["macro_class_mean"]
    if args.command == "bundle":
        from .pipeline import create_bundle

        translators = {}
        for text in args.translator:
            device, path = text.split("=", 1)
            device = canonical_device(device)
            if device in translators:
                raise ValueError("Duplicate translator device")
            translators[device] = path
        path = create_bundle(
            args.output, args.classifier, translators, args.allow_partial, args.allow_untrained
        )
        return {"bundle": str(path)}
    if args.command == "predict":
        from .pipeline import run_pipeline

        return run_pipeline(
            args.bundle, args.input, args.output, args.batch_size, args.device, args.translation_device
        )
    if args.command == "convert-classifier":
        from .classification.model import ClassifierConfig
        from .classification.conversion import convert_legacy

        batch = load_batch(args.parity_input)
        obj = _config(args.config)
        return convert_legacy(
            args.input,
            args.output,
            ClassifierConfig(**obj.get("model", {})),
            args.legacy_source,
            batch.spectra,
            args.trust_legacy_pickle,
        )
    if args.command == "explain":
        from .classification.model import load_checkpoint
        from .explainability import explain

        batch = load_batch(args.input, normalized=False)
        model, meta = load_checkpoint(args.checkpoint, args.device)
        saliency, classes = explain(
            model,
            batch.spectra,
            target=args.target,
            smooth_samples=args.samples,
            noise_std=args.noise_std,
            seed=args.seed,
        )
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix != ".npz":
            raise ValueError("Saliency requires .npz output")
        arrays = {
            "saliency": saliency,
            "sample_ids": batch.sample_ids,
            "target_classes": classes,
            "metadata_json": np.asarray(
                json.dumps(
                    {
                        "target": args.target,
                        "gradient_input_range": [-1, 1],
                        "samples": args.samples,
                        "noise_std": args.noise_std,
                        "seed": args.seed,
                        "checkpoint_sha256": meta["weights_sha256"],
                        "axis": "bin index" if batch.axis_cm1 is None else "provided cm^-1",
                    }
                )
            ),
        }
        if batch.axis_cm1 is not None:
            arrays["axis_cm1"] = batch.axis_cm1
        np.savez_compressed(out, **arrays)
        return {"output": str(out), "rows": len(saliency)}
    if args.command in ("benchmark", "audit-model"):
        if args.kind == "translation":
            from .translation.model import load_checkpoint, build_model, TranslationConfig
            import tensorflow as tf

            if args.command == "benchmark":
                if args.device not in ("cpu", "cuda"):
                    raise ValueError("TensorFlow benchmark device must be cpu or cuda (GPU 0)")
                if args.device == "cuda" and not tf.config.list_physical_devices("GPU"):
                    raise ValueError("TensorFlow GPU requested but its runtime cannot access one")
            placement = "/GPU:0" if args.device == "cuda" else "/CPU:0"
            with tf.device(placement):
                if args.checkpoint:
                    model, _, meta = load_checkpoint(args.checkpoint)
                else:
                    model, meta = build_model(TranslationConfig(width=args.width)), {}
            parameters = int(model.count_params())
            outputs = {k: list(v.shape) for k, v in zip(model.output_names, model.outputs)}
        else:
            from .classification.model import SuperRaman, load_checkpoint

            model, meta = (
                load_checkpoint(args.checkpoint, args.device) if args.checkpoint else (SuperRaman(), {})
            )
            parameters, outputs = sum(p.numel() for p in model.parameters()), {"logits": [None, 20]}
        if args.command == "audit-model":
            return {"kind": args.kind, "parameters": parameters, "outputs": outputs}
        from .evaluation.benchmark import benchmark

        batch = load_batch(args.input, normalized=False)
        report = benchmark(
            model, batch.spectra[: args.batch_size], args.kind, args.device, args.warmup, args.repetitions
        )
        report.update(
            kind=args.kind,
            architecture_id=meta["architecture_id"],
            device=args.device,
            input_scope=batch.metadata,
            checkpoint_sha256=meta["weights_sha256"],
            serialized_weight_bytes=(Path(args.checkpoint) / meta["weights"]).stat().st_size,
        )
        write_json(args.output, report)
        return report
    if args.command == "demo":
        output = fresh_directory(args.output)

        def call(*words):
            subprocess.run([sys.executable, "-m", "sersd2d_superraman", *map(str, words)], check=True)

        call("make-synthetic", "--output", output / "synthetic", "--per-class", 5)
        call("prepare", "--dataset", output / "synthetic", "--output", output / "prepared")
        call(
            "train-classifier",
            "--data",
            output / "prepared",
            "--output",
            output / "classifier",
            "--epochs",
            1,
        )
        call(
            "train-translation",
            "--data",
            output / "prepared",
            "--source",
            "Enwave",
            "--output",
            output / "translator",
            "--epochs",
            1,
            "--width",
            2,
        )
        call(
            "bundle",
            "--classifier",
            output / "classifier/checkpoint",
            "--translator",
            f"Enwave={output / 'translator/checkpoint'}",
            "--allow-partial",
            "--output",
            output / "bundle",
        )
        call(
            "predict",
            "--bundle",
            output / "bundle/bundle.json",
            "--input",
            output / "prepared/Enwave/test.npz",
            "--output",
            output / "prediction",
        )
        return {
            "output": str(output),
            "scope": "synthetic tiny-training verification; not scientific validation",
        }
    raise ValueError("Unknown command")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in (
        "make-synthetic",
        "prepare",
        "inspect",
        "train-classifier",
        "train-translation",
        "translate",
        "classify",
        "evaluate-classification",
        "evaluate-translation",
        "bundle",
        "predict",
        "convert-classifier",
        "explain",
        "benchmark",
        "audit-model",
        "demo",
    ):
        s = sub.add_parser(name)
        if name in (
            "inspect",
            "translate",
            "classify",
            "evaluate-classification",
            "evaluate-translation",
            "predict",
            "convert-classifier",
            "explain",
            "benchmark",
        ):
            s.add_argument("--input", required=True)
        if name not in ("inspect", "audit-model"):
            s.add_argument("--output", required=True)
        if name in ("translate", "classify", "explain", "benchmark", "audit-model"):
            s.add_argument("--checkpoint", required=name != "audit-model")
        if name in ("train-classifier", "train-translation", "convert-classifier"):
            s.add_argument("--config")
        if name in ("train-classifier", "train-translation"):
            s.add_argument("--data", required=True)
            s.add_argument("--epochs", type=int)
        if name in ("make-synthetic", "prepare", "explain"):
            s.add_argument("--seed", type=int, default=37)
        if name in (
            "translate",
            "train-translation",
            "classify",
            "predict",
            "explain",
            "benchmark",
            "audit-model",
            "train-classifier",
        ):
            s.add_argument("--device", default="cpu")
        if name in ("translate", "classify", "predict", "benchmark"):
            s.add_argument("--batch-size", type=int, default=16)
        if name in ("train-translation", "audit-model"):
            s.add_argument("--width", type=int, default=128 if name == "audit-model" else None)
        if name == "make-synthetic":
            s.add_argument("--per-class", type=int, default=8)
        if name == "prepare":
            s.add_argument("--dataset", required=True)
            s.add_argument("--validation-fraction", type=float, default=0.1)
            s.add_argument("--test-fraction", type=float, default=0.2)
        if name == "inspect":
            s.add_argument("--translated", action="store_true")
        if name == "train-translation":
            s.add_argument("--source", choices=SOURCE_DEVICES, required=True)
        if name == "evaluate-classification":
            s.add_argument("--predictions", required=True)
        if name == "evaluate-translation":
            s.add_argument("--translated", required=True)
            s.add_argument("--reference", required=True)
        if name == "bundle":
            s.add_argument("--classifier", required=True)
            s.add_argument("--translator", action="append", default=[])
            s.add_argument("--allow-partial", action="store_true")
            s.add_argument("--allow-untrained", action="store_true")
        if name == "predict":
            s.add_argument("--bundle", required=True)
            s.add_argument("--translation-device", choices=("cpu", "cuda"), default="cpu")
        if name == "convert-classifier":
            s.add_argument("--legacy-source")
            s.add_argument("--parity-input", required=True)
            s.add_argument("--trust-legacy-pickle", action="store_true")
        if name == "explain":
            s.add_argument("--target", choices=("logit", "probability"), default="logit")
            s.add_argument("--samples", type=int, default=1)
            s.add_argument("--noise-std", type=float, default=0)
        if name in ("benchmark", "audit-model"):
            s.add_argument("--kind", choices=("classification", "translation"), required=True)
        if name == "benchmark":
            s.add_argument("--warmup", type=int, default=5)
            s.add_argument("--repetitions", type=int, default=20)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = _run(args)
    except (ValueError, TypeError, OSError, ModuleNotFoundError, RuntimeError) as error:
        raise SystemExit(f"{type(error).__name__}: {error}") from error
    # Reports retain details in their files; stdout remains bounded.
    payload = json.dumps(result, indent=2, allow_nan=False)
    if len(payload) > 5000:
        payload = json.dumps(
            {
                "status": "completed",
                "command": args.command,
                "output": getattr(args, "output", None),
                "details": "Complete results are saved in the output artifacts.",
            },
            indent=2,
        )
    print(payload)
