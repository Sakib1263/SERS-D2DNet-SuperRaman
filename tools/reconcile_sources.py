"""Read original experiments; write bounded evidence in this checkout only."""

from pathlib import Path
import hashlib
import importlib.util
import json
import os

ROOT = Path(__file__).resolve().parents[1]


def main():
    import h5py
    import numpy as np
    import tensorflow as tf

    original = Path(os.environ["SERS_ORIGINAL_TRANSLATION"])
    classification = Path(os.environ["SERS_ORIGINAL_CLASSIFICATION"])
    source = original / "models/MLMRSNet.py"
    spec = importlib.util.spec_from_file_location("original_translation_graph", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tf.keras.utils.set_random_seed(37)
    model = module.MLMRSNet(1400, 3, 1, 128, 3, ds=1, cardinality=5, pooling_type="mix").IMUNet()
    values = model([np.zeros((1, 1400, 1), np.float32)], training=False)
    evidence = {
        "original_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "tensorflow": tf.__version__,
        "keras": tf.keras.__version__,
        "parameters": model.count_params(),
        "output_names": model.output_names,
        "output_shapes": [list(v.shape) for v in values],
        "forward_finite": all(bool(np.isfinite(v.numpy()).all()) for v in values),
        "layers": len(model.layers),
        "trained_weights_found": [],
        "notebook_settings": {},
    }
    for directory in (original, classification):
        for p in directory.rglob("*"):
            if p.is_file() and (p.suffix == ".keras" or p.name.endswith(".weights.h5")):
                evidence["trained_weights_found"].append(str(p))
    for name in ("RAMAN_Task_V2.ipynb", "RAMAN_Task_V3_External.ipynb"):
        obj = json.loads((original / name).read_text(encoding="utf-8-sig"))
        selected = []
        for i, cell in enumerate(obj["cells"]):
            text = "".join(cell.get("source", []))
            if any(
                key in text
                for key in (
                    "model_name =",
                    "def prepareTrainDict",
                    "num_samples_per_class",
                    "samples_per_class",
                )
            ):
                selected.append({"cell": i, "source": text})
            elif ".fit(" in text and ".IMUNet()" in text:
                selected.append(
                    {
                        "cell": i,
                        "training_lines": [
                            line
                            for line in text.splitlines()
                            if any(
                                key in line
                                for key in (
                                    "epochs=",
                                    "batch_size=",
                                    "patience=",
                                    "monitor=",
                                    "loss_weights",
                                    "learning_rate=",
                                )
                            )
                        ],
                    }
                )
        evidence["notebook_settings"][name] = selected
    dataset = original / "SERS_D2DNet_Dataset.h5"
    inventory = {}
    with h5py.File(dataset, "r") as handle:
        evidence["dataset_keys"] = list(handle.keys())
        for device in ("Renishaw", "Enwave", "Tec5", "Fdefender", "RID"):
            x = np.asarray(handle[device + "_GT"])
            y = np.asarray(handle[device + "_GT_Labels"]).reshape(-1)
            counts = dict(zip(*np.unique(y, return_counts=True)))
            inventory[device] = {
                "shape": list(x.shape),
                "dtype": str(x.dtype),
                "min": float(x.min()),
                "max": float(x.max()),
                "finite": bool(np.isfinite(x).all()),
                "label_counts": {str(int(k)): int(v) for k, v in counts.items()},
            }
    evidence["dataset_inventory"] = inventory
    evidence["dataset_sha256"] = hashlib.file_digest(dataset.open("rb"), "sha256").hexdigest()
    out = ROOT / "docs/audit/original_reconciliation.json"
    out.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                k: evidence[k]
                for k in (
                    "parameters",
                    "output_names",
                    "output_shapes",
                    "forward_finite",
                    "layers",
                    "trained_weights_found",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
