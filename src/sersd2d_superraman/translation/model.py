"""Focused recovered graph, named output inference, and portable weight contracts."""

from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
from ..provenance import file_hash, runtime, write_json
from ..data.registry import registry_hash

IMPLEMENTATION_SHA256 = "aac5cda48b1e9130e89d3e3ee05099b2941a4ba6eb8d8068ec76da5f9316b5ae"

SOURCE_SHA256 = "9026c64849b1d27f171d69121684c00a3515eed2e8c175cca4d9d9bb94a19878"


@dataclass(frozen=True)
class TranslationConfig:
    depth: int = 3
    width: int = 128

    def validate(self):
        if type(self.depth) is not int or self.depth not in (1, 2, 3):
            raise ValueError("Translation depth must be 1, 2, or 3 for length 1400")
        if type(self.width) is not int or self.width < 1:
            raise ValueError("Translation width must be a positive integer")
        return self

    @property
    def architecture_id(self):
        return f"imunet_supplied_9026c648_d{self.depth}_w{self.width}_ds1"


def build_model(config=None, seed=37):
    import tensorflow as tf
    from .architecture import MLMRSNet

    config = (config or TranslationConfig()).validate()
    tf.keras.mixed_precision.set_global_policy("float32")
    source = Path(__file__).with_name("architecture.py")
    if file_hash(source) != IMPLEMENTATION_SHA256:
        raise ValueError("Preserved translator source changed; reconcile its architecture identity first")
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    tf.keras.utils.set_random_seed(seed)
    model = MLMRSNet(
        1400,
        config.depth,
        1,
        config.width,
        3,
        ds=1,
        ae=0,
        cardinality=5,
        pooling_type="mix",
        is_transconv=True,
    ).IMUNet()
    names = tuple(f"level{i}" for i in range(1, config.depth + 1)) + ("out",)
    expected = [(1400 // 2**i, 1) for i in range(1, config.depth + 1)] + [(1400, 1)]
    if tuple(model.output_names) != names or [tuple(x.shape[1:]) for x in model.outputs] != expected:
        raise ValueError("Translator output contract changed")
    if config == TranslationConfig() and model.count_params() != 61_511_562:
        raise ValueError("Supplied translator parameter count changed")
    return model


def primary_output(model, spectra, batch_size=16):
    x = np.asarray(spectra, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 1400 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("Translation requires finite, nonempty [N,1400]")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    index = list(model.output_names).index("out")
    pieces = []
    for start in range(0, len(x), batch_size):
        values = model([x[start : start + batch_size, :, None]], training=False)
        output = values[index] if isinstance(values, (list, tuple)) else values
        pieces.append(np.asarray(output, dtype=np.float32)[:, :, 0])
    result = np.concatenate(pieces)
    if result.shape != x.shape or not np.isfinite(result).all():
        raise ValueError("Translator produced invalid output")
    return result


def save_checkpoint(directory, model, config, source_device, training=None):
    import tensorflow as tf

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    weights = directory / "model.weights.h5"
    # A fresh inference wrapper shares graph weights but has no tracked optimizer.
    # This keeps portable weight artifacts independent of Adam's slot arrays.
    tf.keras.Model(inputs=model.inputs, outputs=model.outputs).save_weights(weights)
    metadata = {
        "schema_version": 1,
        "kind": "translation",
        "architecture_id": config.architecture_id,
        "config": asdict(config),
        "source_sha256": SOURCE_SHA256,
        "implementation_sha256": IMPLEMENTATION_SHA256,
        "source_device": source_device,
        "target_device": "Renishaw",
        "input_layout": "N,L,1",
        "input_length": 1400,
        "input_range": [0, 1],
        "output_head": "out",
        "output_range_policy": "linear_unclipped",
        "output_names": list(model.output_names),
        "parameters": int(model.count_params()),
        "class_registry_sha256": registry_hash(),
        "runtime": runtime("tensorflow"),
        "weights": weights.name,
        "weights_sha256": file_hash(weights),
        "training": training or {"status": "untrained_initialization"},
        "paper_reproduction_certified": False,
    }
    write_json(directory / "metadata.json", metadata)
    return metadata


def load_checkpoint(directory):
    directory = Path(directory)
    from ..artifacts import checkpoint_metadata

    metadata = checkpoint_metadata(directory)
    config = TranslationConfig(**metadata["config"]).validate()
    if metadata.get("kind") != "translation" or metadata.get("schema_version") != 1:
        raise ValueError("Not a supported translation checkpoint")
    if (
        metadata.get("architecture_id") != config.architecture_id
        or metadata.get("source_sha256") != SOURCE_SHA256
    ):
        raise ValueError("Translator architecture provenance mismatch")
    if metadata.get("class_registry_sha256") != registry_hash():
        raise ValueError("Translator class registry differs")
    weights = directory / metadata["weights"]
    if Path(metadata["weights"]).name != metadata["weights"]:
        raise ValueError("Translator weight name must be a simple filename")
    if file_hash(weights) != metadata["weights_sha256"]:
        raise ValueError("Translator checkpoint hash mismatch")
    model = build_model(config)
    if model.count_params() != metadata["parameters"] or model.output_names != metadata.get("output_names"):
        raise ValueError("Translator parameter/output metadata mismatch")
    model.load_weights(weights)
    return model, config, metadata
