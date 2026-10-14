"""Standalone class-matched supervised translation with independent split roles."""

from dataclasses import asdict, dataclass
import time
import numpy as np
from .model import TranslationConfig, build_model, primary_output, save_checkpoint, load_checkpoint
from ..data.splits import assert_disjoint, pseudo_pairs
from ..provenance import fresh_directory, write_json


@dataclass(frozen=True)
class TranslationTrainingConfig:
    epochs: int = 300
    batch_size: int = 16
    learning_rate: float = 5e-4
    early_stopping_patience: int = 30
    plateau_patience: int = 15
    lr_factor: float = 0.1
    minimum_learning_rate: float = 1e-7
    samples_per_class: int = 50
    seed: int = 37

    def validate(self):
        for key in (
            "epochs",
            "batch_size",
            "early_stopping_patience",
            "plateau_patience",
            "samples_per_class",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0 or not 0 < self.lr_factor < 1:
            raise ValueError("Invalid translation optimizer settings")
        if not 0 <= self.minimum_learning_rate <= self.learning_rate:
            raise ValueError("minimum_learning_rate must lie between zero and learning_rate")
        return self


def supervision_targets(spectra, depth=3):
    values = np.asarray(spectra, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 1400:
        raise ValueError("Targets must have shape [N,1400]")
    targets = {"out": values[:, :, None]}
    for level in range(1, depth + 1):
        factor = 2**level
        targets[f"level{level}"] = values.reshape(len(values), 1400 // factor, factor).mean(2)[:, :, None]
    return targets


def train_translation(
    source_train,
    reference_train,
    source_validation,
    reference_validation,
    output,
    model_config=None,
    training_config=None,
):
    import tensorflow as tf

    for batch in (source_train, reference_train, source_validation, reference_validation):
        batch.validate()
    assert_disjoint(source_train, source_validation)
    assert_disjoint(reference_train, reference_validation)
    assert_disjoint(source_train, reference_validation)
    assert_disjoint(reference_train, source_validation)
    if source_train.source_device != source_validation.source_device:
        raise ValueError("Source train/validation devices differ")
    config = (training_config or TranslationTrainingConfig()).validate()
    model_config = (model_config or TranslationConfig()).validate()
    output = fresh_directory(output)
    fit_x, fit_y, fit_manifest = pseudo_pairs(
        source_train, reference_train, config.samples_per_class, config.seed
    )
    val_x, val_y, val_manifest = pseudo_pairs(
        source_validation, reference_validation, config.samples_per_class, config.seed + 1
    )
    model = build_model(model_config, seed=config.seed)
    weights = {"out": 1.0, **{f"level{i}": 1 - i * 0.1 for i in range(1, model_config.depth + 1)}}
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss={k: tf.keras.losses.MeanAbsoluteError() for k in model.output_names},
        loss_weights=weights,
    )
    targets = supervision_targets(fit_y.spectra, model_config.depth)
    write_json(output / "pseudo_pairs.json", {"fit": fit_manifest, "validation": val_manifest})
    provenance = {
        "status": "trained",
        "objective": "weighted_MAE_deep_supervision",
        "head_weights": weights,
        "config": asdict(config),
        "selection": "validation_out_mae",
        "selection_direction": "min",
        "pairing": "class_matched_physically_unpaired",
        "fit_pair_manifest_sha256": fit_manifest["sha256"],
        "validation_pair_manifest_sha256": val_manifest["sha256"],
        "test_used_for_selection": False,
        "fit_data": source_train.identity(),
        "reference_fit_data": reference_train.identity(),
        "validation_data": source_validation.identity(),
        "reference_validation_data": reference_validation.identity(),
        "resume_semantics": "new run; no historical optimizer/RNG resume",
    }
    rng = np.random.default_rng(config.seed)
    best, stale, plateau = float("inf"), 0, 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        order = rng.permutation(len(fit_x.spectra))
        loss_sum = 0.0
        for start in range(0, len(order), config.batch_size):
            rows = order[start : start + config.batch_size]
            model.reset_metrics()
            values = model.train_on_batch(
                [fit_x.spectra[rows, :, None]], {k: v[rows] for k, v in targets.items()}, return_dict=True
            )
            loss = float(values["loss"])
            if not np.isfinite(loss):
                raise ValueError("Translator training produced nonfinite loss")
            loss_sum += loss * len(rows)
        predictions = primary_output(model, val_x.spectra, config.batch_size)
        score = float(np.abs(predictions - val_y.spectra).mean())
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / len(order),
            "validation_out_mae": score,
            "learning_rate": float(model.optimizer.learning_rate.numpy()),
        }
        history.append(row)
        if score < best:
            best, stale, plateau = score, 0, 0
            provenance.update(best_epoch=epoch, best_validation_out_mae=score)
            save_checkpoint(
                output / "checkpoint", model, model_config, source_train.source_device, provenance
            )
        else:
            stale, plateau = stale + 1, plateau + 1
            if plateau >= config.plateau_patience:
                rate = max(config.minimum_learning_rate, row["learning_rate"] * config.lr_factor)
                model.optimizer.learning_rate.assign(rate)
                plateau = 0
        write_json(output / "history.json", history)
        if stale >= config.early_stopping_patience:
            break
    provenance.update(epochs_completed=len(history), seconds=time.perf_counter() - started)
    write_json(output / "run.json", provenance)
    model, _, _ = load_checkpoint(output / "checkpoint")
    return model, provenance
