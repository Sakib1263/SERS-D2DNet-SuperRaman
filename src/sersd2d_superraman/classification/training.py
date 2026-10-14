"""Renishaw training with validation-only model selection and no test input."""

from dataclasses import asdict, dataclass
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .model import ClassifierConfig, SuperRaman, save_checkpoint, load_checkpoint
from ..data.splits import assert_disjoint
from ..preprocessing import classifier_input
from ..provenance import fresh_directory, write_json


@dataclass(frozen=True)
class ClassifierTrainingConfig:
    epochs: int = 500
    batch_size: int = 16
    learning_rate: float = 3e-5
    weight_decay: float = 1e-4
    early_stopping_patience: int = 50
    plateau_patience: int = 25
    lr_factor: float = 0.1
    seed: int = 37

    def validate(self):
        for key in ("epochs", "batch_size", "early_stopping_patience", "plateau_patience"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if (
            not np.isfinite(self.learning_rate)
            or self.learning_rate <= 0
            or not np.isfinite(self.weight_decay)
            or self.weight_decay < 0
        ):
            raise ValueError("Invalid optimizer settings")
        if not 0 < self.lr_factor < 1:
            raise ValueError("lr_factor must lie between zero and one")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        return self


def train_classifier(train, validation, output, model_config=None, training_config=None, device="cpu"):
    train.validate()
    validation.validate()
    assert_disjoint(train, validation)
    if train.source_device != "Renishaw" or validation.source_device != "Renishaw":
        raise ValueError("The reusable classifier trains and selects on Renishaw only")
    if (
        train.labels is None
        or validation.labels is None
        or set(train.labels) != set(range(20))
        or set(validation.labels) != set(range(20))
    ):
        raise ValueError("Classifier fitting requires all 20 classes and labeled validation")
    config = (training_config or ClassifierTrainingConfig()).validate()
    model_config = (model_config or ClassifierConfig()).validate()
    output = fresh_directory(output)
    torch.manual_seed(config.seed)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but PyTorch cannot access it")
    generator = torch.Generator().manual_seed(config.seed)
    datasets = [
        TensorDataset(torch.from_numpy(classifier_input(b.spectra)), torch.from_numpy(b.labels))
        for b in (train, validation)
    ]
    loaders = [
        DataLoader(ds, batch_size=config.batch_size, shuffle=(i == 0), generator=generator)
        for i, ds in enumerate(datasets)
    ]
    model = SuperRaman(model_config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.lr_factor, patience=config.plateau_patience
    )
    loss_function = nn.CrossEntropyLoss()
    best = float("inf")
    no_improvement = 0
    history = []
    started = time.perf_counter()
    provenance = {
        "status": "trained",
        "synthetic": bool(train.metadata.get("synthetic", False)),
        "protocol": "maintained_independent_validation",
        "objective": "cross_entropy_on_logits",
        "selection": "validation_loss",
        "selection_direction": "min",
        "config": asdict(config),
        "fit_data": train.identity(),
        "validation_data": validation.identity(),
        "test_used_for_selection": False,
        "device": device,
        "cuda_backward_determinism": "not guaranteed for grid_sample",
        "resume_semantics": "new run; no historical optimizer/RNG resume",
    }
    write_json(output / "run.json", provenance)
    for epoch in range(1, config.epochs + 1):
        results = []
        for role, loader in enumerate(loaders):
            model.train(role == 0)
            total_loss, correct, count = 0.0, 0, 0
            with torch.set_grad_enabled(role == 0):
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    if role == 0:
                        optimizer.zero_grad(set_to_none=True)
                    logits = model(x)
                    loss = loss_function(logits, y)
                    if not torch.isfinite(loss):
                        raise ValueError("Classifier training produced nonfinite loss")
                    if role == 0:
                        loss.backward()
                        optimizer.step()
                    total_loss += float(loss.detach()) * len(y)
                    correct += int((logits.argmax(1) == y).sum())
                    count += len(y)
            results.append((total_loss / count, correct / count))
        row = {
            "epoch": epoch,
            "train_loss": results[0][0],
            "validation_loss": results[1][0],
            "train_accuracy": results[0][1],
            "validation_accuracy": results[1][1],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        print(
            f"SuperRaman epoch {epoch}/{config.epochs}: "
            f"train_ce={row['train_loss']:.6g}, val_ce={row['validation_loss']:.6g}, "
            f"val_accuracy={row['validation_accuracy']:.4f}, lr={row['learning_rate']:.3g}",
            flush=True,
        )
        history.append(row)
        scheduler.step(row["validation_loss"])
        if row["validation_loss"] < best:
            best, no_improvement = row["validation_loss"], 0
            provenance.update(best_epoch=epoch, best_validation_loss=best)
            save_checkpoint(output / "checkpoint", model, provenance)
        else:
            no_improvement += 1
        write_json(output / "history.json", history)
        if no_improvement >= config.early_stopping_patience:
            break
    provenance.update(epochs_completed=len(history), seconds=time.perf_counter() - started)
    write_json(output / "run.json", provenance)
    from ..provenance import read_json

    metadata = read_json(output / "checkpoint/metadata.json")
    metadata["training"] = provenance
    write_json(output / "checkpoint/metadata.json", metadata)
    model, _ = load_checkpoint(output / "checkpoint", device=device)
    return model, provenance
