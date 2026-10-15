"""Independent implementation of the supplied SuperRaman tensor contract.

The legacy profile preserves shared two-coordinate shifts, spectral-axis concatenation,
power-major expansion, and state names. Bounded 1D shifts are an explicit new profile.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ..data.registry import registry_hash
from ..preprocessing import CLASSIFIER_TRANSFORM, classifier_input
from ..provenance import file_hash, runtime, publish_checkpoint


@dataclass(frozen=True)
class ClassifierConfig:
    filters: int = 16
    q: int = 3
    max_shift: float = 2.0
    shift_mode: str = "legacy_fractional"

    def validate(self):
        if type(self.filters) is not int or self.filters < 1 or type(self.q) is not int or self.q < 1:
            raise ValueError("filters and q must be positive integers")
        if not math.isfinite(self.max_shift) or self.max_shift < 0:
            raise ValueError("max_shift must be finite and nonnegative")
        if self.shift_mode not in ("legacy_fractional", "bounded_fractional_1d", "fixed_integer_1d"):
            raise ValueError("Unknown shift profile")
        return self

    @property
    def architecture_id(self):
        return f"superraman_v2_n{self.filters}_q{self.q}_g{self.max_shift:g}_{self.shift_mode}"


class PolynomialShift1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, config):
        super().__init__()
        self.config = config
        self.weights = nn.Parameter(torch.empty(out_channels, config.q * in_channels, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        shifts = torch.empty(in_channels, 2).uniform_(-config.max_shift, config.max_shift)
        shifts[:, 1] = 0
        if config.shift_mode == "fixed_integer_1d":
            shifts.round_()
            self.register_buffer("shifts", shifts)
        else:
            self.shifts = nn.Parameter(shifts)
        nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain("tanh"))
        bound = 1 / math.sqrt(config.q * in_channels * kernel_size)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        shifts = self.shifts
        if self.config.shift_mode != "legacy_fractional":
            shifts = torch.stack(
                (
                    shifts[:, 0].clamp(-self.config.max_shift, self.config.max_shift),
                    torch.zeros_like(shifts[:, 0]),
                ),
                dim=1,
            )
        # Preserving this zero-shift branch is necessary for legacy input-gradient parity.
        if torch.any(shifts != 0):
            image = x.permute(1, 0, 2).unsqueeze(-2)
            channels, _, height, width = image.shape
            affine = torch.eye(3, dtype=x.dtype, device=x.device).repeat(channels, 1, 1)
            affine[:, 0, 2:3] += shifts[:, :1] / (width / 2)
            affine[:, 1, 2:3] += shifts[:, 1:] / (height / 2)
            grid = F.affine_grid(affine[:, :2, :3], image.size(), align_corners=False)
            x = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            x = x.permute(1, 0, 2, 3).squeeze(-2)
        if self.config.q != 1:
            x = torch.cat([x**power for power in range(1, self.config.q + 1)], dim=1)
        return F.conv1d(x, self.weights, self.bias, padding=1)


class SuperRaman(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = (config or ClassifierConfig()).validate()
        for k in (1, 3, 5, 7):
            setattr(self, f"conv{k}", PolynomialShift1d(1, self.config.filters, k, self.config))
        self.pool = nn.AvgPool1d(4)
        self.BN1 = nn.BatchNorm1d(self.config.filters)
        self.TH = nn.Tanh()
        self.FT = nn.Flatten()
        self.classifier = nn.Linear(self.config.filters * 1399, 20)

    def forward(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (1, 1400):
            raise ValueError("SuperRaman expects [N,1,1400]")
        branches = [getattr(self, f"conv{k}")(x) for k in (1, 3, 5, 7)]
        return self.classifier(self.FT(self.TH(self.BN1(self.pool(torch.cat(branches, dim=-1))))))


def predict_logits(model, spectra, batch_size=64, device="cpu"):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    values = classifier_input(spectra)
    model = model.to(device).eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            outputs.append(
                model(torch.from_numpy(values[start : start + batch_size]).to(device)).cpu().numpy()
            )
    result = np.concatenate(outputs)
    if result.shape != (len(values), 20) or not np.isfinite(result).all():
        raise ValueError("Classifier produced invalid logits")
    return result


def save_checkpoint(directory, model, training=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    metadata = {
        "schema_version": 1,
        "kind": "classification",
        "config": asdict(model.config),
        "architecture_id": model.config.architecture_id,
        "source_sha256": file_hash(__file__),
        "source_device": "Renishaw",
        "input_layout": "N,1,L",
        "input_length": 1400,
        "input_transform": CLASSIFIER_TRANSFORM,
        "class_registry_sha256": registry_hash(),
        "parameters": sum(x.numel() for x in model.parameters()),
        "runtime": runtime("torch"),
        "training": training or {"status": "untrained_initialization"},
        "paper_reproduction_certified": False,
    }
    return publish_checkpoint(directory, "model.pt", lambda path: torch.save(state, path), metadata)


def load_checkpoint(directory, device="cpu"):
    directory = Path(directory)
    from ..artifacts import checkpoint_metadata

    metadata = checkpoint_metadata(directory)
    config = ClassifierConfig(**metadata["config"]).validate()
    if metadata.get("kind") != "classification" or metadata.get("schema_version") != 1:
        raise ValueError("Not a supported classifier checkpoint")
    if (
        metadata.get("architecture_id") != config.architecture_id
        or metadata.get("class_registry_sha256") != registry_hash()
    ):
        raise ValueError("Classifier architecture/class mapping mismatch")
    if metadata.get("input_transform") != CLASSIFIER_TRANSFORM:
        raise ValueError("Classifier checkpoint requires an unsupported input transform")
    weights = directory / metadata["weights"]
    if Path(metadata["weights"]).name != metadata["weights"]:
        raise ValueError("Classifier weight name must be a simple filename")
    if file_hash(weights) != metadata["weights_sha256"]:
        raise ValueError("Classifier checkpoint hash mismatch")
    model = SuperRaman(config)
    if sum(p.numel() for p in model.parameters()) != metadata["parameters"]:
        raise ValueError("Classifier parameter metadata mismatch")
    model.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"), strict=True)
    return model.to(device).eval(), metadata
