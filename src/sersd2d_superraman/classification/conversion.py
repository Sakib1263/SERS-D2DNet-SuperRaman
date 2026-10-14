"""Explicit migration of a user-trusted historical PyTorch checkpoint."""

from pathlib import Path
import sys
import numpy as np
import torch
from .model import SuperRaman, ClassifierConfig, save_checkpoint
from ..preprocessing import classifier_input
from ..provenance import file_hash, fresh_directory


def convert_legacy(
    path, output, config=None, legacy_source_directory=None, parity_spectra=None, trust_pickle=False
):
    if not trust_pickle:
        raise ValueError(
            "Historical object checkpoints require explicit trust_pickle=True; use only your trusted files"
        )
    if legacy_source_directory is not None:
        directory = str(Path(legacy_source_directory).resolve())
        sys.path.insert(0, directory)
    try:
        old = torch.load(path, weights_only=False, map_location="cpu")
    finally:
        if legacy_source_directory is not None:
            sys.path.remove(directory)
    old_model = old.get("model") if isinstance(old, dict) else old
    if isinstance(old_model, torch.nn.DataParallel):
        old_model = old_model.module
    if isinstance(old_model, torch.nn.Module):
        state = old_model.state_dict()
    elif isinstance(old, dict) and "model_state_dict" in old:
        state = old["model_state_dict"]
    elif isinstance(old, dict) and all(isinstance(v, torch.Tensor) for v in old.values()):
        state = old
    else:
        raise ValueError("Unsupported historical checkpoint structure")
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model = SuperRaman(config or ClassifierConfig())
    model.load_state_dict(state, strict=True)
    provenance = {
        "status": "converted_historical_checkpoint",
        "source_file_sha256": file_hash(path),
        "original_training_configuration": "not inferred from parameter arrays",
        "parity_checked": False,
    }
    if parity_spectra is not None and isinstance(old_model, torch.nn.Module):
        x = torch.from_numpy(classifier_input(parity_spectra))
        with torch.inference_mode():
            before, after = old_model.cpu().eval()(x).numpy(), model.eval()(x).numpy()
        if not np.allclose(before, after, rtol=1e-5, atol=1e-6):
            raise ValueError("Converted model does not reproduce historical logits")
        provenance.update(
            parity_checked=True, maximum_absolute_logit_error=float(np.max(np.abs(before - after)))
        )
    if isinstance(old_model, torch.nn.Module) and parity_spectra is None:
        raise ValueError("Object migration requires parity_spectra to verify output compatibility")
    save_checkpoint(fresh_directory(output), model, provenance)
    return provenance
