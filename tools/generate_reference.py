"""Generate a synthetic forward/gradient parity fixture from the supplied classifier."""

from pathlib import Path
import importlib.util
import json
import os
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = Path(os.environ["SERS_ORIGINAL_CLASSIFICATION"])
    sys.path.insert(0, str(directory))
    source = directory / "models_utils.py"
    spec = importlib.util.spec_from_file_location("legacy_models_utils", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    torch.manual_seed(37)
    model = module.SuperRAMAN_V2(input_ch=1, class_num=20).eval()
    x = torch.linspace(-1, 1, 1400).repeat(2, 1, 1)
    x[1] = torch.sin(x[1] * 5)
    # Include the learned singleton-height coordinate in the reference behavior.
    with torch.no_grad():
        model.conv3.shifts[:, 1] = 0.25
    x.requires_grad_(True)
    logits = model(x)
    logits[:, 3].sum().backward()
    arrays = {
        "input": x.detach().numpy(),
        "logits": logits.detach().numpy(),
        "input_gradients": x.grad.numpy(),
    }
    arrays.update({"state:" + k: v.detach().numpy() for k, v in model.state_dict().items()})
    dest = ROOT / "tests/fixtures"
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest / "legacy_classifier_reference.npz", **arrays)
    import hashlib

    metadata = {
        "scope": "synthetic initialization; not pretrained research weights",
        "seed": 37,
        "original_models_utils_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "original_SelfONN_sha256": hashlib.sha256((directory / "SelfONN.py").read_bytes()).hexdigest(),
        "torch": str(torch.__version__),
        "parameter_count": sum(v.numel() for v in model.parameters()),
    }
    (dest / "legacy_classifier_reference.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
