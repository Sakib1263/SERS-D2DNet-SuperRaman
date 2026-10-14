"""PyTorch CUDA forward/backward and CPU parity on synthetic inputs."""

from pathlib import Path
import numpy as np
import torch
from sersd2d_superraman.classification.model import SuperRaman
from sersd2d_superraman.provenance import write_json


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this optional hardware check cannot run")
    torch.manual_seed(37)
    model = SuperRaman().cuda()
    x = torch.rand(4, 1, 1400, device="cuda") * 2 - 1
    logits = model(x)
    torch.nn.functional.cross_entropy(logits, torch.arange(4, device="cuda")).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    with torch.inference_mode():
        gpu = model(x).cpu().numpy()
        cpu = model.cpu()(x.cpu()).numpy()
    np.testing.assert_allclose(cpu, gpu, rtol=1e-4, atol=1e-5)
    report = {
        "scope": "synthetic hardware smoke; not model efficacy or deterministic CUDA certification",
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "forward_backward_finite": True,
        "maximum_cpu_cuda_logit_difference": float(np.max(np.abs(cpu - gpu))),
    }
    write_json(Path(__file__).resolve().parents[1] / "docs/audit/cuda_verification.json", report)
    print(report)


if __name__ == "__main__":
    main()
