"""Model-only repeated latency with one averaging step and explicit device synchronization."""

import time
import numpy as np


def benchmark(model, spectra, kind, device="cpu", warmup=5, repetitions=20):
    if warmup < 0 or repetitions < 1:
        raise ValueError("warmup must be nonnegative and repetitions positive")
    if kind == "classification":
        import torch
        from ..preprocessing import classifier_input

        x = torch.from_numpy(classifier_input(spectra)).to(device)
        model = model.to(device).eval()

        def call():
            with torch.inference_mode():
                model(x)
            if str(device).startswith("cuda"):
                torch.cuda.synchronize(device)

        parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    elif kind == "translation":
        import tensorflow as tf

        if device not in ("cpu", "cuda"):
            raise ValueError("TensorFlow benchmark device must be cpu or cuda (GPU 0)")
        if device == "cuda" and not tf.config.list_physical_devices("GPU"):
            raise ValueError("TensorFlow GPU requested but its runtime cannot access one")
        placement = "/CPU:0" if device == "cpu" else "/GPU:0"
        with tf.device(placement):
            x = tf.convert_to_tensor(np.asarray(spectra, np.float32)[:, :, None])

        def call():
            # Reading every output ensures device computation has finished.
            with tf.device(placement):
                for value in model([x], training=False):
                    value.numpy()

        parameter_bytes = sum(
            np.prod(w.shape).item() * np.dtype(str(w.dtype)).itemsize for w in model.weights
        )
    else:
        raise ValueError("Unknown benchmark stage")
    for _ in range(warmup):
        call()
    timings = []
    for _ in range(repetitions):
        if kind == "classification" and str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        call()
        timings.append((time.perf_counter() - start) * 1000)
    return {
        "scope": "model only; input transform and host/device input transfer excluded",
        "batch_size": len(spectra),
        "requested_execution_device": device,
        "warmup": warmup,
        "repetitions": repetitions,
        "mean_batch_ms": float(np.mean(timings)),
        "median_batch_ms": float(np.median(timings)),
        "std_batch_ms": float(np.std(timings)),
        "p95_batch_ms": float(np.percentile(timings, 95)),
        "mean_ms_per_spectrum_at_this_batch_size": float(np.mean(timings) / len(spectra)),
        "parameter_bytes": int(parameter_bytes),
        "individual_batch_ms": timings,
    }
