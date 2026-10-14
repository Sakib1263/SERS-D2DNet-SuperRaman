"""Explicit backend placement; PyTorch GPU availability does not imply TensorFlow GPU availability."""

from contextlib import contextmanager


@contextmanager
def tensorflow_device(device="cpu"):
    import tensorflow as tf

    if device not in ("cpu", "cuda"):
        raise ValueError("TensorFlow device must be cpu or cuda (GPU 0)")
    if device == "cuda" and not tf.config.list_physical_devices("GPU"):
        raise ValueError("TensorFlow GPU requested but its runtime cannot access one")
    with tf.device("/GPU:0" if device == "cuda" else "/CPU:0"):
        yield
