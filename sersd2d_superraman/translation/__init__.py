"""TensorFlow stage; backend imports occur only when model operations are requested."""


def build_model(*args, **kwargs):
    from .model import build_model as build

    return build(*args, **kwargs)
