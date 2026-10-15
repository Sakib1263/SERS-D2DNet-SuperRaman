"""PyTorch stage; importing this namespace alone does not load PyTorch."""


def build_model(*args, **kwargs):
    from .model import SuperRaman

    return SuperRaman(*args, **kwargs)
