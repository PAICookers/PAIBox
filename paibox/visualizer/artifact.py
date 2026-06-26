from pathlib import Path

from .model import ViewerModel

DEFAULT_BACKEND = "v2"


def load_artifact(path: str | Path, *, backend: str = DEFAULT_BACKEND) -> ViewerModel:
    if backend != DEFAULT_BACKEND:
        raise ValueError(
            f"unknown visualizer backend {backend!r}; available: {DEFAULT_BACKEND}"
        )

    from .backends.v2.artifact import load_artifact as load_v2_artifact

    return load_v2_artifact(path)
