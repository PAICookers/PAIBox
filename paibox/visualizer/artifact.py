from pathlib import Path
from typing import TYPE_CHECKING

from .model import ViewerModel

if TYPE_CHECKING:
    from .backends.v2.artifact import ArtifactSession

DEFAULT_BACKEND = "v2"


def load_artifact(path: str | Path, *, backend: str = DEFAULT_BACKEND) -> ViewerModel:
    if backend != DEFAULT_BACKEND:
        raise ValueError(
            f"unknown visualizer backend {backend!r}; available: {DEFAULT_BACKEND}"
        )

    from .backends.v2.artifact import load_artifact as load_v2_artifact

    return load_v2_artifact(path)


def load_artifact_session(
    path: str | Path, *, backend: str = DEFAULT_BACKEND
) -> ArtifactSession:
    """Load a summary-first session for the local visualizer server."""
    if backend != DEFAULT_BACKEND:
        raise ValueError(
            f"unknown visualizer backend {backend!r}; available: {DEFAULT_BACKEND}"
        )

    from .backends.v2.artifact import load_artifact_session as load_v2_session

    return load_v2_session(path)
