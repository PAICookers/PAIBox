"""Opt-in diagnostics for backendv2 hot paths."""

import builtins
import os


def debug_print(*args: object, **kwargs: object) -> None:
    """Emit legacy diagnostics only when ``PAIBOX_DEBUG=1`` is set."""
    if os.environ.get("PAIBOX_DEBUG") == "1":
        builtins.print(*args, **kwargs)
