from typing import Any


class FrameDecodeError(ValueError):
    """Raised when a visualizer frame stream is structurally invalid.

    The decoder is intentionally frame-first. Errors therefore keep raw frame
    provenance and contextual fields so CLI/API failures can point back to the
    exact generated word instead of only reporting a high-level artifact issue.
    """

    def __init__(
        self,
        message: str,
        frame_index: int | None = None,
        raw_frame: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.reason = message
        self.frame_index = frame_index
        self.raw_frame = raw_frame
        self.context = context or {}
        details = []
        if frame_index is not None:
            details.append(f"frame_index={frame_index}")
        if raw_frame is not None:
            details.append(f"raw=0x{raw_frame:016x}")
        details.extend(f"{key}={value}" for key, value in self.context.items())
        suffix = f" ({', '.join(details)})" if details else ""
        super().__init__(f"{message}{suffix}")

    def with_context(self, **context: Any) -> "FrameDecodeError":
        """Return a new error with outer decoder context prepended.

        Lower-level parsers know frame/package details; callers add artifact or
        core coordinates. Merging this way keeps both layers visible while
        preserving the original reason and raw frame.
        """
        return FrameDecodeError(
            self.reason, self.frame_index, self.raw_frame, {**context, **self.context}
        )
