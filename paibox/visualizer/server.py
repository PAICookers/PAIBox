import webbrowser
from dataclasses import asdict
from pathlib import Path
from socket import socket
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .artifact import DEFAULT_BACKEND, load_artifact_session
from .model import (
    ChipView,
    CoreView,
    IoCoreSummary,
    IoCoreView,
    IoEntryView,
    IoView,
    TensorRegionView,
    ViewerModel,
)


def create_app(artifact: str | Path, *, backend: str = DEFAULT_BACKEND) -> FastAPI:
    """Create a local, artifact-bound visualizer API and static UI app."""
    session = load_artifact_session(artifact, backend=backend)
    model = session.model
    app = FastAPI(title="PAIBox Visualizer")
    app.state.viewer_model = model
    app.state.artifact_session = session

    @app.get("/api/summary")
    def summary() -> dict[str, Any]:
        return model.to_dict()["summary"]

    @app.get("/api/chips")
    def chips() -> list[dict[str, Any]]:
        return [_chip_overview_dict(chip) for chip in model.chips]

    @app.get("/api/cores/{chip_id}/{x}/{y}")
    def core(chip_id: int, x: int, y: int) -> dict[str, Any]:
        _find_core(model, chip_id, x, y)
        return asdict(session.core(chip_id, x, y))

    @app.get("/api/io/summary")
    def io_summary() -> dict[str, Any]:
        return {
            "available": model.io.available,
            "tensors": [asdict(item) for item in model.io.tensors],
            "core_summaries": [asdict(item) for item in model.io.core_summaries],
        }

    @app.get("/api/io/cores/{chip_id}/{x}/{y}")
    def io_core(chip_id: int, x: int, y: int) -> dict[str, Any]:
        _find_core(model, chip_id, x, y)
        return asdict(_build_io_core_view(session.io_view(), chip_id, x, y))

    @app.get("/api/io/regions")
    def io_regions(
        direction: str = Query("input", pattern="^(input|output)$"),
        thread_id: int | None = Query(None, ge=0),
        tensor_name: str | None = Query(None),
        slice_key: str | None = Query(None),
        chip_id: int | None = Query(None, ge=0),
        x: int | None = Query(None),
        y: int | None = Query(None),
        offset: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=500),
    ) -> dict[str, Any]:
        io_view = session.io_view()
        records = (
            io_view.input_regions if direction == "input" else io_view.output_regions
        )
        records = _filter_regions(
            records, thread_id, tensor_name, slice_key, chip_id, x, y
        )
        return _page(records, offset, limit)

    @app.get("/api/io/entries")
    def io_entries(
        direction: str = Query("input", pattern="^(input|output)$"),
        thread_id: int | None = Query(None, ge=0),
        tensor_name: str | None = Query(None),
        slice_key: str | None = Query(None),
        chip_id: int | None = Query(None, ge=0),
        x: int | None = Query(None),
        y: int | None = Query(None),
        offset: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=500),
    ) -> dict[str, Any]:
        io_view = session.io_view()
        records = (
            io_view.input_entries if direction == "input" else io_view.output_entries
        )
        records = _filter_entries(
            records, thread_id, tensor_name, slice_key, chip_id, x, y
        )
        return _page(records, offset, limit)

    @app.get("/api/validation")
    def validation() -> list[dict[str, Any]]:
        return [asdict(item) for item in model.validation]

    @app.get("/favicon.ico")
    def favicon() -> Response:
        return Response(status_code=204)

    ui_dist = _find_ui_dist()
    if ui_dist is not None:
        app.mount("/assets", StaticFiles(directory=ui_dist / "assets"), name="assets")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(ui_dist / "index.html")

    else:

        @app.get("/")
        def missing_ui() -> Response:
            return Response(
                "PAIBox visualizer UI assets are not built.",
                status_code=503,
                media_type="text/plain",
            )

    return app


def serve(
    artifact: str | Path,
    host: str = "127.0.0.1",
    port: int = 0,
    open_browser: bool = True,
    backend: str = DEFAULT_BACKEND,
) -> int:
    if port == 0:
        port = _pick_free_port(host)
    app = create_app(artifact, backend=backend)
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    print(f"serving PAIBox visualizer at http://{host}:{port}")
    if open_browser and port:
        webbrowser.open(f"http://{host}:{port}")
    try:
        server.run()
    except KeyboardInterrupt:
        return 130
    return 0


def _find_core(model: ViewerModel, chip_id: int, x: int, y: int) -> CoreView:
    for chip in model.chips:
        if chip.chip_id == chip_id:
            for core_item in chip.cores:
                if core_item.x == x and core_item.y == y:
                    return core_item
    raise HTTPException(status_code=404, detail="core not found")


def _chip_overview_dict(chip: ChipView) -> dict[str, Any]:
    return {
        "chip_id": chip.chip_id,
        "grid_width": chip.grid_width,
        "grid_height": chip.grid_height,
        "cores": [_core_overview_dict(core_item) for core_item in chip.cores],
    }


def _core_overview_dict(core_item: CoreView) -> dict[str, Any]:
    """Return only the fields needed by chip-level UI panels."""
    return {
        "chip_id": core_item.chip_id,
        "x": core_item.x,
        "y": core_item.y,
        "role": core_item.role,
        "used": core_item.used,
        "source": core_item.source,
        "nodes": list(core_item.nodes),
        "thread_id": core_item.thread_id,
        "core_config": dict(core_item.core_config),
        "io_summary": (
            asdict(core_item.io_summary) if core_item.io_summary is not None else None
        ),
        "global_signal": asdict(core_item.global_signal),
        "frames": asdict(core_item.frames),
        "neurons": {"summary": asdict(core_item.neurons.summary)},
    }


def _build_io_core_view(model: IoView, chip_id: int, x: int, y: int) -> IoCoreView:
    """Return the per-core IO subset used by the right-side IO inspector."""
    summary = next(
        (
            item
            for item in model.core_summaries
            if item.chip_id == chip_id and item.x == x and item.y == y
        ),
        IoCoreSummary(chip_id=chip_id, x=x, y=y),
    )
    input_regions = _filter_regions(
        model.input_regions, None, None, None, chip_id, x, y
    )
    output_regions = _filter_regions(
        model.output_regions, None, None, None, chip_id, x, y
    )
    input_entries = _filter_entries(
        model.input_entries, None, None, None, chip_id, x, y
    )
    output_entries = _filter_entries(
        model.output_entries, None, None, None, chip_id, x, y
    )
    input_buffer_spans = [
        span
        for span in model.input_buffer_spans
        if span.chip_id == chip_id and span.x == x and span.y == y
    ]
    return IoCoreView(
        chip_id=chip_id,
        x=x,
        y=y,
        summary=summary,
        input_regions=input_regions,
        output_regions=output_regions,
        input_buffer_spans=input_buffer_spans,
        input_entries=input_entries,
        output_entries=output_entries,
    )


def _filter_regions(
    records: list[TensorRegionView],
    thread_id: int | None,
    tensor_name: str | None,
    slice_key: str | None,
    chip_id: int | None,
    x: int | None,
    y: int | None,
) -> list[TensorRegionView]:
    return [
        item
        for item in records
        if (thread_id is None or item.thread_id == thread_id)
        and (tensor_name is None or item.tensor_name == tensor_name)
        and (slice_key is None or item.slice_key == slice_key)
        and (chip_id is None or item.chip_id == chip_id)
        and (x is None or item.target_x == x)
        and (y is None or item.target_y == y)
    ]


def _filter_entries(
    records: list[IoEntryView],
    thread_id: int | None,
    tensor_name: str | None,
    slice_key: str | None,
    chip_id: int | None,
    x: int | None,
    y: int | None,
) -> list[IoEntryView]:
    return [
        item
        for item in records
        if (thread_id is None or item.thread_id == thread_id)
        and (tensor_name is None or item.tensor_name == tensor_name)
        and (slice_key is None or item.slice_key == slice_key)
        and (chip_id is None or item.chip_id == chip_id)
        and (x is None or item.target_x == x)
        and (y is None or item.target_y == y)
    ]


def _page(records, offset: int, limit: int) -> dict[str, Any]:
    return {
        "offset": offset,
        "limit": limit,
        "total": len(records),
        "items": [asdict(item) for item in records[offset : offset + limit]],
    }


def _pick_free_port(host: str) -> int:
    with socket() as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _find_ui_dist() -> Path | None:
    """Prefer packaged static assets, with source-tree dist as dev fallback."""
    package_static = Path(__file__).resolve().parent / "static"
    source_dist = (
        Path(__file__).resolve().parents[3] / "tools" / "visualizer_ui" / "dist"
    )
    for path in (package_static, source_dist):
        if (path / "index.html").is_file() and (path / "assets").is_dir():
            return path
    return None
