"""Measure paiviz startup and one on-demand core decode in isolation.

The command intentionally runs one artifact load per process. It is suitable
for large artifacts without retaining a growing collection of viewer models.
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import pstats
import resource
import time
from pathlib import Path


def _rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument(
        "--core",
        help="Optional offline core to decode as X,Y; defaults to the first used offline core.",
    )
    parser.add_argument("--cprofile", type=Path, help="Write a cProfile output file.")
    parser.add_argument(
        "--max-rss-mib",
        type=float,
        default=1024,
        help="Fail after the run if peak RSS exceeds this bound (default: 1024).",
    )
    parser.add_argument(
        "--trace-memory",
        action="store_true",
        help="Enable tracemalloc; this adds substantial measurement overhead.",
    )
    parser.add_argument(
        "--load-io",
        action="store_true",
        help="Materialize the full IO view once, matching the first IO Map request.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.artifact.exists():
        raise SystemExit(f"artifact does not exist: {args.artifact}")

    tracer = None
    if args.trace_memory:
        import tracemalloc

        tracer = tracemalloc
        tracer.start()

    from paibox.visualizer.server import create_app

    profile = cProfile.Profile() if args.cprofile else None
    start = time.perf_counter()
    if profile is not None:
        profile.enable()
    app = create_app(args.artifact)
    startup_seconds = time.perf_counter() - start
    if profile is not None:
        profile.disable()
        profile.dump_stats(str(args.cprofile))
        pstats.Stats(profile).sort_stats("cumulative").print_stats(20)

    model = app.state.viewer_model
    print(
        f"startup_seconds={startup_seconds:.3f} peak_rss_mib={_rss_mib():.1f} "
        f"cores={len(model.chips[0].cores)} io_entries="
        f"{len(model.io.input_entries) + len(model.io.output_entries)}"
    )

    core = None
    if args.core:
        try:
            core = tuple(int(value) for value in args.core.split(",", 1))
        except ValueError as exc:
            raise SystemExit("--core must be X,Y") from exc
    if core is None:
        core = next(
            (item.x, item.y)
            for item in model.chips[0].cores
            if item.used and item.role == "offline"
        )
    endpoints = {
        route.path: route.endpoint for route in app.routes if hasattr(route, "endpoint")
    }
    start = time.perf_counter()
    endpoints["/api/cores/{chip_id}/{x}/{y}"](0, core[0], core[1])
    print(
        f"core_seconds={time.perf_counter() - start:.3f} peak_rss_mib={_rss_mib():.1f}"
    )

    if args.load_io:
        start = time.perf_counter()
        endpoints["/api/io/cores/{chip_id}/{x}/{y}"](0, core[0], core[1])
        print(
            f"io_seconds={time.perf_counter() - start:.3f} peak_rss_mib={_rss_mib():.1f}"
        )

    if tracer is not None:
        current, peak = tracer.get_traced_memory()
        print(
            f"tracemalloc_current_mib={current / 2**20:.1f} peak_mib={peak / 2**20:.1f}"
        )
        tracer.stop()
    del app
    gc.collect()
    if _rss_mib() > args.max_rss_mib:
        print(f"peak RSS exceeded --max-rss-mib={args.max_rss_mib}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
