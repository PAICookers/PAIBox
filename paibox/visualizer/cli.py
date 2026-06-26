from pathlib import Path
from typing import NoReturn

import click

from .artifact import DEFAULT_BACKEND, load_artifact
from .backends.v2.errors import FrameDecodeError

BACKEND_CHOICE = click.Choice([DEFAULT_BACKEND])
ARTIFACT_PATH = click.Path(exists=True, path_type=Path)


def _serve_options(command):
    command = click.option(
        "--backend",
        default=DEFAULT_BACKEND,
        show_default=True,
        type=BACKEND_CHOICE,
        help="Artifact backend.",
    )(command)
    command = click.option(
        "--no-browser",
        is_flag=True,
        help="Do not open a browser after starting the server.",
    )(command)
    command = click.option(
        "--port",
        default=0,
        show_default=True,
        type=int,
        help="Port to bind; 0 picks a free port.",
    )(command)
    command = click.option(
        "--host", default="127.0.0.1", show_default=True, help="Host to bind."
    )(command)
    command = click.option(
        "--artifact",
        type=ARTIFACT_PATH,
        help="Compile artifact file or directory to visualize.",
    )(command)
    return command


def _handle_decode_error(artifact: Path, exc: FrameDecodeError) -> NoReturn:
    click.echo(f"artifact={artifact.resolve()}")
    click.echo(f"frame decode error: {exc}")
    raise click.exceptions.Exit(1)


def _run_validate(artifact: Path, *, backend: str, as_json: bool) -> None:
    try:
        model = load_artifact(artifact, backend=backend)
    except FrameDecodeError as exc:
        _handle_decode_error(artifact, exc)

    if as_json:
        import json

        click.echo(json.dumps(model.to_dict()["validation"], indent=2, sort_keys=True))
    else:
        for item in model.validation:
            click.echo(f"{item.severity}: {item.code}: {item.message}")
        click.echo(
            f"errors={model.summary.validation_error_count} "
            f"warnings={model.summary.validation_warning_count}"
        )

    if model.summary.validation_error_count:
        raise click.exceptions.Exit(1)


def _run_serve(
    artifact: Path, backend: str, host: str, port: int, open_browser: bool
) -> None:
    try:
        from .server import serve as serve_app
    except ModuleNotFoundError as exc:
        if exc.name in {"fastapi", "uvicorn"}:
            click.echo(
                "paiviz serve requires the optional visualizer dependencies. "
                "Install them with `pip install 'paibox[visualizer]'`."
            )
            raise click.exceptions.Exit(2) from exc
        raise

    try:
        exit_code = serve_app(
            artifact,
            host=host,
            port=port,
            open_browser=open_browser,
            backend=backend,
        )
    except FrameDecodeError as exc:
        _handle_decode_error(artifact, exc)
    if exit_code:
        raise click.exceptions.Exit(exit_code)


@click.group(
    invoke_without_command=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
@_serve_options
@click.pass_context
def cli(
    ctx: click.Context,
    artifact: Path | None,
    host: str,
    port: int,
    no_browser: bool,
    backend: str,
) -> None:
    """Validate or serve PAIBox visualizer artifacts."""
    if ctx.invoked_subcommand is not None:
        return
    if artifact is None:
        raise click.UsageError("Missing option '--artifact'.")
    _run_serve(artifact, backend, host, port, not no_browser)


@cli.command()
@_serve_options
def serve(
    artifact: Path | None, host: str, port: int, no_browser: bool, backend: str
) -> None:
    """Serve the visualizer web UI."""
    if artifact is None:
        raise click.UsageError("Missing option '--artifact'.")
    _run_serve(artifact, backend, host, port, not no_browser)


@cli.command()
@click.option(
    "--artifact",
    required=True,
    type=ARTIFACT_PATH,
    help="Compile artifact file or directory to validate.",
)
@click.option(
    "--json", "as_json", is_flag=True, help="Print validation messages as JSON."
)
@click.option(
    "--backend",
    default=DEFAULT_BACKEND,
    show_default=True,
    type=BACKEND_CHOICE,
    help="Artifact backend.",
)
def validate(artifact: Path, as_json: bool, backend: str) -> None:
    """Validate an artifact without starting the UI."""
    _run_validate(artifact, backend=backend, as_json=as_json)


def main(argv: list[str] | None = None) -> int:
    try:
        return cli.main(args=argv, prog_name="paiviz", standalone_mode=False) or 0
    except click.ClickException as exc:
        exc.show()
        return exc.exit_code
    except click.exceptions.Exit as exc:
        return int(exc.exit_code or 0)


if __name__ == "__main__":
    raise SystemExit(main())
