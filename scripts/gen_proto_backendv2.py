import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path
from shutil import which


def _run_ruff(paths: list[Path]) -> None:
    path_args = [str(path) for path in paths]
    ruff = which("ruff")
    if ruff is None and find_spec("ruff") is None:
        return

    ruff_cmd = [ruff] if ruff is not None else [sys.executable, "-m", "ruff"]
    subprocess.run([*ruff_cmd, "check", "--fix", *path_args], check=True)
    subprocess.run([*ruff_cmd, "format", *path_args], check=True)


def _ensure_ruff_noqa(path: Path, code: str) -> None:
    lines = path.read_text().splitlines()
    if lines and lines[0] == code:
        return
    path.write_text("\n".join([code, *lines]) + "\n")


def _ensure_pyi_nested_class_spacing(path: Path) -> None:
    lines = path.read_text().splitlines()
    spaced: list[str] = []
    for line in lines:
        if line.startswith("    class ") and spaced and spaced[-1] != "":
            spaced.append("")
        spaced.append(line)
    path.write_text("\n".join(spaced) + "\n")


def _strip_pb2_runtime_version_guard(path: Path) -> None:
    """Remove strict gencode/runtime skew checks from generated Python code.

    The project runtime currently pins ``protobuf<7``. Some developer
    environments may still provide a newer ``protoc`` executable, which emits
    a hard version check in ``*_pb2.py`` and blocks import even when the
    generated schema itself remains compatible with the pinned runtime.
    """
    lines = path.read_text().splitlines()
    stripped: list[str] = []
    skip_validate_block = False

    for line in lines:
        if "from google.protobuf import runtime_version as _runtime_version" in line:
            continue

        if line.startswith("_runtime_version.ValidateProtobufRuntimeVersion("):
            skip_validate_block = True
            continue

        if skip_validate_block:
            if line == ")":
                skip_validate_block = False
            continue

        stripped.append(line)

    path.write_text("\n".join(stripped) + "\n")


def main() -> None:
    repo_root = Path.cwd()
    proto_dir = repo_root / "paibox" / "backendv2" / "proto"
    proto_file = proto_dir / "compile_artifacts.proto"
    protoc = which("protoc")
    if protoc is None:
        env_protoc = Path(sys.prefix) / "Library" / "bin" / "protoc.exe"
        if env_protoc.exists():
            protoc = str(env_protoc)

    if protoc is not None:
        cmd = [
            protoc,
            f"-I{proto_dir}",
            f"--python_out={proto_dir}",
            f"--pyi_out={proto_dir}",
            str(proto_file),
        ]
    elif find_spec("grpc_tools.protoc") is not None:
        cmd = [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"-I{proto_dir}",
            f"--python_out={proto_dir}",
            f"--pyi_out={proto_dir}",
            str(proto_file),
        ]
    else:
        raise RuntimeError(
            "No protobuf compiler available. Install 'protoc' into the current "
            "conda environment (recommended) or install 'grpcio-tools'."
        )

    subprocess.run(cmd, check=True, cwd=repo_root)
    generated_files = [
        proto_dir / "compile_artifacts_pb2.py",
        proto_dir / "compile_artifacts_pb2.pyi",
    ]
    _strip_pb2_runtime_version_guard(proto_dir / "compile_artifacts_pb2.py")
    for generated_file in generated_files:
        _ensure_ruff_noqa(generated_file, "# ruff: noqa: I001")
    _run_ruff(generated_files)
    _ensure_pyi_nested_class_spacing(proto_dir / "compile_artifacts_pb2.pyi")


if __name__ == "__main__":
    main()
