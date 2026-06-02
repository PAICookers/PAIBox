import subprocess
from pathlib import Path


def _run_ruff(paths: list[Path]) -> None:
    path_args = [str(path) for path in paths]
    subprocess.run(["ruff", "check", "--fix", *path_args], check=True)
    subprocess.run(["ruff", "format", *path_args], check=True)


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


def main() -> None:
    repo_root = Path.cwd()
    proto_dir = repo_root / "paibox" / "backendv2" / "proto"
    proto_file = proto_dir / "compile_artifacts.proto"
    subprocess.run(
        [
            "protoc",
            f"-I{proto_dir}",
            f"--python_out={proto_dir}",
            f"--pyi_out={proto_dir}",
            str(proto_file),
        ],
        check=True,
        cwd=repo_root,
    )
    generated_files = [
        proto_dir / "compile_artifacts_pb2.py",
        proto_dir / "compile_artifacts_pb2.pyi",
    ]
    for generated_file in generated_files:
        _ensure_ruff_noqa(generated_file, "# ruff: noqa: I001")
    _run_ruff(generated_files)
    _ensure_pyi_nested_class_spacing(proto_dir / "compile_artifacts_pb2.pyi")


if __name__ == "__main__":
    main()
