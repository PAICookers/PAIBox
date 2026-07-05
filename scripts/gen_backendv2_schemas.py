import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKENDV2_ROOT = REPO_ROOT / "paibox" / "backendv2"
SCHEMAS_ROOT = BACKENDV2_ROOT / "schemas"
GENERATED_ROOT = BACKENDV2_ROOT / "generated"

PROTO_SCHEMA = SCHEMAS_ROOT / "compile_artifacts.proto"
PROTO_OUT_DIR = GENERATED_ROOT / "proto"

FBS_SCHEMA = SCHEMAS_ROOT / "compile_artifacts.fbs"
FBS_OUT_DIR = GENERATED_ROOT / "fbs"


def _prepare_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "__init__.py").unlink(missing_ok=True)


def generate_proto(protoc: str) -> None:
    _prepare_output_dir(GENERATED_ROOT)
    _prepare_output_dir(PROTO_OUT_DIR)
    subprocess.run(
        [
            protoc,
            f"-I{SCHEMAS_ROOT}",
            f"--python_out={PROTO_OUT_DIR}",
            f"--pyi_out={PROTO_OUT_DIR}",
            str(PROTO_SCHEMA),
        ],
        check=True,
        cwd=REPO_ROOT,
    )


def generate_fbs(flatc: str) -> None:
    _prepare_output_dir(GENERATED_ROOT)
    _prepare_output_dir(FBS_OUT_DIR)
    with tempfile.TemporaryDirectory(prefix="paibox-fbs-") as tmp:
        tmp_path = Path(tmp)
        subprocess.run(
            [flatc, "--python", "-o", str(tmp_path), str(FBS_SCHEMA)],
            check=True,
            cwd=REPO_ROOT,
        )
        generated_dir = tmp_path / "paibox" / "backendv2" / "generated" / "fbs"
        if not generated_dir.is_dir():
            raise FileNotFoundError(generated_dir)

        for old_file in FBS_OUT_DIR.glob("*.py"):
            old_file.unlink()
        for src_file in generated_dir.glob("*.py"):
            if src_file.name == "__init__.py":
                continue
            shutil.copy2(src_file, FBS_OUT_DIR / src_file.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate backendv2 protobuf and FlatBuffers Python bindings."
    )
    parser.add_argument(
        "--target",
        choices=("proto", "fbs", "all"),
        default="all",
        help="Generated binding family to refresh.",
    )
    parser.add_argument("--protoc", default="protoc", help="Path to protoc.")
    parser.add_argument("--flatc", default="flatc", help="Path to flatc.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target in ("proto", "all"):
        generate_proto(args.protoc)
    if args.target in ("fbs", "all"):
        generate_fbs(args.flatc)


if __name__ == "__main__":
    main()
