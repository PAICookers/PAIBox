import subprocess
from pathlib import Path


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


if __name__ == "__main__":
    main()
