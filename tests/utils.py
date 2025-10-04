import os
import time
import tracemalloc
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional, Union

import numpy as np
import pytest
from numpy.typing import DTypeLike

from paibox.types import Shape
from paibox.utils import as_shape


@contextmanager
def measure_time(desc: str) -> Generator[None, Any, None]:
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"{desc} executed in: {elapsed:.2f} secs")


def measure_peak_memory(func, *args, **kwargs) -> float:
    tracemalloc.start()
    _ = func(*args, **kwargs)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return peak / (1024 << 1)  # MiB


def file_not_exist_fail(_fp: Union[str, Path]) -> None:
    """Raise a `pytest.fail` if the file does not exist."""
    fp = Path(_fp)
    if Path.is_file(fp) and not fp.exists():
        pytest.fail(f"Test file {fp} does not exist.")


def gen_random_array(
    shape_: Shape, dtype_: DTypeLike, rng: Optional[np.random.Generator] = None
):
    shape = as_shape(shape_)
    if rng is None:
        rng = np.random.default_rng()

    if dtype_ == np.bool:
        return rng.integers(0, 2, shape, dtype_)
    else:
        return rng.integers(
            np.iinfo(dtype_).min, np.iinfo(dtype_).max + 1, shape, dtype_
        )


CI_INDICATORS = ["CI", "CI_ENV", "GITHUB_ACTIONS"]


def is_ci_env() -> bool:
    return any(os.getenv(var) for var in CI_INDICATORS)


def make_dump_dir(
    test_path: Path, temp_path_fac: pytest.TempPathFactory, dir_name: str = "debug"
) -> Path:
    if is_ci_env():
        p = temp_path_fac.mktemp(dir_name)
    else:
        p = test_path / dir_name

    if not p.is_dir():
        p.mkdir(parents=True, exist_ok=True)
    else:
        for f in p.iterdir():
            f.unlink(missing_ok=True)

    return p
