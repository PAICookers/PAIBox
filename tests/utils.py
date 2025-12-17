import os
import time
import tracemalloc
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pytest
from numpy.typing import DTypeLike
from packaging import version as pkg_version

from paibox.types import Shape
from paibox.utils import as_shape


class ParamTestCase(NamedTuple):
    """Parametrized test cases."""

    argnames: str | tuple[str, ...]
    argvalues: Sequence[Any]
    ids: Sequence[str] | None = None


def make_test(
    cases: ParamTestCase,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable) -> Callable:
        return pytest.mark.parametrize(cases.argnames, cases.argvalues, ids=cases.ids)(
            func
        )

    return decorator


class TestCase:
    """Base class for test cases."""

    __test__ = False


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


def file_not_exist_fail(_fp: str | Path) -> None:
    """Raise a `pytest.fail` if the file does not exist."""
    fp = Path(_fp)
    if (not Path.is_file(fp)) or (not fp.exists()):
        pytest.fail(f"{fp} is not a file or does not exist.")


def dir_not_exist_fail(_fp: str | Path) -> None:
    """Raise a `pytest.fail` if the directory does not exist."""
    fp = Path(_fp)
    if (not Path.is_dir(fp)) or (not fp.exists()):
        pytest.fail(f"{fp} is not a directory or does not exist.")


def gen_random_array(
    shape_: Shape, dtype: DTypeLike, rng: np.random.Generator | None = None
):
    shape = as_shape(shape_)
    if rng is None:
        rng = np.random.default_rng()

    if np.issubdtype(dtype, bool):
        return rng.integers(0, 1, shape, dtype, endpoint=True)
    else:
        return rng.integers(
            np.iinfo(dtype).min, np.iinfo(dtype).max, shape, dtype, endpoint=True
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


def skip_if_in_ci_env() -> pytest.MarkDecorator:
    return pytest.mark.skipif(is_ci_env(), reason="Skipped in CI environment")


def skip_if_version_less_than(lib_name: str, min_version: str) -> pytest.MarkDecorator:
    current = version(lib_name)
    return pytest.mark.skipif(
        pkg_version.parse(current) < pkg_version.parse(min_version),
        reason=f"requires {lib_name} >= {min_version}, but installed version is {current}",
    )


def skip_if_version_greater_than(
    lib_name: str, max_version: str
) -> pytest.MarkDecorator:
    current = version(lib_name)
    return pytest.mark.skipif(
        pkg_version.parse(current) > pkg_version.parse(max_version),
        reason=f"requires {lib_name} <= {max_version}, but installed version is {current}",
    )


def skip_if_method_removed(
    cls, method: str, reason: str | None = None
) -> pytest.MarkDecorator:
    if reason is None:
        reason = f"method '{method}' has been removed from {cls.__name__}"

    return pytest.mark.skipif(not hasattr(cls, method), reason=reason)
