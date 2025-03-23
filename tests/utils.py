import pytest
import time
from contextlib import contextmanager
from typing import Any, Generator, Union
from pathlib import Path

__all__ = ["measure_time"]


@contextmanager
def measure_time(desc: str) -> Generator[None, Any, None]:
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"{desc} executed in: {elapsed:.2f} secs")


def file_not_exist_fail(_fp: Union[str, Path]) -> None:
    """Raise a `pytest.fail` if the file does not exist."""
    fp = Path(_fp)
    if Path.is_file(fp) and not fp.exists():
        pytest.fail(f"Test file {fp} does not exist.")
