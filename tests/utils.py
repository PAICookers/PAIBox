import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional, Union

import pytest
import numpy as np
from numpy.typing import DTypeLike

from paibox.types import Shape
from paibox.utils import as_shape

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


def gen_random_array(
    shape_: Shape, dtype_: DTypeLike, rng: Optional[np.random.Generator] = None
):
    shape = as_shape(shape_)
    if rng is None:
        rng = np.random.default_rng()

    if dtype_ == np.bool_:
        return rng.integers(0, 2, shape, dtype_)
    else:
        return rng.integers(
            np.iinfo(dtype_).min, np.iinfo(dtype_).max + 1, shape, dtype_
        )
