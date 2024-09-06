import time
from contextlib import contextmanager
from typing import Any, Generator


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
