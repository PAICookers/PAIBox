from typing import Any, Optional
from paibox.general import unique_name


class PAIBoxObject:
    def __init__(self, name: Optional[str] = None) -> None:
        self.name: str = unique_name(self, name)


class SliceView:
    def __init__(self, obj: Any, key=slice(None)) -> None:
        self.obj = obj
