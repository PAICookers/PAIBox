from typing import Optional
from paibox.base import PAIBoxObject


class Probe(PAIBoxObject):
    def __init__(
        self, target: PAIBoxObject, attr: str, name: Optional[str] = None
    ) -> None:
        super().__init__(name)

        self.target = target
        self.attr = attr

        self._check_attr_in_target()

    def _check_attr_in_target(self):
        if not hasattr(self.target, self.attr):
            raise ValueError(
                f"Attribute {self.attr} not found in target {self.target}."
            )

    def __str__(self) -> str:
        label_txt = f' "{self.name}"'
        return f"<Probe{label_txt} of '{self.attr}' of {self.target}>"

    def __repr__(self) -> str:
        label_txt = f' "{self.name}"'
        return f"<Probe{label_txt} at 0x{id(self):x} of '{self.attr}' of {self.target}>"

    @property
    def obj(self) -> PAIBoxObject:
        return self.target

    @property
    def shape_in(self):
        # TODO
        pass

    @property
    def shape_out(self):
        return 0
