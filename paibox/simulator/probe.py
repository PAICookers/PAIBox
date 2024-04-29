from typing import Optional

from paibox.base import PAIBoxObject


class Probe(PAIBoxObject):
    def __init__(
        self,
        target: PAIBoxObject,
        attr: str,
        *,
        name: Optional[str] = None,
    ) -> None:
        """
        Arguments:
            - target: the main target.
            - attr: the attribute to probe.
            - name: the name of the probe. Optional.
        """
        super().__init__(name)

        self.target: PAIBoxObject
        self.attr = attr
        self._check_attr(target)

    def _check_attr(self, target: PAIBoxObject) -> None:
        if not hasattr(target, self.attr):
            raise AttributeError(
                f"Attribute '{self.attr}' not found in target {self.target}."
            )

        self.target = target

    def __str__(self) -> str:
        label_txt = f" '{self.name}'"
        return f"<Probe{label_txt} of '{self.attr}' of {self.target}>"

    def __repr__(self) -> str:
        label_txt = f" '{self.name}'"
        return f"<Probe{label_txt} at 0x{id(self):x} of '{self.attr}' of {self.target}>"
