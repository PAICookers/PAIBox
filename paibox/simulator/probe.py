from typing import Optional

from paibox.base import PAIBoxObject


class Probe(PAIBoxObject):
    _avoid_name_conflict = True

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
        self.target: PAIBoxObject
        self.attr = attr
        self._check_attr(target)

        super().__init__(name)

    def _check_attr(self, target: PAIBoxObject) -> None:
        if not hasattr(target, self.attr):
            raise AttributeError(
                f"attribute '{self.attr}' not found in target {self.target}."
            )

        self.target = target

    @property
    def _label_txt(self) -> str:
        return f" '{self.name}'" if hasattr(self, "name") else ""

    def __str__(self) -> str:
        return f"<Probe{self._label_txt} of '{self.attr}' of {self.target}>"

    def __repr__(self) -> str:
        return f"<Probe{self._label_txt} at 0x{id(self):x} of '{self.attr}' of {self.target}>"
