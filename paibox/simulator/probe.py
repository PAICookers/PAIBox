from typing import Optional

from paibox.base import PAIBoxObject


class Probe(PAIBoxObject):
    def __init__(
        self,
        target: PAIBoxObject,
        attr: str,
        *,
        subtarget: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """
        Arguments:
            - target: the main target.
            - attr: the attribute to probe.
            - subtarget: the subtarget in target. It can probe \
                the attribute such as `target.subtarget.attr`. Optional.
            - name: the name of the probe. Optional.
        """
        super().__init__(name)

        self.target: PAIBoxObject
        self.attr = attr
        self.subtarget = subtarget

        self._check_attr_in_target(target)

    def _check_attr_in_target(self, target: PAIBoxObject) -> None:
        if not self.subtarget:
            self.target = target
            if not hasattr(self.target, self.attr):
                raise AttributeError(
                    f"Attribute '{self.attr}' not found in target {self.target}."
                )
        else:
            if self.subtarget not in target.__dict__.keys():
                raise AttributeError(
                    f"Attribute '{self.attr}' not found in target {self.target}."
                )

            self.target = target.__dict__[self.subtarget]

    def __str__(self) -> str:
        label_txt = f' "{self.name}"'
        based_on_txt = f"{self.subtarget}." if self.subtarget else ""
        return f"<Probe{label_txt} of '{based_on_txt}{self.attr}' of {self.target}>"

    def __repr__(self) -> str:
        label_txt = f' "{self.name}"'
        based_on_txt = f"{self.subtarget}." if self.subtarget else ""
        return f"<Probe{label_txt} at 0x{id(self):x} of '{based_on_txt}{self.attr}' of {self.target}>"

    @property
    def obj(self) -> PAIBoxObject:
        if self.subtarget:
            return getattr(self.target, self.subtarget)

        return self.target

    @property
    def shape_in(self):
        raise NotImplementedError

    @property
    def shape_out(self):
        return 0
