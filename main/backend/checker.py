from paicorelib import LCN_EX
from paicorelib import WeightPrecision as WP

__all__ = ["ConfigChecker"]


class _Checker:
    pass


class ConfigChecker(_Checker):
    @staticmethod
    def n_config_estimate(n_neuron: int, wp: WP, lcn_ex: LCN_EX) -> int:
        _base = n_neuron * (1 << wp) * (1 << lcn_ex)

        n_total = 3 + 3 + (1 + 4 * _base) + (1 + 18 * _base)

        return n_total
