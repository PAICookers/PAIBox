import warnings
import numpy as np
from numpy.typing import ArrayLike
from typing import Any, ClassVar, Optional, TypedDict, Union

from .base import FullConnectedSyn
from .weight_dtype import weight_width2range
from .lut import LUT

from paibox._logging import get_artifact_logger
from paibox.base import LearnableSys
from paibox.exceptions import ParamNotSimulatedWarning
from paibox.types import WEIGHT_DTYPE, NeuOutType, WeightType
from paibox.utils import arg_check_pos


syn_stdp_log = get_artifact_logger(__name__, "stdp")

SPIKE_CNT_DTYPE = np.uint8
SPIKE_CNT_BIT_MAX = 5
PRE_CNT_MAX = (1 << SPIKE_CNT_BIT_MAX) - 1
POST_CNT_MAX = PRE_CNT_MAX


class STDPLearner(LearnableSys):
    CFLAG_ENABLE_WP_OPTIMIZATION: ClassVar[bool] = False

    def __init__(
        self,
        syn: FullConnectedSyn,
        weight_decay: int = 0,
        upper_weight: Optional[int] = None,
        lower_weight: Optional[int] = None,
        weight_decay_random: bool = False,
        lut: Optional[ArrayLike] = None,
        lut_offset: Optional[int] = None,
        lut_random: Union[bool, ArrayLike] = False,
        random_seed: int = 1,
        learn_by_default: bool = True,
    ) -> None:
        self.syn = syn
        self.weight_decay = WEIGHT_DTYPE(weight_decay)
        self.weight_decay_random = weight_decay_random

        # XXX This feature needs LFSR support
        if weight_decay_random:
            warnings.warn(
                "Random weight decay is enabled, but will not be simulated.",
                ParamNotSimulatedWarning,
            )

        self._set_weight_range(upper_weight, lower_weight)
        self.lut = LUT(lut, lut_offset, lut_random)

        self.plasticity_range = (0, syn.num_in)  # Fixed
        # XXX This feature needs LFSR support
        self.random_seed = arg_check_pos(random_seed, "random seed")  # must >0

        # Spike counters
        self.pre_cnt = np.full((syn.num_in,), PRE_CNT_MAX, dtype=SPIKE_CNT_DTYPE)
        self.post_cnt = np.full((syn.num_out,), POST_CNT_MAX, dtype=SPIKE_CNT_DTYPE)

        # For the init value of 'online_mode_en' of online cores
        self.learn_by_default = learn_by_default
        self.learn(learn_by_default)

    def _set_weight_range(
        self, upper_weight: Optional[int], lower_weight: Optional[int]
    ) -> None:
        _min, _max = weight_width2range[self.syn.weight_width]
        if upper_weight is None:
            self.upper_weight = _max
        else:
            self.upper_weight = upper_weight

        if lower_weight is None:
            self.lower_weight = _min
        else:
            self.lower_weight = lower_weight

        if self.lower_weight > self.upper_weight:
            raise ValueError(
                "lower weight must be less than or equal to upper weight, but got "
                + f"{self.lower_weight} > {self.upper_weight}."
            )

    def learn(self, mode: bool = True) -> None:
        self.syn.weights.setflags(write=mode)
        super().learn(mode)

    def step(self, pre_spike: NeuOutType, post_spike: NeuOutType) -> None:
        # Learning is the function of core, so the (post-)neuron should be working
        self.update_spike_counter(pre_spike, post_spike)
        self.update_weight(self.syn.weights)

    def reset_state(self, *args, **kwargs) -> None:
        self.pre_cnt.fill(PRE_CNT_MAX)
        self.post_cnt.fill(POST_CNT_MAX)

    def update_spike_counter(
        self, pre_spike: NeuOutType, post_spike: NeuOutType
    ) -> None:
        self.pre_cnt[pre_spike > 0] = 0
        self.pre_cnt[pre_spike == 0] += 1
        self.post_cnt[post_spike > 0] = 0
        self.post_cnt[post_spike == 0] += 1

        syn_stdp_log.debug(f"pre cnt:\n{self.pre_cnt}")
        syn_stdp_log.debug(f"post cnt:\n{self.post_cnt}")

        np.clip(self.pre_cnt, 0, PRE_CNT_MAX, out=self.pre_cnt)
        np.clip(self.post_cnt, 0, POST_CNT_MAX, out=self.post_cnt)

    def update_weight(self, w: WeightType) -> None:
        # Long-term potentiation (LTP) & long-term depression (LTD)
        ltp_indices = np.where(self.post_cnt == 0)[0]
        no_ltp_indices = np.where(self.post_cnt > 0)[0]
        ltd_indices = np.where(self.pre_cnt == 0)[0]
        no_ltd_indices = np.where(self.pre_cnt > 0)[0]

        # Post-spike (whether pre-spike or not), LTP
        if len(ltp_indices) > 0:
            delta_t_ltp = np.subtract.outer(
                self.pre_cnt, self.post_cnt[ltp_indices], dtype=np.int8
            )
            delta_lut = self.lut[delta_t_ltp]
            w[:, ltp_indices] += delta_lut
            syn_stdp_log.debug(f"Got LUT, indices: {delta_t_ltp}")
            syn_stdp_log.debug(f"After LTP:\n {w}")

        # Pre-spike but no post-spike, LTD
        if len(ltd_indices) > 0 and len(no_ltp_indices) > 0:
            delta_t_ltd = np.subtract.outer(
                self.pre_cnt[ltd_indices],
                self.post_cnt[no_ltp_indices],
                dtype=np.int8,
            )
            delta_lut = self.lut[delta_t_ltd]
            w[np.ix_(ltd_indices, no_ltp_indices)] += delta_lut
            syn_stdp_log.debug(f"Got LUT, indices: {delta_t_ltd}")
            syn_stdp_log.debug(f"After LTD:\n {w}")

        # No pre-spike but post-spike, weight will decay
        if len(ltp_indices) > 0 and len(no_ltd_indices) > 0:
            w[np.ix_(no_ltd_indices, ltp_indices)] += self.weight_decay

        # Finally, clip the weight to the range
        np.clip(w, self.lower_weight, self.upper_weight, out=w)

    def attrs(self, for_copy: bool = False) -> dict[str, Any]:
        attrs = {
            "weight_decay_value": self.weight_decay,
            "upper_weight": self.upper_weight,
            "lower_weight": self.lower_weight,
            "lut": self.lut.lut,
            # np.bool -> np.uint8
            "lut_random_en": self.lut.lut_random_en.astype(np.uint8),
            "decay_random_en": self.weight_decay_random,
            "random_seed": self.random_seed,
            "online_mode_en": self.learn_by_default,  # init value
            # Attributes for the online neurons
            "plasticity_start": self.plasticity_range[0],
            "plasticity_end": self.plasticity_range[1],
        }

        return attrs


class STDPSynAttrKwds(TypedDict, total=False):
    weight_decay: int
    upper_weight: Optional[int]
    lower_weight: Optional[int]
    weight_decay_random: bool
    lut: Optional[ArrayLike]
    lut_offset: Optional[int]
    lut_random: Union[bool, ArrayLike]
    random_seed: int
    learn_by_default: bool
