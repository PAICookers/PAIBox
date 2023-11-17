import time
from dataclasses import dataclass

import numpy as np
import pytest

from paibox.libpaicore.v2.reg_types import LCNExtensionType as LCNX


@dataclass
class AxonGroup:
    lcnx: LCNX
    test_axons: np.ndarray
    indices: np.ndarray


greedy_grouping_lcnx_opt_data = [
    (
        np.array([100, 200, 300, 400]),
        4,
        [
            AxonGroup(
                LCNX.LCN_1X, np.array([100, 200, 300, 400]), np.array([0, 1, 2, 3])
            )
        ],
    ),
    (
        np.array([100, 200, 300, 400]),
        3,
        [
            AxonGroup(LCNX.LCN_1X, np.array([100, 200, 300]), np.array([0, 1, 2])),
            AxonGroup(LCNX.LCN_1X, np.array([400]), np.array([3])),
        ],
    ),
    (
        # Length is 16
        np.array(
            [
                100,
                200,
                300,
                1300,
                800,
                1500,
                400,
                700,
                1200,
                1000,
                500,
                1100,
                600,
                900,
                1400,
                1600,
            ]
        ),
        4,
        [
            AxonGroup(
                LCNX.LCN_1X, np.array([100, 200, 300, 400]), np.array([0, 1, 2, 6])
            ),
            AxonGroup(LCNX.LCN_1X, np.array([500, 600]), np.array([10, 12])),
            AxonGroup(LCNX.LCN_1X, np.array([700]), np.array([7])),
            AxonGroup(LCNX.LCN_1X, np.array([800]), np.array([4])),
            AxonGroup(LCNX.LCN_1X, np.array([900]), np.array([13])),
            AxonGroup(LCNX.LCN_1X, np.array([1000]), np.array([9])),
            AxonGroup(LCNX.LCN_1X, np.array([1100]), np.array([11])),
            AxonGroup(LCNX.LCN_2X, np.array([1200]), np.array([8])),
            AxonGroup(LCNX.LCN_2X, np.array([1300]), np.array([3])),
            AxonGroup(LCNX.LCN_2X, np.array([1400]), np.array([14])),
            AxonGroup(LCNX.LCN_2X, np.array([1500]), np.array([5])),
            AxonGroup(LCNX.LCN_2X, np.array([1600]), np.array([15])),
        ],
    ),
    (
        # Length is 10
        np.array([512, 513, 1200, 2048, 2047, 200, 400]),
        4,
        [
            AxonGroup(LCNX.LCN_1X, np.array([200, 400, 512]), np.array([5, 6, 0])),
            AxonGroup(LCNX.LCN_2X, np.array([513, 1200]), np.array([1, 2])),
            AxonGroup(LCNX.LCN_2X, np.array([2047]), np.array([4])),
            AxonGroup(LCNX.LCN_2X, np.array([2048]), np.array([3])),
        ],
    ),
]

greedy_grouping_core_opt_data = [
    (
        np.array([100, 200, 300, 400]),
        4,
        [
            AxonGroup(
                LCNX.LCN_1X, np.array([100, 200, 300, 400]), np.array([0, 1, 2, 3])
            )
        ],
    ),
    (
        # Length is 16
        np.array(
            [
                100,
                200,
                300,
                1300,
                800,
                1500,
                400,
                700,
                1200,
                1000,
                500,
                1100,
                600,
                900,
                1400,
                1600,
            ]
        ),
        4,
        [
            AxonGroup(
                LCNX.LCN_1X, np.array([100, 200, 300, 400]), np.array([0, 1, 2, 6])
            ),
            AxonGroup(LCNX.LCN_1X, np.array([500, 600]), np.array([10, 12])),
            AxonGroup(LCNX.LCN_2X, np.array([700, 800]), np.array([7, 4])),
            AxonGroup(LCNX.LCN_2X, np.array([900, 1000]), np.array([13, 9])),
            AxonGroup(LCNX.LCN_2X, np.array([1100, 1200]), np.array([11, 8])),
            AxonGroup(LCNX.LCN_2X, np.array([1300]), np.array([3])),
            AxonGroup(LCNX.LCN_2X, np.array([1400]), np.array([14])),
            AxonGroup(LCNX.LCN_2X, np.array([1500]), np.array([5])),
            AxonGroup(LCNX.LCN_2X, np.array([1600]), np.array([15])),
        ],
    ),
]

test_case = [
    (
        np.array([100, 200, 300, 400]),
        1,
        [
            AxonGroup(LCNX.LCN_1X, np.array([100]), np.array([0])),
            AxonGroup(LCNX.LCN_1X, np.array([200]), np.array([1])),
            AxonGroup(LCNX.LCN_1X, np.array([300]), np.array([2])),
            AxonGroup(LCNX.LCN_1X, np.array([400]), np.array([3])),
        ],
    )
]


@pytest.mark.parametrize(
    "test_axons, n_neuron_in_core, expected_axons_grouped", test_case
)
def test_greedy_grouping_lcnx_opt(test_axons, n_neuron_in_core, expected_axons_grouped):
    """LCN extension Optimization Strategy

    Description:
        Always looking for the minimum LCN extension that the group of axons is **ALL** satisfied.

    For testing, the `n_neuron_in_core` is much smaller than 512, such as 4, 8, etc.
    """
    # Sort the axons group from minimum to maximum
    indices = np.argsort(test_axons, kind="heapsort")
    test_axons.sort(kind="heapsort")

    lcn_each = []

    # 根据每个neuron的axon数，计算出对应的LCN
    for i in range(len(test_axons)):
        if test_axons[i] <= 1152:
            lcn_x = LCNX.LCN_1X
        elif test_axons[i] <= 1152 * 2:
            lcn_x = LCNX.LCN_2X
        elif test_axons[i] <= 1152 * 4:
            lcn_x = LCNX.LCN_4X
        elif test_axons[i] <= 1152 * 8:
            lcn_x = LCNX.LCN_8X
        elif test_axons[i] <= 1152 * 16:
            lcn_x = LCNX.LCN_16X
        elif test_axons[i] <= 1152 * 32:
            lcn_x = LCNX.LCN_32X
        elif test_axons[i] <= 1152 * 64:
            lcn_x = LCNX.LCN_64X
        else:
            raise Exception

        lcn_each.append(lcn_x)
        # lcn_x = np.ceil(test_axons[i] / 1152) - 1

    # Traverse the lcn and put them in core
    def _get_limit(indice: int):
        """Get the limit when axon is [indice]"""
        lcn_ex_max_in_core = lcn_each[indice]
        n_max_in_core = int(
            n_neuron_in_core / (2**lcn_ex_max_in_core)
        )  # the number of neurons in a core
        axons_max_in_core = 1152 * (
            lcn_ex_max_in_core - LCNX.LCN_1X + 1
        )  # LCN下的最大axon数

        return lcn_ex_max_in_core, n_max_in_core, axons_max_in_core

    i = i_start = 0  # i_start：当前core的第一个axon在排序后的axon数组中的index
    axons_grouped = []  # The length of it is the number of cores needed.

    t1 = time.time()  # Use for timing

    while i_start < len(test_axons):
        axons_group_sum = test_axons[i_start]
        l, n, a = _get_limit(i_start)

        if i != len(test_axons):
            i += 1

        while True:
            if i == len(test_axons):
                break

            if not ((i - i_start) + 1 <= n and axons_group_sum + test_axons[i] <= a):
                break

            axons_group_sum += test_axons[i]
            l, n, a = _get_limit(i)
            i += 1

        axons_grouped.append(
            # Slice the array [i_last: i+1], which means [i_last, i].
            AxonGroup(l, test_axons[i_start:i], indices[i_start:i])
        )
        i_start = i

    print(time.time() - t1)
    print()
    # print(axons_grouped)
    for expected_group, group in zip(expected_axons_grouped, axons_grouped):
        assert group.lcnx == expected_group.lcnx
        assert np.array_equal(group.test_axons, expected_group.test_axons)
        assert np.array_equal(group.indices, expected_group.indices)
