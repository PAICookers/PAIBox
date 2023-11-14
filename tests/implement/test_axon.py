import pytest

from paibox.implement.axon_segment import AxonSegment, get_axon_segments
from paibox.neuron import IF

expected_value = [
    [
        [
            AxonSegment(
                index=slice(0, 556, 1), tick_relative=0, addr_axon=slice(0, 556, 1)
            ),
            AxonSegment(
                index=slice(556, 1112, 1), tick_relative=1, addr_axon=slice(0, 556, 1)
            ),
            AxonSegment(
                index=slice(1112, 1668, 1), tick_relative=2, addr_axon=slice(0, 556, 1)
            ),
            AxonSegment(
                index=slice(1668, 2222, 1), tick_relative=3, addr_axon=slice(0, 554, 1)
            ),
        ],
        [
            AxonSegment(
                index=slice(0, 595, 1), tick_relative=0, addr_axon=slice(556, 1151, 1)
            ),
            AxonSegment(
                index=slice(595, 1190, 1),
                tick_relative=1,
                addr_axon=slice(556, 1151, 1),
            ),
            AxonSegment(
                index=slice(1190, 1785, 1),
                tick_relative=2,
                addr_axon=slice(556, 1151, 1),
            ),
            AxonSegment(
                index=slice(1785, 2378, 1),
                tick_relative=3,
                addr_axon=slice(556, 1149, 1),
            ),
        ],
    ]
]


@pytest.mark.parametrize(
    "node_sequence, lcn_ex, expect_error, expect_index",
    [
        (
            [IF(1151, threshold=127, reset_v=0), IF(1153, threshold=127, reset_v=0)],
            1,
            True,
            -1,
        ),  # Test case with ValueError
        (
            [IF(2222, threshold=127, reset_v=0), IF(2378, threshold=127, reset_v=0)],
            2,
            False,
            0,
        ),  # Example without ValueError
    ],
)
def test_get_axon_segments(node_sequence, lcn_ex, expect_error, expect_index):
    if expect_error:
        with pytest.raises(ValueError):
            get_axon_segments(node_sequence, lcn_ex)

    else:
        axon_segments_list = get_axon_segments(node_sequence, lcn_ex)
        assert axon_segments_list == expected_value[expect_index]
