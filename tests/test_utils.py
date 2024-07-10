import pytest

from paibox.utils import reverse_8bit, reverse_16bit, fn_sgn, typical_round


@pytest.mark.parametrize("a,b, expected", [(1, 0, 1), (1, 2, -1), (3, 3, 0)])
def test_fn_sgn(a, b, expected):
    assert fn_sgn(a, b) == expected


@pytest.mark.parametrize(
    "n, expected", [(10.2, 10), (0.5, 1), (1.4, 1), (2.5, 3), (0.4, 0)]
)
def test_typical_round(n, expected):
    assert typical_round(n) == expected


@pytest.mark.parametrize(
    "x, expected",
    [
        (0b1001_0110, 0b0110_1001),
        (0b0001_1001, 0b1001_1000),
        (0b1100_1101, 0b1011_0011),
    ],
)
def test_reverse_8bit(x, expected):
    assert reverse_8bit(x) == expected


@pytest.mark.parametrize(
    "x, expected",
    [
        (0b0110_0001_1001_0111, 0b1110_1001_1000_0110),
        (0b1110_0011_0001_1001, 0b1001_1000_1100_0111),
        (0b1100_1101_1001_1101, 0b1011_1001_1011_0011),
    ],
)
def test_reverse_16bit(x, expected):
    assert reverse_16bit(x) == expected
