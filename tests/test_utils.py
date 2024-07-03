import pytest

from paibox.utils import bit_reversal, fn_sgn, typical_round


@pytest.mark.parametrize("a,b, expected", [(1, 0, 1), (1, 2, -1), (3, 3, 0)])
def test_fn_sgn(a, b, expected):
    assert fn_sgn(a, b) == expected


@pytest.mark.parametrize(
    "n, expected", [(10.2, 10), (0.5, 1), (1.4, 1), (2.5, 3), (0.4, 0)]
)
def test_typical_round(n, expected):
    assert typical_round(n) == expected


@pytest.mark.parametrize(
    "uint, n_bit, expected",
    [
        (0b10110, 5, 0b01101),
        (0b0111_0111_1001_1001, 10, 0b1001_1001_11),
        (0b1010_1100_1101, 7, 0b1011_001),
    ],
)
def test_bit_reversal(uint, n_bit, expected):
    assert bit_reversal(uint, n_bit) == expected
