import pytest

from paibox.utils import fn_sgn, reverse_8bit, reverse_16bit, typical_round


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


@pytest.mark.parametrize(
    "s1, idx, expected",
    [
        (slice(5, 10), 2, slice(7, 8)),
        (slice(100, 200), 20, slice(120, 121)),
        (slice(100, 200), -1, slice(199, 200)),
        (slice(100, 200), -10, slice(190, 191)),
    ],
)
def test_slice_by_index(s1, idx, expected):
    n_s1 = s1.stop - s1.start
    if idx < 0:
        _idx = n_s1 + idx
        if _idx < 0:
            raise ValueError(f"index out of range: {idx} < 0")
    else:
        _idx = idx
        if _idx > n_s1 - 1:
            raise ValueError(f"index out of range: {idx} > {n_s1-1}")

    start = s1.start + _idx
    end = start + 1
    new_slice = slice(start, end, s1.step)

    assert new_slice == expected


@pytest.mark.parametrize(
    "s1, s2, expected",
    [
        (slice(5, 10), slice(0, 3), slice(5, 8)),
        (slice(100, 200), slice(50, 100), slice(150, 200)),
        (slice(100, 200), slice(None, 20), slice(100, 120)),
        (slice(100, 200), slice(10, None), slice(110, 200)),
        (slice(100, 300), slice(None, -40), slice(100, 260)),
    ],
)
def test_slice_by_slice(s1, s2, expected):
    n_s1 = s1.stop - s1.start
    _s2_start = s2.start if s2.start is not None else 0
    if s2.stop is None:
        _s2_stop = n_s1
    elif s2.stop < 0:
        _s2_stop = n_s1 + s2.stop
    else:
        _s2_stop = s2.stop

    if (_n_s2 := _s2_stop - _s2_start) > n_s1:
        raise ValueError(f"index out of range: {_n_s2} > {n_s1}")

    start = s1.start + _s2_start
    end = s1.start + _s2_stop
    new_slice = slice(start, end, s1.step)

    assert new_slice == expected
