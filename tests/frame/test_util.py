import numpy as np

from paibox.frame.util import bin_array_split, bin_split


def test_bin_split():
    x = 0b101
    assert bin_split(x, 2, 1) == (0b10, 0b1)
    h, l = bin_split(x, 2, 1)
    print(h, l)


def test_bin_array_split():
    x = [0b101, 0b110]
    a, b = bin_array_split(x, 2, 1)
    print(a, b)
