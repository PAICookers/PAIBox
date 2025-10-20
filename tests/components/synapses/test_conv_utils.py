import timeit

import numpy as np
import pytest

from paibox.components.synapses.conv_utils import (
    _conv1d_oshape,
    _conv1d_unroll,
    _conv2d_oshape,
    _conv2d_unroll,
    conv1d_faster,
    conv2d_faster,
)
from tests.components.utils import conv1d_golden, conv2d_golden
from tests.utils import ParamTestCase, gen_random_array, is_ci_env, make_test

try:
    from paibox.components.synapses.conv_utils import conv1d_faster_legacy

    skip_conv1d_faster_test = False
except ImportError:
    skip_conv1d_faster_test = True


test_conv1d_faster_data = ParamTestCase(
    argnames="in_shape, co, ksize, stride, padding, dilation, groups",
    argvalues=[
        ((16, 2048), 32, (128,), (20,), (50,), (8,), 1),
        ((64, 512), 32, (16,), (8,), (0,), (1,), 4),
    ],
)


@pytest.mark.skipif(
    skip_conv1d_faster_test,
    reason="Legacy function 'conv1d_faster_legacy' is removed",
)
@make_test(test_conv1d_faster_data)
def test_conv1d_perf(in_shape, co, ksize, stride, padding, dilation, groups, fixed_rng):
    ci = in_shape[0]
    oshape = _conv1d_oshape(in_shape[1:], ksize, stride, padding, dilation)
    assert ci % groups == 0 and co % groups == 0
    ci_in_grp = ci // groups

    x = gen_random_array(in_shape, np.uint8, fixed_rng)
    kernel = gen_random_array((co, ci_in_grp) + ksize, np.int8, fixed_rng)

    def run_conv1d_faster():
        return conv1d_faster(x, oshape, kernel, stride, padding, dilation, groups)

    def run_conv1d_faster_legacy():
        return conv1d_faster_legacy(
            x, oshape, kernel, stride, padding, dilation, groups
        )

    def run_conv1d_golden():
        return conv1d_golden(x, oshape, kernel, stride, padding, dilation, groups)

    if not is_ci_env():
        # ~10x faster
        n = 10
        t1 = timeit.timeit(lambda: run_conv1d_faster(), number=n)
        t2 = timeit.timeit(lambda: run_conv1d_faster_legacy(), number=n)
        t3 = timeit.timeit(lambda: run_conv1d_golden(), number=n)
        print("Optimized: ", t1 / n, "Legacy: ", t2 / n, "Golden: ", t3 / n)

    r_opt = run_conv1d_faster()
    r_legacy = run_conv1d_faster_legacy()
    assert np.array_equal(r_opt, r_legacy)


try:
    from paibox.components.synapses.conv_utils import _conv1d_unroll_legacy

    skip_conv1d_unroll_test = False
except ImportError:
    skip_conv1d_unroll_test = True


test_conv1d_unroll_data = ParamTestCase(
    argnames="in_shape, ci, co, ksize, stride, padding, groups",
    argvalues=[
        ((32,), 8, 16, (3,), (1,), (2,), 1),
        ((240,), 16, 32, (40,), (5,), (0,), 1),
        ((480,), 16, 32, (32,), (16,), (8,), 4),
        ((1000,), 32, 64, (100,), (20,), (20,), 2),
    ],
)


@pytest.mark.skipif(
    skip_conv1d_unroll_test,
    reason="Legacy function '_conv1d_unroll_legacy' is removed",
)
@make_test(test_conv1d_unroll_data)
def test_conv1d_unroll_perf(
    in_shape, ci, co, ksize, stride, padding, groups, fixed_rng
):
    oshape = _conv1d_oshape(in_shape, ksize, stride, padding)
    assert ci % groups == 0 and co % groups == 0
    ci_in_grp = ci // groups

    k_shape = (co, ci_in_grp) + ksize
    kernel = gen_random_array(k_shape, np.int8, fixed_rng)

    def run_conv1d_unroll():
        return _conv1d_unroll(in_shape, oshape, kernel, stride, padding, groups)

    def run_conv1d_unroll_legacy():
        return _conv1d_unroll_legacy(in_shape, oshape, kernel, stride, padding, groups)

    if not is_ci_env():
        # 10~40x faster
        n = 5
        t1 = timeit.timeit(lambda: run_conv1d_unroll(), number=n)
        t2 = timeit.timeit(lambda: run_conv1d_unroll_legacy(), number=n)
        print("Optimized: ", t1 / n, "Legacy: ", t2 / n)

    if groups > 1:
        pytest.skip("'_conv1d_unroll_legacy()' has bugs handling groups > 1")

    r_opt = run_conv1d_unroll()
    r_legacy = run_conv1d_unroll_legacy()
    assert np.array_equal(r_opt, r_legacy)


@make_test(test_conv1d_unroll_data)
def test_conv1d_unroll(in_shape, ci, co, ksize, stride, padding, groups, fixed_rng):
    oshape = _conv1d_oshape(in_shape, ksize, stride, padding)
    assert ci % groups == 0 and co % groups == 0
    ci_in_grp = ci // groups

    k_shape = (co, ci_in_grp) + ksize
    kernel = gen_random_array(k_shape, np.int8, fixed_rng)

    x = gen_random_array((ci,) + in_shape, np.uint8, fixed_rng)

    y_golden = conv1d_golden(x, oshape, kernel, stride, padding, groups=groups)

    k_unrolled = _conv1d_unroll(in_shape, oshape, kernel, stride, padding, groups)
    y_unrolled = x.ravel().astype(np.int32) @ k_unrolled

    assert np.array_equal(y_golden.ravel(), y_unrolled)


try:
    from paibox.components.synapses.conv_utils import conv2d_faster_legacy

    skip_conv2d_faster_test = False
except ImportError:
    skip_conv2d_faster_test = True


test_conv2d_faster_data = ParamTestCase(
    argnames="in_shape, co, ksize, stride, padding, dilation, groups",
    argvalues=[
        ((16, 64, 64), 32, (3, 3), (1, 1), (1, 1), (2, 2), 4),
        ((24, 32, 32), 24, (4, 4), (2, 2), (2, 1), (1, 1), 2),
        ((16, 24, 24), 8, (3, 3), (1, 2), (0, 0), (1, 1), 1),
        ((16, 64, 64), 32, (3, 3), (2, 2), (1, 1), (0, 0), 1),
    ],
)


@pytest.mark.skipif(
    skip_conv2d_faster_test, reason="Legacy function 'conv2d_faster_legacy' is removed"
)
@make_test(test_conv2d_faster_data)
def test_conv2d_perf(in_shape, co, ksize, stride, padding, dilation, groups, fixed_rng):
    ci = in_shape[0]
    oshape = _conv2d_oshape(in_shape[1:], ksize, stride, padding, dilation)
    assert ci % groups == 0 and co % groups == 0
    ci_in_grp = ci // groups

    x = gen_random_array(in_shape, np.uint8, fixed_rng)
    kernel = gen_random_array((co, ci_in_grp) + ksize, np.int8, fixed_rng)

    def run_conv2d_faster():
        return conv2d_faster(x, oshape, kernel, stride, padding, dilation, groups)

    def run_conv2d_faster_legacy():
        return conv2d_faster_legacy(
            x, oshape, kernel, stride, padding, dilation, groups
        )

    def run_conv2d_golden():
        return conv2d_golden(x, oshape, kernel, stride, padding, dilation, groups)

    if not is_ci_env():
        # ~3x faster
        n = 10
        t1 = timeit.timeit(lambda: run_conv2d_faster(), number=n)
        t2 = timeit.timeit(lambda: run_conv2d_faster_legacy(), number=n)
        t3 = timeit.timeit(lambda: run_conv2d_golden(), number=n)
        print("Optimized: ", t1 / n, "Legacy: ", t2 / n, "Golden: ", t3 / n)

    r_opt = run_conv2d_faster()
    r_legacy = run_conv2d_faster_legacy()
    assert np.array_equal(r_opt, r_legacy)


try:
    from paibox.components.synapses.conv_utils import _conv2d_unroll_legacy

    skip_conv2d_unroll_test = False
except ImportError:
    skip_conv2d_unroll_test = True


test_conv2d_unroll_data = ParamTestCase(
    argnames="in_shape, ci, co, ksize, stride, padding, groups",
    argvalues=[
        ((16, 16), 4, 16, (3, 3), (1, 1), (0, 0), 1),
        ((32, 32), 16, 32, (3, 3), (1, 1), (1, 1), 1),
        ((32, 32), 32, 8, (4, 4), (2, 2), (0, 0), 2),
        ((32, 24), 16, 24, (4, 4), (2, 2), (2, 2), 8),
    ],
)


@pytest.mark.skipif(
    skip_conv2d_unroll_test,
    reason="Legacy function '_conv2d_unroll_legacy' is removed",
)
@make_test(test_conv2d_unroll_data)
def test_conv2d_unroll_perf(
    in_shape, ci, co, ksize, stride, padding, groups, fixed_rng
):
    oshape = _conv2d_oshape(in_shape, ksize, stride, padding)
    assert ci % groups == 0 and co % groups == 0
    ci_in_grp = ci // groups

    k_shape = (co, ci_in_grp) + ksize
    kernel = gen_random_array(k_shape, np.int8, fixed_rng)

    def run_conv2d_unroll():
        return _conv2d_unroll(in_shape, oshape, kernel, stride, padding, groups)

    def run_conv2d_unroll_legacy():
        return _conv2d_unroll_legacy(in_shape, oshape, kernel, stride, padding, groups)

    if not is_ci_env():
        # 50~100x faster
        n = 5
        t1 = timeit.timeit(lambda: run_conv2d_unroll(), number=n)
        t2 = timeit.timeit(lambda: run_conv2d_unroll_legacy(), number=n)
        print("Optimized: ", t1 / n, "Legacy: ", t2 / n)

    if groups > 1:
        pytest.skip("'_conv2d_unroll_legacy()' has bugs handling groups > 1")

    r_opt = run_conv2d_unroll()
    r_legacy = run_conv2d_unroll_legacy()
    assert np.array_equal(r_opt, r_legacy)


@make_test(test_conv2d_unroll_data)
def test_conv2d_unroll(in_shape, ci, co, ksize, stride, padding, groups, fixed_rng):
    oshape = _conv2d_oshape(in_shape, ksize, stride, padding)
    assert ci % groups == 0 and co % groups == 0
    ci_in_grp = ci // groups

    k_shape = (co, ci_in_grp) + ksize
    kernel = gen_random_array(k_shape, np.int8, fixed_rng)

    x = gen_random_array((ci,) + in_shape, np.uint8, fixed_rng)

    y_expected = conv2d_faster(x, oshape, kernel, stride, padding, groups=groups)

    k_unrolled = _conv2d_unroll(in_shape, oshape, kernel, stride, padding, groups)
    y_unrolled = (x.ravel().astype(np.int32) @ k_unrolled).astype(np.int32)

    assert np.array_equal(y_expected.ravel(), y_unrolled)
