import os
import tempfile

import numpy as np
import pytest

import paibox as pb
from paibox.base import SynSys
from paibox.naming import clear_name_cache

# Import the logging hooks from logging_utils
from ._logging.logging_utils import captured_logs, log_settings_patch  # noqa: F401
from .shared_networks import *
from .utils import is_ci_env, make_dump_dir, measure_time


# Add custom markers to eliminate pytest warning
def pytest_configure(config: pytest.Config):
    config.addinivalue_line(
        "markers",
        "make_settings_test(**settings_dict): mark test to set custom settings for logging & perform teardown.",
    )
    config.addinivalue_line(
        "markers",
        "perf: mark test to measure performance. Skip if running in ci environment.",
    )


def pytest_runtest_setup(item: pytest.Item):
    if "perf" in item.keywords and is_ci_env():
        pytest.skip("Skipping perf test in CI environment")


@pytest.fixture(scope="module")
def ensure_dump_dir(request, tmp_path_factory):
    p = make_dump_dir(request.path.parent, tmp_path_factory)
    yield p


@pytest.fixture(scope="module")
def ensure_dump_dir_and_clean(request, tmp_path_factory):
    p = make_dump_dir(request.path.parent, tmp_path_factory)
    yield p
    for f in p.iterdir():
        f.unlink(missing_ok=True)


@pytest.fixture
def cleandir():
    with tempfile.TemporaryDirectory() as newpath:
        old_cwd = os.getcwd()
        os.chdir(newpath)
        yield
        os.chdir(old_cwd)


def _reset_context() -> None:
    clear_name_cache(ignore_warn=True)
    pb.FRONTEND_ENV["t"] = 0
    pb.BACKEND_CONFIG.set_default()
    # To avoid overlapping with multi-chip coordinates
    pb.BACKEND_CONFIG.output_chip_addr = (9, 9)
    SynSys.CFLAG_ENABLE_WP_OPTIMIZATION = True


@pytest.fixture(autouse=True)
def context_reset():
    """Reset the context after each test automatically."""
    _reset_context()
    yield
    _reset_context()


@pytest.fixture
def perf_fixture(request):
    with measure_time(f"{request.node.name}"):
        yield


@pytest.fixture(scope="session")
def fixed_rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture(scope="class")
def build_Input_to_N1():
    return Input_to_N1()


@pytest.fixture(scope="class")
def build_NotNested_Net():
    return Input_to_N1()


@pytest.fixture(scope="class")
def build_NotNested_Net_Exp():
    return NotNested_Net_Exp()


@pytest.fixture(scope="class")
def build_Network_with_container():
    return Network_with_container()


@pytest.fixture(scope="class")
def build_multi_inodes_onodes():
    return Network_with_multi_inodes_onodes()


@pytest.fixture(scope="class")
def build_Nested_Net_L1():
    return Nested_Net_L1()


@pytest.fixture(scope="class")
def build_Nested_Net_L2():
    return Nested_Net_L2()


@pytest.fixture(scope="class")
def build_Nested_Net_L3():
    return Nested_Net_L3()


@pytest.fixture(scope="class")
def build_BitwiseAND_Net():
    return FunctionalModule_2to1_Net("and")


@pytest.fixture(scope="class")
def build_BitwiseNOT_Net():
    return FunctionalModule_1to1_Net("not")


@pytest.fixture(scope="class")
def build_BitwiseOR_Net():
    return FunctionalModule_2to1_Net("or")


@pytest.fixture(scope="class")
def build_BitwiseXOR_Net():
    return FunctionalModule_2to1_Net("xor")


@pytest.fixture(scope="class")
def build_DelayChain_Net():
    return FunctionalModule_1to1_Net("delay")


@pytest.fixture(scope="class")
def build_SpikingAdd_Net():
    return FunctionalModule_2to1_Net("add")


@pytest.fixture(scope="class")
def build_SpikingSub_Net():
    return FunctionalModule_2to1_Net("sub")


@pytest.fixture(scope="class")
def build_FModule_ConnWithInput_Net():
    return FModule_ConnWithInput_Net()


@pytest.fixture(scope="class")
def build_FModule_ConnWithModule_Net():
    return FModule_ConnWithModule_Net()


@pytest.fixture(scope="class")
def build_FModule_ConnWithFModule_Net():
    return FModule_ConnWithFModule_Net()


@pytest.fixture(scope="class")
def build_ANN_Network_1():
    return ANNNetwork()
