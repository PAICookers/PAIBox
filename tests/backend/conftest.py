import pytest
from .backend_testcase import *


@pytest.fixture(scope="class")
def build_example_net1():
    return NetForTest1()


@pytest.fixture(scope="class")
def build_example_net2():
    return NetForTest2()


@pytest.fixture(scope="class")
def build_multi_inputproj_net1():
    return NetForTest2()


@pytest.fixture(scope="class")
def build_multi_inputproj_net2():
    return Network_with_multi_inodes1()


@pytest.fixture(scope="class")
def build_multi_inputproj_net3():
    return Network_with_multi_inodes2()


@pytest.fixture(scope="class")
def build_example_net3():
    return NetForTest3()


@pytest.fixture(scope="class")
def build_example_net4():
    return NetForTest4()


@pytest.fixture(scope="class")
def build_example_net5():
    return NetForTest5()


@pytest.fixture(scope="class")
def build_example_net6():
    return NetForTest6()


@pytest.fixture(scope="class")
def build_example_net4_large_scale():
    return NetForTest4(large_scale=True)


@pytest.fixture(scope="class")
def build_multi_onodes_net():
    return Network_with_multi_onodes()


@pytest.fixture(scope="class")
def build_multi_onodes_net2():
    return Network_with_multi_onodes(connect_n4=True)


@pytest.fixture(scope="class")
def build_multi_onodes_net_more1152():
    return Network_with_multi_onodes(connect_n4=True, onode_more1152=True)


@pytest.fixture(scope="class")
def build_multi_inodes_onodes():
    return Network_with_multi_inodes_onodes()


@pytest.fixture(scope="class", params=[30, 32, 60, 100])
def build_Network_with_N_onodes(request):
    return Network_with_N_onodes(n_onodes=request.param)


@pytest.fixture(scope="class")
def build_network_with_branches_4bit():
    return Network_with_Branches_4bit(seed=42)


@pytest.fixture(scope="class")
def build_Network_8bit_dense():
    return Network_with_Branches_8bit(seed=42)


@pytest.fixture(scope="class")
def build_Network_with_container():
    return Network_with_container()


@pytest.fixture(scope="class")
def build_Nested_Net_level_2():
    return Nested_Net_level_2()


@pytest.fixture(scope="class")
def build_Nested_Net_level_3():
    return Nested_Net_level_3()


@pytest.fixture(scope="class")
def build_MultichipNet1_s1():
    return MultichipNet1(scale=1)


@pytest.fixture(scope="class")
def build_MultichipNet1_s2():
    return MultichipNet1(scale=2)


@pytest.fixture(
    scope="function",
    params=[Network_branch_nodes1, Network_branch_nodes2, Network_branch_nodes3],
    ids=["net1", "net2", "net3"],
)
def build_Network_branch_nodes(request):
    return request.param()
