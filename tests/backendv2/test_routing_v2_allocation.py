from unittest.mock import MagicMock, PropertyMock, call, patch

import numpy as np
import pytest
from paicorelib import OfflineCoreRegV2, WeightCompressType

from paibox.backendv2.coreplacement import OfflineCorePlamentV2
from paibox.backendv2.neuron import NeuronType, OfflineNeuronPlacement
from paibox.backendv2.routing import RoutingGroup

# 模拟导入 (防止环境缺失)
from paibox.backendv2.weight import Weight


class TestRoutingGroupAllocation:

    @pytest.fixture
    def mock_deps(self):
        """Mock external dependencies."""
        with (
            patch("paibox.backendv2.routing.get_raw_weights") as mock_get_weights,
            patch("paibox.backendv2.routing.Weight") as mock_weight_cls,
            patch(
                "paibox.backendv2.routing.OfflineNeuronPlacement"
            ) as mock_neu_placement_cls,
            patch(
                "paibox.backendv2.routing.OfflineCorePlamentV2"
            ) as mock_core_placement_cls,
            patch("paibox.backendv2.routing.OfflineCoreRegV2") as mock_reg_v2_cls,
        ):

            # [修复 1] 正确模拟 Pydantic 的 model_fields
            # 让 model_fields 表现为一个字典，这样 .keys() 就能正常工作
            mock_fields = {
                "weight_width": MagicMock(),
                "snn_ann": MagicMock(),
                "neuron_number": MagicMock(),
            }
            mock_reg_v2_cls.model_fields = mock_fields

            yield mock_weight_cls, mock_neu_placement_cls, mock_core_placement_cls, mock_reg_v2_cls, mock_get_weights

    def create_mock_neuron(self, weight_width=1, core_config_id=1, config_obj=None):
        """Helper to create a mock neuron."""
        neu = MagicMock()
        attrs = MagicMock()
        attrs.weight_compress = None
        # 默认让 attrs 相等，方便测试 Half 逻辑
        attrs.__eq__.return_value = True
        neu.attrs_part2.return_value = attrs

        if config_obj:
            config = config_obj
        else:
            config = MagicMock()
            config.weight_width = weight_width
            config.snn_ann = 1
            config._id = core_config_id
            # 默认 Config 比较基于 ID (模拟不同对象)
            config.__eq__.side_effect = lambda other: config._id == getattr(
                other, "_id", -1
            )

        neu.core_config.return_value = config

        return neu, attrs, config

    def _setup_core_mock_list_behavior(self, core_mock):
        """Helper: 让 Core.neus 表现得像一个列表 (支持 append 和 len)"""
        real_list = []
        mock_list = MagicMock()
        mock_list.__len__.side_effect = lambda: len(real_list)
        mock_list.append.side_effect = lambda x: real_list.append(x)
        mock_list.__iter__.side_effect = lambda: iter(real_list)
        # 确保布尔值为 True (只要列表不为空) 或者根据 len 动态判断
        # MagicMock 默认 bool 可能是 False，这里显式绑定 len
        core_mock.neus = mock_list
        return real_list

    def test_allocate_single_neuron_success(self, mock_deps):
        """Test basic allocation."""
        MockWeight, MockNeuPlacement, MockCorePlacement, MockRegV2, MockGetWeights = (
            mock_deps
        )

        MockGetWeights.return_value = np.zeros((1, 10), dtype=np.int8)

        rg = RoutingGroup()
        rg.lcn = 1
        neu1, attrs1, config1 = self.create_mock_neuron(weight_width=8)

        rg.raw_neus = [neu1]
        rg.input_list = [MagicMock()]
        rg.dests = {neu1: MagicMock(lcn=1)}

        # Setup Core
        core_instance = MockCorePlacement.return_value
        self._setup_core_mock_list_behavior(core_instance)  # [修复 3]
        core_instance.weights = []
        core_instance.neu_weight_map = {}
        core_instance.n_sram_required.return_value = 0

        # Setup Weight
        w_dense = MagicMock()
        w_sparse = MagicMock()
        w_dense.n_sram_required.return_value = 100
        w_sparse.n_sram_required.return_value = 200
        MockWeight.side_effect = [w_dense, w_sparse]

        # Setup NeuronPlacement
        neu_place_instance = MockNeuPlacement.return_value
        neu_place_instance.n_sram_required.return_value = 50

        rg.allocate_neurons()

        assert len(rg.core_placements) == 1
        assert MockWeight.call_args_list[0][0][2] == 8
        assert attrs1.weight_compress == WeightCompressType.DENSE

        # 验证是否正确提取了 weight_width
        assert MockRegV2.call_args.kwargs.get("weight_width") == 8

    def test_grouping_logic(self, mock_deps):
        """Test grouping."""
        MockWeight, MockNeuPlacement, MockCorePlacement, MockRegV2, MockGetWeights = (
            mock_deps
        )

        MockGetWeights.return_value = np.zeros((2, 10), dtype=np.int8)
        rg = RoutingGroup()
        rg.lcn = 1

        # 使用不同的 config ID，确保分为两组
        neu1, _, cfg1 = self.create_mock_neuron(core_config_id=1)
        neu2, _, cfg2 = self.create_mock_neuron(core_config_id=2)

        rg.raw_neus = [neu1, neu2]
        rg.input_list = [MagicMock()] * 10
        rg.dests = {neu1: MagicMock(lcn=1), neu2: MagicMock(lcn=1)}

        # Setup Core & List Behavior
        core_mock = MockCorePlacement.return_value
        core_mock.n_sram_required.return_value = 0
        self._setup_core_mock_list_behavior(
            core_mock
        )  # [修复 3] 必须模拟列表，否则 len=0 导致不保存

        # Weight / Placement return values
        MockWeight.return_value.n_sram_required.return_value = 10
        MockNeuPlacement.return_value.n_sram_required.return_value = 10

        rg.allocate_neurons()

        assert len(rg.core_placements) == 2
        assert MockRegV2.call_count == 2

    def test_overflow_creates_new_core_with_inheritance(self, mock_deps):
        """Test overflow."""
        MockWeight, MockNeuPlacement, MockCorePlacement, MockRegV2, MockGetWeights = (
            mock_deps
        )

        MockGetWeights.return_value = np.zeros((2, 10), dtype=np.int8)
        rg = RoutingGroup()
        rg.lcn = 1

        # 同一组 (config ID 相同)
        neu1, _, cfg1 = self.create_mock_neuron(weight_width=4, core_config_id=1)
        # 显式传入 cfg1 确保对象相等性万无一失
        neu2, _, _ = self.create_mock_neuron(weight_width=4, config_obj=cfg1)

        rg.raw_neus = [neu1, neu2]
        rg.input_list = [MagicMock()]
        rg.dests = {neu1: MagicMock(lcn=1), neu2: MagicMock(lcn=1)}

        # Setup Cores
        core_1 = MagicMock()
        self._setup_core_mock_list_behavior(core_1)
        # 第一次检查返回0（放入neu1），第二次检查返回2500（放入neu2前检测），加上neu2导致溢出
        core_1.n_sram_required.side_effect = [0, 2500]

        core_2 = MagicMock()
        self._setup_core_mock_list_behavior(core_2)
        core_2.n_sram_required.return_value = 0

        MockCorePlacement.side_effect = [core_1, core_2]

        # Weight / Placement
        MockWeight.return_value.n_sram_required.return_value = 2000

        np_mock = MagicMock()
        np_mock.n_sram_required.return_value = 50
        MockNeuPlacement.return_value = np_mock

        rg.allocate_neurons()

        assert len(rg.core_placements) == 2
        assert rg.core_placements[0] == core_1
        assert rg.core_placements[1] == core_2
        # Neu2 溢出到新 Core，必须强制为 FULL
        assert np_mock.neuron_type == NeuronType.FULL

    def test_half_neuron_optimization(self, mock_deps):
        """Test Half Neuron assignment."""
        MockWeight, MockNeuPlacement, MockCorePlacement, MockRegV2, MockGetWeights = (
            mock_deps
        )

        MockGetWeights.return_value = np.zeros((2, 10), dtype=np.int8)
        rg = RoutingGroup()
        rg.lcn = 1

        # [修复 2] 关键：必须使用同一个 Config 对象，否则它们会被分到不同组，Half 优化不生效
        neu1, attrs1, cfg1 = self.create_mock_neuron()
        neu2, attrs2, _ = self.create_mock_neuron(config_obj=cfg1)

        # 确保 attrs 相等
        attrs1.__eq__.return_value = True
        attrs2.__eq__.return_value = True

        rg.raw_neus = [neu1, neu2]
        rg.dests = {neu1: MagicMock(lcn=1), neu2: MagicMock(lcn=1)}
        rg.input_list = [MagicMock()]

        # Setup Core
        core_mock = MockCorePlacement.return_value
        core_mock.n_sram_required.return_value = 0
        self._setup_core_mock_list_behavior(core_mock)  # [修复 3]

        MockWeight.return_value.n_sram_required.return_value = 10

        # 两个不同的 Placement 实例，用于验证类型
        p1 = MagicMock()
        p1.n_sram_required.return_value = 10
        p2 = MagicMock()
        p2.n_sram_required.return_value = 10
        MockNeuPlacement.side_effect = [p1, p2]

        rg.allocate_neurons()

        assert p1.neuron_type == NeuronType.FULL
        assert p2.neuron_type == NeuronType.HALF

    def test_weight_compression_selection(self, mock_deps):
        """Test optimal weight compression."""
        MockWeight, MockNeuPlacement, MockCorePlacement, MockRegV2, MockGetWeights = (
            mock_deps
        )

        MockGetWeights.return_value = np.zeros((1, 10), dtype=np.int8)
        rg = RoutingGroup()
        rg.lcn = 1
        neu1, attrs1, _ = self.create_mock_neuron(weight_width=2)

        rg.raw_neus = [neu1]
        rg.dests = {neu1: MagicMock(lcn=1)}
        rg.input_list = [MagicMock()]

        core_mock = MockCorePlacement.return_value
        core_mock.n_sram_required.return_value = 0
        self._setup_core_mock_list_behavior(core_mock)

        MockNeuPlacement.return_value.n_sram_required.return_value = 10

        w_dense = MagicMock()
        w_dense.n_sram_required.return_value = 100
        w_sparse = MagicMock()
        w_sparse.n_sram_required.return_value = 50
        MockWeight.side_effect = [w_dense, w_sparse]

        rg.allocate_neurons()

        assert attrs1.weight_compress == WeightCompressType.SPARSE
        assert MockWeight.call_args_list[0][0][2] == 2
