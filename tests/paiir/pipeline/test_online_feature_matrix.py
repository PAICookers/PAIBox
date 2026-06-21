from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from paicorelib import LCN_EX
from torch import Tensor, nn

from paibox.backendv2.proto.runtime import (
    build_output_mapping_tables,
    encode_online_boundary_output_frames,
    encode_online_input_frames,
    load_compile_artifacts,
)
from paibox.paiir import compile_to_paiir, mark_online
from paibox.paiir.ir.calc_params import (
    OnlineCoreType,
)
from paibox.paiir.ir.op_node import SequentialOp, StandaloneCompOp
from paibox.paiir.pipeline.online import analyze_online_update_stage_plans
from tests.paiir.conftest import (
    SJMNISTValidationNet,
    find_nodes,
    find_transform_nodes,
    make_img_1ch_28x28,
    make_vec_8d,
    offline_nodes,
)
from tests.paiir.online_test_utils import (
    MNISTFlattenOnlineLinear,
    OnlineLinear,
    TwoLayerOnlineLinear,
    WideOnlineLinear,
    export_online_graph,
    first_online_forward_node,
    last_online_forward_node,
    online_nodes,
    uniform_online_lcn_kwargs,
)

DEBUG_EXPORT_ROOT = Path(__file__).with_name("debug") / "online_feature_export"


def _print_case_header(feature: str, case_name: str) -> None:
    print()
    print(f"功能项：{feature}")
    print(f"样例：{case_name}")


def _online_core_type_name(value: object) -> str:
    if isinstance(value, OnlineCoreType):
        return value.name
    return OnlineCoreType(int(value)).name


def _print_update_plan_bindings(plans) -> None:
    print("更新归属关系：")
    for plan in plans:
        role = plan.gradient_role.value if plan.gradient_role is not None else "unknown"
        layer_idx = plan.layer_idx_from_output + 1
        print(
            f"  第{layer_idx}层（从输出侧计，{role}）："
            f"forward={plan.forward_name}，"
            f"gradient={plan.backward_peer_name}，"
            f"update={plan.update_name}"
        )


class TestOnlineStageExpansionReport:
    @pytest.mark.parametrize(
        (
            "case_name",
            "model_factory",
            "sample_factory",
            "expected_sequence",
            "expected_plan_count",
            "expected_roles",
        ),
        (
            pytest.param(
                "single_layer_fc",
                OnlineLinear,
                make_vec_8d,
                ("forward", "loss", "gradient", "update"),
                1,
                ("output",),
                id="single_layer_fc",
            ),
            pytest.param(
                "two_layer_fc",
                TwoLayerOnlineLinear,
                make_vec_8d,
                (
                    "forward",
                    "forward",
                    "loss",
                    "gradient",
                    "gradient",
                    "update",
                    "update",
                ),
                2,
                ("output", "hidden"),
                id="two_layer_fc",
            ),
        ),
    )
    def test_stage_and_update_bindings(
        self,
        case_name: str,
        model_factory: Callable[[], nn.Module],
        sample_factory: Callable[[], Tensor],
        expected_sequence: tuple[str, ...],
        expected_plan_count: int,
        expected_roles: tuple[str, ...],
    ):
        graph = compile_to_paiir(mark_online(model_factory()), sample_factory())
        plans = analyze_online_update_stage_plans(graph)
        stage_sequence = tuple(
            node.core_params.semantic_mode.value for node in online_nodes(graph)
        )
        roles = tuple(plan.gradient_role.value for plan in plans)

        _print_case_header("训练阶段展开与更新绑定", case_name)
        print("阶段顺序：", " -> ".join(stage_sequence))
        print("更新计划数量：", len(plans))
        print("梯度角色顺序：", " -> ".join(roles))
        _print_update_plan_bindings(plans)

        assert stage_sequence == expected_sequence
        assert len(plans) == expected_plan_count
        assert roles == expected_roles


class TestOnlineBoundarySemanticsReport:
    @pytest.mark.parametrize(
        (
            "case_name",
            "model_factory",
            "sample_factory",
            "mark_overrides",
            "expected_input_core",
            "expected_output_core",
            "expected_transform_count",
        ),
        (
            pytest.param(
                "vector_default_entry",
                OnlineLinear,
                make_vec_8d,
                {},
                OnlineCoreType.ONLINE,
                OnlineCoreType.ONLINE,
                0,
                id="vector_default_entry",
            ),
            pytest.param(
                "vector_offline_entry",
                OnlineLinear,
                make_vec_8d,
                {"input_core": OnlineCoreType.OFFLINE},
                OnlineCoreType.OFFLINE,
                OnlineCoreType.ONLINE,
                0,
                id="vector_offline_entry",
            ),
            pytest.param(
                "vector_offline_output_boundary",
                OnlineLinear,
                make_vec_8d,
                {"output_core": OnlineCoreType.OFFLINE},
                OnlineCoreType.ONLINE,
                OnlineCoreType.OFFLINE,
                0,
                id="vector_offline_output_boundary",
            ),
            pytest.param(
                "mnist_flatten_offline_entry",
                MNISTFlattenOnlineLinear,
                make_img_1ch_28x28,
                {"input_core": OnlineCoreType.OFFLINE},
                OnlineCoreType.OFFLINE,
                OnlineCoreType.ONLINE,
                1,
                id="mnist_flatten_offline_entry",
            ),
        ),
    )
    def test_graph_entry_boundary_cases(
        self,
        case_name: str,
        model_factory: Callable[[], nn.Module],
        sample_factory: Callable[[], Tensor],
        mark_overrides: dict[str, object],
        expected_input_core: OnlineCoreType,
        expected_output_core: OnlineCoreType,
        expected_transform_count: int,
    ):
        graph = compile_to_paiir(
            mark_online(model_factory(), **mark_overrides), sample_factory()
        )
        forward = first_online_forward_node(graph)
        last_forward = last_online_forward_node(graph)
        transform_count = len(find_transform_nodes(graph))

        _print_case_header("图入口边界语义", case_name)
        print("第一层在线前向输入边界：", forward.core_params.input_core.name)
        print("末层在线前向输出边界：", last_forward.core_params.output_core.name)
        print("前置变换节点数量：", transform_count)

        assert forward.core_params.input_core is expected_input_core
        assert last_forward.core_params.output_core is expected_output_core
        assert transform_count == expected_transform_count


class TestOnlineExportBridgeReport:
    @pytest.mark.parametrize(
        (
            "case_name",
            "model_factory",
            "sample_factory",
            "mark_overrides",
            "expected_cores",
            "expected_input_entries",
            "expected_output_entries",
            "expected_first_input_core",
            "expected_last_output_core",
        ),
        (
            pytest.param(
                "single_layer_fc_export",
                OnlineLinear,
                make_vec_8d,
                {},
                4,
                8,
                4,
                OnlineCoreType.ONLINE,
                OnlineCoreType.ONLINE,
                id="single_layer_fc_export",
            ),
            pytest.param(
                "two_layer_fc_export",
                TwoLayerOnlineLinear,
                make_vec_8d,
                {},
                7,
                8,
                4,
                OnlineCoreType.ONLINE,
                OnlineCoreType.ONLINE,
                id="two_layer_fc_export",
            ),
            pytest.param(
                "single_layer_offline_entry_export",
                OnlineLinear,
                make_vec_8d,
                {"input_core": OnlineCoreType.OFFLINE},
                4,
                8,
                4,
                OnlineCoreType.OFFLINE,
                OnlineCoreType.ONLINE,
                id="single_layer_offline_entry_export",
            ),
            pytest.param(
                "single_layer_offline_output_export",
                OnlineLinear,
                make_vec_8d,
                {"output_core": OnlineCoreType.OFFLINE},
                4,
                8,
                4,
                OnlineCoreType.ONLINE,
                OnlineCoreType.OFFLINE,
                id="single_layer_offline_output_export",
            ),
        ),
    )
    def test_export_bridge_cases(
        self,
        case_name: str,
        model_factory: Callable[[], nn.Module],
        sample_factory: Callable[[], Tensor],
        mark_overrides: dict[str, object],
        expected_cores: int,
        expected_input_entries: int,
        expected_output_entries: int,
        expected_first_input_core: OnlineCoreType,
        expected_last_output_core: OnlineCoreType,
    ):
        export_dir, graph, mapper = export_online_graph(
            DEBUG_EXPORT_ROOT,
            case_name,
            model_factory(),
            sample_input=sample_factory(),
            **mark_overrides,
        )
        artifacts = load_compile_artifacts(export_dir)
        thread = artifacts.io_mapping.threads[0]
        first_core = mapper.coreplacements[0].core_config
        placements_by_name = {
            core_placement.node_names[0]: core_placement
            for core_placement in mapper.coreplacements
        }
        last_forward = last_online_forward_node(graph)
        last_forward_core = placements_by_name[last_forward.name].core_config

        _print_case_header("backendv2 在线桥接导出", case_name)
        print("导出配置文件：", export_dir / "proto" / "config.pb")
        print("在线核数量：", len(mapper.coreplacements))
        print("core_ticks 数量：", len(thread.core_ticks))
        print("输入映射条目：", len(thread.input_mappings.items[0].entries))
        print("输出映射条目：", len(thread.output_mappings.items[0].entries))
        print("首个在线核输入边界：", _online_core_type_name(first_core.input_core))
        print(
            "末层在线核输出边界：",
            _online_core_type_name(last_forward_core.output_core),
        )

        assert len(mapper.coreplacements) == expected_cores
        assert len(thread.core_ticks) == expected_cores
        assert len(thread.input_mappings.items[0].entries) == expected_input_entries
        assert len(thread.output_mappings.items[0].entries) == expected_output_entries
        assert first_core.input_core == expected_first_input_core
        assert last_forward_core.output_core == expected_last_output_core


class TestOnlineRuntimeMappingReport:
    @pytest.mark.parametrize(
        (
            "case_name",
            "model_factory",
            "sample_factory",
            "mark_overrides",
            "expected_input_entries",
            "expected_output_entries",
            "expected_target_lcn",
        ),
        (
            pytest.param(
                "single_layer_default_lcn",
                OnlineLinear,
                make_vec_8d,
                {},
                8,
                4,
                LCN_EX.LCN_1X,
                id="single_layer_default_lcn",
            ),
            pytest.param(
                "wide_lcn_2x",
                WideOnlineLinear,
                lambda: torch.randn(1, 2048),
                uniform_online_lcn_kwargs(LCN_EX.LCN_2X),
                2048,
                4,
                LCN_EX.LCN_2X,
                id="wide_lcn_2x",
            ),
            pytest.param(
                "mnist_flatten_default_lcn",
                MNISTFlattenOnlineLinear,
                make_img_1ch_28x28,
                {},
                784,
                10,
                LCN_EX.LCN_1X,
                id="mnist_flatten_default_lcn",
            ),
        ),
    )
    def test_runtime_mapping_cases(
        self,
        case_name: str,
        model_factory: Callable[[], nn.Module],
        sample_factory: Callable[[], Tensor],
        mark_overrides: dict[str, object],
        expected_input_entries: int,
        expected_output_entries: int,
        expected_target_lcn: LCN_EX,
    ):
        sample_input = sample_factory()
        export_dir, _, _ = export_online_graph(
            DEBUG_EXPORT_ROOT,
            case_name,
            model_factory(),
            sample_input=sample_input,
            **mark_overrides,
        )
        artifacts = load_compile_artifacts(export_dir)
        thread = artifacts.io_mapping.threads[0]
        input_mapping = thread.input_mappings.items[0]
        output_mapping = thread.output_mappings.items[0]
        frames = encode_online_input_frames(
            artifacts,
            input_mapping.name,
            sample_input.numpy().astype(np.float16, copy=False),
            thread_id=int(thread.thread_id),
        )
        output_tables = build_output_mapping_tables(
            artifacts, thread_id=int(thread.thread_id)
        )
        first_boundary = next(iter(output_tables.values())).boundary
        boundary_frames = None
        if first_boundary is not None and first_boundary.data_route_offset is not None:
            boundary_frames = encode_online_boundary_output_frames(
                output_tables,
                output_mapping.name,
                np.ones(tuple(output_mapping.shape.size), dtype=np.float16),
            )
        max_tick_relative = max(
            int(entry.tick_relative) for entry in input_mapping.entries
        )

        _print_case_header("输入映射与运行时编解码", case_name)
        print("target_lcn：", int(thread.output_mappings.target_lcn))
        print("输入映射条目：", len(input_mapping.entries))
        print("输出映射条目：", len(output_mapping.entries))
        print("最大 tick_relative：", max_tick_relative)
        print("编码后输入帧数：", frames.shape[0])
        print("输出映射表名称：", "，".join(sorted(output_tables)))
        if first_boundary is not None:
            print("输出边界目标坐标：", first_boundary.target_coord)
            print("输出边界入口侧：", first_boundary.data_ingress_side)
        if boundary_frames is not None:
            print("边界 route 编码后输出帧数：", boundary_frames.shape[0])

        assert len(input_mapping.entries) == expected_input_entries
        assert len(output_mapping.entries) == expected_output_entries
        assert int(thread.output_mappings.target_lcn) == int(expected_target_lcn)
        assert frames.shape[0] > 0
        assert output_tables
        if boundary_frames is not None:
            assert boundary_frames.shape[0] > 0


class TestOnlineValidationNetworkReport:
    @pytest.mark.parametrize(
        ("case_name", "hidden_features"),
        (
            pytest.param("sj_mnist_hidden64", 64, id="sj_mnist_hidden64"),
            pytest.param("sj_mnist_hidden128", 128, id="sj_mnist_hidden128"),
        ),
    )
    def test_validation_network_compile_cases(
        self, case_name: str, hidden_features: int
    ):
        graph = compile_to_paiir(
            SJMNISTValidationNet(hidden_features=hidden_features),
            make_img_1ch_28x28(),
        )
        linear_nodes = [
            node
            for node in find_nodes(graph, StandaloneCompOp)
            if isinstance(node.comp, nn.Linear)
        ]
        seq_nodes = find_nodes(graph, SequentialOp)
        offline_core_nodes = offline_nodes(graph)

        _print_case_header("统一验证网络编译", case_name)
        print("线性层节点数：", len(linear_nodes))
        print("串行计算节点数：", len(seq_nodes))
        print("离线核节点数：", len(offline_core_nodes))

        assert len(linear_nodes) >= 1
        assert len(seq_nodes) >= 1
        assert len(offline_core_nodes) >= 2
