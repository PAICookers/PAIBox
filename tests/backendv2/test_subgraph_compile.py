from pathlib import Path

import pytest
import torch
from torch import nn

from paibox.backendv2 import Mapper
from paibox.backendv2.compile_plan import build_subgraph_compile_plan
from paibox.backendv2.generated.proto.compile_artifacts_pb2 import CompileArtifacts
from paibox.paiir import LIFNodeV25, PAIIRGraph, compile_to_paiir
from paibox.paiir.ir import CPUOp, InputNode, OutputNode


class _LinearLIF(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2)
        self.neuron = LIFNodeV25(tau=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.neuron(self.linear(x))


def _graph() -> PAIIRGraph:
    return compile_to_paiir(_LinearLIF().eval(), torch.zeros(1, 3), timesteps=2)


def test_subgraph_plan_namespaces_without_mutating_sources() -> None:
    first = _graph()
    second = _graph()
    first_names = tuple(first.nodes)
    second_names = tuple(second.nodes)

    plan = build_subgraph_compile_plan((first, second), timesteps=2, auto_reset=True)

    assert tuple(first.nodes) == first_names
    assert tuple(second.nodes) == second_names
    assert plan.prefixes == ("g0/", "g1/")
    assert all(name.startswith(("g0/", "g1/")) for name in plan.graph.nodes)

    core_nodes = [node for node in plan.graph.nodes.values() if hasattr(node, "core_params")]
    assert [node.core_params.tick_start for node in core_nodes] == [1, 2]
    assert all(node.core_params.tick_duration == 0 for node in core_nodes)
    assert all(node.core_params.tick_initial == 2 for node in core_nodes)


def test_subgraph_plan_rejects_cpu_nodes() -> None:
    graph = PAIIRGraph("cpu")
    input_node = InputNode(torch.Size([1]))
    cpu_node = CPUOp()
    output_node = OutputNode(torch.Size([1]))
    graph.add_node(input_node)
    graph.add_node(cpu_node)
    graph.add_node(output_node)
    graph.add_edge(input_node.name, cpu_node.name)
    graph.add_edge(cpu_node.name, output_node.name)

    with pytest.raises(ValueError, match="pure PAICORE"):
        build_subgraph_compile_plan((graph,), timesteps=1, auto_reset=True)


def test_mapper_exports_multiple_namespaced_subgraphs(tmp_path: Path) -> None:
    Mapper().compile_subgraphs(
        (_graph(), _graph()),
        tmp_path,
        target_platform="x86",
        export_merged_frames=False,
        export_proto_python=False,
    )

    artifacts = CompileArtifacts()
    artifacts.ParseFromString((tmp_path / "proto" / "config.pb").read_bytes())
    thread = artifacts.io_mapping.threads[0]
    input_names = {item.name for item in thread.input_mappings.items}
    assert len(input_names) == 2
    assert {name.split("/", 1)[0] for name in input_names} == {"g0", "g1"}
    assert thread.runtime.timesteps == 2
    assert thread.runtime.tick_depth == 2
