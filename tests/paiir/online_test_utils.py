import shutil
from pathlib import Path

from paicorelib import LCN_EX
from torch import Tensor, nn

from paibox.backendv2.mapper import Mapper
from paibox.paiir import compile_to_paiir, mark_online
from paibox.paiir.ir.calc_params import OnlineCoreSemanticMode
from paibox.paiir.ir.op_node import OnlineCoreOp
from tests.paiir.conftest import make_vec_8d


class OnlineLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(8, 4, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class TwoLayerOnlineLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(8, 6, bias=False)
        self.linear2 = nn.Linear(6, 4, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.linear1(x))


class MNISTFlattenOnlineLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(28 * 28, 10, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x.flatten(1))


class WideOnlineLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2048, 4, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


def online_nodes(graph) -> list[OnlineCoreOp]:
    return [
        graph.nodes[name]
        for name in graph.topo_sort()
        if isinstance(graph.nodes[name], OnlineCoreOp)
    ]


def first_online_forward_node(graph) -> OnlineCoreOp:
    for node in online_nodes(graph):
        if node.core_params.semantic_mode is OnlineCoreSemanticMode.FORWARD:
            return node
    raise AssertionError("expected at least one online forward node")


def last_online_forward_node(graph) -> OnlineCoreOp:
    for node in reversed(online_nodes(graph)):
        if node.core_params.semantic_mode is OnlineCoreSemanticMode.FORWARD:
            return node
    raise AssertionError("expected at least one online forward node")


def single_online_node(graph, semantic_mode: OnlineCoreSemanticMode) -> OnlineCoreOp:
    nodes = [
        node
        for node in online_nodes(graph)
        if node.core_params.semantic_mode is semantic_mode
    ]
    assert len(nodes) == 1
    return nodes[0]


def uniform_online_lcn_kwargs(lcn: LCN_EX) -> dict[str, LCN_EX]:
    return {
        "lcn_at": lcn,
        "lcn_mp": lcn,
        "lcn_lg": lcn,
        "target_lcn_at": lcn,
        "target_lcn_mp": lcn,
        "target_lcn_lg": lcn,
    }


def export_online_graph(
    export_root: Path,
    case_name: str,
    model: nn.Module,
    *,
    sample_input: Tensor | None = None,
    compile_kwargs: dict[str, object] | None = None,
    mapper_timesteps: int | None = None,
    **mark_overrides,
) -> tuple[Path, object, Mapper]:
    export_dir = export_root / case_name
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    if sample_input is None:
        sample_input = make_vec_8d()

    graph = compile_to_paiir(
        mark_online(model, **mark_overrides),
        sample_input,
        **({} if compile_kwargs is None else compile_kwargs),
    )
    mapper = Mapper()
    mapper.compile(
        graph,
        export_dir,
        target_platform="x86",
        debug=True,
        timesteps=mapper_timesteps,
    )
    return export_dir, graph, mapper
