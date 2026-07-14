import numpy as np
import pytest
import torch
from torch import nn

from paibox.paiir.exceptions import UnsupportedNIRNodeError
from paibox.paiir.ir import (
    CoreNeuronV25,
    IFNodeV25,
    InputNode,
    LeakyBeta0NodeV25,
    LIFNodeV25,
    OutputNode,
    PAIIRGraph,
    ShapeStage,
    StandaloneActOp,
    StandaloneCompOp,
    TensorLayout,
    TransformOp,
)
from paibox.paiir.nn import SumPool2d

nir = pytest.importorskip("nir")

from paibox.paiir.nir_exchange import (  # noqa: E402
    compile_from_nir,
    export_to_nir,
    import_from_nir,
)


def _single_nir_op_graph(
    node,
    *,
    input_shape: tuple[int, ...] | None = None,
    output_shape: tuple[int, ...] | None = None,
) -> nir.NIRGraph:
    if input_shape is None:
        input_shape = tuple(int(dim) for dim in node.input_type["input"])
    if output_shape is None:
        output_shape = tuple(int(dim) for dim in node.output_type["output"])

    return nir.NIRGraph(
        nodes={
            "input": nir.Input({"input": np.asarray(input_shape)}),
            "op": node,
            "output": nir.Output({"output": np.asarray(output_shape)}),
        },
        edges=[("input", "op"), ("op", "output")],
    )


def _linear_if_graph() -> nir.NIRGraph:
    return nir.NIRGraph(
        nodes={
            "input": nir.Input({"input": np.asarray([3])}),
            "affine": nir.Affine(
                weight=np.ones((2, 3), dtype=np.float32),
                bias=np.zeros((2,), dtype=np.float32),
            ),
            "if": nir.IF(
                r=np.full((2,), 10_000.0, dtype=np.float32),
                v_threshold=np.ones((2,), dtype=np.float32),
                v_reset=np.zeros((2,), dtype=np.float32),
            ),
            "output": nir.Output({"output": np.asarray([2])}),
        },
        edges=[("input", "affine"), ("affine", "if"), ("if", "output")],
    )


def _layout(shape: tuple[int, ...]) -> TensorLayout:
    full_shape = torch.Size((1, *shape))
    return TensorLayout(full_shape, tuple(range(len(full_shape))))


def _manual_paiir_graph(
    node, input_shape: tuple[int, ...], output_shape: tuple[int, ...]
):
    layout_in = _layout(input_shape)
    layout_out = _layout(output_shape)

    graph = PAIIRGraph("manual")
    inp = InputNode(layout_in.shape, layout_in.dims)
    inp.name = "input"
    graph.add_node(inp)

    node.name = "op"
    node.input_layouts = (layout_in,)
    node.output_layouts = (layout_out,)
    graph.add_node(node)

    out = OutputNode(layout_out.shape, layout_out.dims)
    out.name = "output"
    graph.add_node(out)

    graph.add_edge("input", "op")
    graph.add_edge("op", "output")
    return graph


def _manual_comp_graph(
    comp: nn.Module, input_shape: tuple[int, ...], output_shape: tuple[int, ...]
) -> PAIIRGraph:
    return _manual_paiir_graph(StandaloneCompOp(comp), input_shape, output_shape)


def _manual_flatten_graph() -> PAIIRGraph:
    output_shape = torch.Size((1, 24))

    def shape_fn(_shape: torch.Size) -> torch.Size:
        return output_shape

    return _manual_paiir_graph(
        TransformOp((ShapeStage(shape_fn),)), input_shape=(2, 3, 4), output_shape=(24,)
    )


def _manual_linear_if_paiir_graph(*, soft_reset: bool = False) -> PAIIRGraph:
    layout_in = _layout((3,))
    layout_out = _layout((2,))

    graph = PAIIRGraph("manual")
    inp = InputNode(layout_in.shape, layout_in.dims)
    inp.name = "input"
    graph.add_node(inp)

    comp = StandaloneCompOp(nn.Linear(3, 2, bias=True))
    comp.name = "op"
    comp.input_layouts = (layout_in,)
    comp.output_layouts = (layout_out,)
    graph.add_node(comp)

    act = StandaloneActOp(
        IFNodeV25(v_threshold=1.0, v_reset=None if soft_reset else 0.0)
    )
    act.name = "if"
    act.input_layouts = (layout_out,)
    act.output_layouts = (layout_out,)
    graph.add_node(act)

    out = OutputNode(layout_out.shape, layout_out.dims)
    out.name = "output"
    graph.add_node(out)

    graph.add_edge("input", "op")
    graph.add_edge("op", "if")
    graph.add_edge("if", "output")
    return graph


@pytest.mark.parametrize(
    ("node", "has_bias"),
    [
        pytest.param(nir.Linear(np.ones((2, 3), dtype=np.float32)), False, id="linear"),
        pytest.param(
            nir.Affine(
                weight=np.ones((2, 3), dtype=np.float32),
                bias=np.zeros((2,), dtype=np.float32),
            ),
            True,
            id="affine",
        ),
    ],
)
def test_import_nir_linear_and_affine(node, has_bias):
    graph = import_from_nir(_single_nir_op_graph(node))

    comp_op = graph.nodes["op"]
    assert isinstance(comp_op, StandaloneCompOp)
    assert isinstance(comp_op.comp, nn.Linear)
    assert (comp_op.comp.bias is not None) is has_bias


def test_import_nir_if_graph():
    graph = import_from_nir(_linear_if_graph(), dt=1e-4)

    assert isinstance(graph.nodes["input"], InputNode)
    assert isinstance(graph.nodes["affine"], StandaloneCompOp)
    assert isinstance(graph.nodes["if"], StandaloneActOp)
    assert isinstance(graph.nodes["if"].act, IFNodeV25)
    assert graph.nodes["if"].act.thres_pos == 2.0
    assert isinstance(graph.nodes["output"], OutputNode)


@pytest.mark.parametrize(
    ("node", "expected_type", "input_shape", "output_shape"),
    [
        pytest.param(
            nir.Conv1d(
                input_shape=8,
                weight=np.ones((3, 2, 3), dtype=np.float32),
                stride=1,
                padding=1,
                dilation=1,
                groups=1,
                bias=np.zeros((3,), dtype=np.float32),
            ),
            nn.Conv1d,
            None,
            None,
            id="conv1d",
        ),
        pytest.param(
            nir.Conv2d(
                input_shape=(5, 5),
                weight=np.ones((3, 2, 3, 3), dtype=np.float32),
                stride=1,
                padding=1,
                dilation=1,
                groups=1,
                bias=np.zeros((3,), dtype=np.float32),
            ),
            nn.Conv2d,
            None,
            None,
            id="conv2d",
        ),
        pytest.param(
            nir.AvgPool2d(
                kernel_size=np.asarray([2, 2]),
                stride=np.asarray([2, 2]),
                padding=np.asarray([0, 0]),
            ),
            nn.AvgPool2d,
            (3, 4, 4),
            (3, 2, 2),
            id="avgpool2d",
        ),
        pytest.param(
            nir.SumPool2d(
                kernel_size=np.asarray([2, 2]),
                stride=np.asarray([2, 2]),
                padding=np.asarray([0, 0]),
            ),
            SumPool2d,
            (3, 4, 4),
            (3, 2, 2),
            id="sumpool2d",
        ),
    ],
)
def test_import_nir_supported_compute_ops(
    node, expected_type, input_shape, output_shape
):
    graph = import_from_nir(
        _single_nir_op_graph(node, input_shape=input_shape, output_shape=output_shape)
    )

    comp_op = graph.nodes["op"]
    assert isinstance(comp_op, StandaloneCompOp)
    assert isinstance(comp_op.comp, expected_type)


def test_import_nir_flatten():
    graph = import_from_nir(
        _single_nir_op_graph(
            nir.Flatten(input_type={"input": np.asarray([2, 3, 4])}),
        )
    )

    assert isinstance(graph.nodes["op"], TransformOp)


@pytest.mark.parametrize(
    ("r_value", "expected_decay_input"),
    [
        pytest.param(1.0, True, id="decay_input_true"),
        pytest.param(2.0, False, id="decay_input_false"),
    ],
)
def test_import_export_nir_lif_graph(r_value, expected_decay_input):
    graph = _single_nir_op_graph(
        nir.LIF(
            tau=np.full((2,), 2e-4, dtype=np.float64),
            r=np.full((2,), r_value, dtype=np.float64),
            v_leak=np.zeros((2,), dtype=np.float64),
            v_threshold=np.ones((2,), dtype=np.float64),
            v_reset=np.zeros((2,), dtype=np.float64),
        )
    )

    paiir_graph = import_from_nir(graph, dt=1e-4)
    lif_op = paiir_graph.nodes["op"]
    assert isinstance(lif_op, StandaloneActOp)
    lif = lif_op.act
    assert isinstance(lif, LIFNodeV25)
    assert lif.tau == pytest.approx(2.0)
    assert lif.leak_multi_input == expected_decay_input

    exported_lif = export_to_nir(paiir_graph, dt=1e-4).nodes["op"]
    assert isinstance(exported_lif, nir.LIF)
    assert np.allclose(exported_lif.tau, 2e-4)
    assert np.allclose(exported_lif.r, r_value)


@pytest.mark.parametrize(
    ("graph_factory", "expected_type"),
    [
        pytest.param(
            lambda: _manual_comp_graph(nn.Linear(3, 2, bias=False), (3,), (2,)),
            nir.Linear,
            id="linear",
        ),
        pytest.param(
            lambda: _manual_comp_graph(
                nn.Conv1d(2, 3, kernel_size=3, padding=1), (2, 8), (3, 8)
            ),
            nir.Conv1d,
            id="conv1d",
        ),
        pytest.param(
            lambda: _manual_comp_graph(
                nn.Conv2d(2, 3, kernel_size=3, padding=1), (2, 5, 5), (3, 5, 5)
            ),
            nir.Conv2d,
            id="conv2d",
        ),
        pytest.param(
            lambda: _manual_comp_graph(nn.AvgPool2d(2, 2), (3, 4, 4), (3, 2, 2)),
            nir.AvgPool2d,
            id="avgpool2d",
        ),
        pytest.param(
            lambda: _manual_comp_graph(SumPool2d(2, 2), (3, 4, 4), (3, 2, 2)),
            nir.SumPool2d,
            id="sumpool2d",
        ),
        pytest.param(_manual_flatten_graph, nir.Flatten, id="flatten"),
    ],
)
def test_export_paiir_supported_ops(graph_factory, expected_type):
    exported = export_to_nir(graph_factory(), dt=1e-4)

    assert isinstance(exported.nodes["op"], expected_type)


def _string_padding_graph() -> nir.NIRGraph:
    conv = nir.Conv2d(
        input_shape=(4, 4),
        weight=np.ones((1, 1, 3, 3), dtype=np.float32),
        stride=1,
        padding="valid",
        dilation=1,
        groups=1,
        bias=np.zeros((1,), dtype=np.float32),
    )
    return _single_nir_op_graph(conv)


def _unsupported_node_graph(node) -> nir.NIRGraph:
    return _single_nir_op_graph(node, input_shape=(2,), output_shape=(2,))


def _fan_in_graph() -> nir.NIRGraph:
    return nir.NIRGraph(
        nodes={
            "a": nir.Input({"input": np.asarray([2])}),
            "b": nir.Input({"input": np.asarray([2])}),
            "op": nir.Linear(np.ones((2, 2), dtype=np.float32)),
            "output": nir.Output({"output": np.asarray([2])}),
        },
        edges=[("a", "op"), ("b", "op"), ("op", "output")],
        type_check=False,
    )


def _nested_graph() -> nir.NIRGraph:
    nested = nir.NIRGraph(
        nodes={
            "input": nir.Input({"input": np.asarray([2])}),
            "output": nir.Output({"output": np.asarray([2])}),
        },
        edges=[("input", "output")],
    )
    return nir.NIRGraph(
        nodes={
            "input": nir.Input({"input": np.asarray([2])}),
            "op": nested,
            "output": nir.Output({"output": np.asarray([2])}),
        },
        edges=[("input", "op"), ("op", "output")],
        type_check=False,
    )


def _cycle_graph() -> nir.NIRGraph:
    return nir.NIRGraph(
        nodes={
            "input": nir.Input({"input": np.asarray([2])}),
            "op": nir.Linear(np.ones((2, 2), dtype=np.float32)),
        },
        edges=[("input", "op"), ("op", "input")],
        type_check=False,
    )


@pytest.mark.parametrize(
    ("graph_factory", "match"),
    [
        pytest.param(_string_padding_graph, "padding", id="string_padding"),
        pytest.param(
            lambda: _unsupported_node_graph(
                nir.Delay(
                    delay=np.ones((2,), dtype=np.float32),
                    input_type={"input": np.asarray([2])},
                )
            ),
            "whitelist",
            id="delay",
        ),
        pytest.param(
            lambda: _unsupported_node_graph(nir.Scale(np.ones((2,), dtype=np.float32))),
            "whitelist",
            id="scale",
        ),
        pytest.param(
            lambda: _unsupported_node_graph(
                nir.Threshold(
                    np.ones((2,), dtype=np.float32),
                    input_type={"input": np.asarray([2])},
                )
            ),
            "whitelist",
            id="threshold",
        ),
        pytest.param(
            lambda: _unsupported_node_graph(
                nir.CubaLIF(
                    tau_syn=np.ones((2,), dtype=np.float32),
                    tau_mem=np.ones((2,), dtype=np.float32),
                    r=np.ones((2,), dtype=np.float32),
                    v_leak=np.zeros((2,), dtype=np.float32),
                    v_threshold=np.ones((2,), dtype=np.float32),
                    v_reset=np.zeros((2,), dtype=np.float32),
                    input_type={"input": np.asarray([2])},
                )
            ),
            "whitelist",
            id="cubalif",
        ),
        pytest.param(_fan_in_graph, "fan-in", id="fan_in"),
        pytest.param(_nested_graph, "nested", id="nested"),
        pytest.param(_cycle_graph, "cycle|cyclic", id="cycle"),
    ],
)
def test_import_nir_rejects_unsupported_graphs(graph_factory, match):
    with pytest.raises(UnsupportedNIRNodeError, match=match):
        import_from_nir(graph_factory())


def test_export_nir_adjusts_threshold_boundary_and_round_trips():
    nir_graph = export_to_nir(_manual_linear_if_paiir_graph(), dt=1e-4)
    exported_if = nir_graph.nodes["if"]
    assert isinstance(exported_if, nir.IF)
    assert np.allclose(exported_if.v_threshold, 0.5)

    linear = nir_graph.nodes["op"]
    assert isinstance(linear, nir.Affine)

    round_trip = import_from_nir(nir_graph, dt=1e-4)

    assert isinstance(round_trip.nodes["op"], StandaloneCompOp)
    assert isinstance(round_trip.nodes["if"].act, IFNodeV25)
    assert round_trip.nodes["if"].act.thres_pos == 1.0


def test_export_nir_rejects_soft_reset():
    with pytest.raises(UnsupportedNIRNodeError, match="hard reset"):
        export_to_nir(_manual_linear_if_paiir_graph(soft_reset=True), dt=1e-4)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("reset_v", torch.tensor([0.0, 0.0]), id="reset-v"),
        pytest.param("leak_multi_mode", torch.tensor([1, 1]), id="leak-mode"),
    ],
)
def test_export_nir_rejects_vector_neuron_parameters(field, value):
    graph = _manual_linear_if_paiir_graph()
    setattr(graph.nodes["if"].act, field, value)

    with pytest.raises(UnsupportedNIRNodeError, match="vector neuron parameters"):
        export_to_nir(graph, dt=1e-4)


def test_export_nir_rejects_mixed_neuron_dynamics():
    graph = _manual_linear_if_paiir_graph()
    graph.nodes["if"].act = CoreNeuronV25(
        leak_multi_mode=torch.tensor([0, 1]),
        leak_tau_shift=torch.tensor([0, -1]),
    )

    with pytest.raises(UnsupportedNIRNodeError, match="mixed IF/LIF"):
        export_to_nir(graph, dt=1e-4)


def test_export_nir_rejects_beta_zero_lif_endpoint():
    graph = _manual_linear_if_paiir_graph()
    graph.nodes["if"].act = LeakyBeta0NodeV25()

    with pytest.raises(UnsupportedNIRNodeError, match="cannot represent beta=0"):
        export_to_nir(graph, dt=1e-4)


def test_compile_from_nir_runs_standard_paiir_passes():
    graph = compile_from_nir(_linear_if_graph(), dt=1e-4)

    assert graph.topo_sort() == ["input", "SequentialOp_0", "output"]
