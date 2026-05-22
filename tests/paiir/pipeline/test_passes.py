"""Tests for `paibox.paiir.pipeline.passes`.

This file intentionally groups compile-time pass behavior that is implemented
in `PAIBox/paibox/paiir/pipeline/passes.py`:

- add specialization / fusion
- tick assignment
- validation / deployability checks
- signal-semantics propagation
"""

import pytest
import torch
from paicorelib import DataSign, DataWidth, OutputType, PoolingMode, SNNMode
from spikingjelly.activation_based import neuron
from torch import nn

from paibox.paiir.ir.add_ops import (
    AddOperandKind,
    AddOperandSpec,
    GeneralAddOp,
    PotentialAddOp,
)
from paibox.paiir.ir.calc_params import OfflineCoreParams
from paibox.paiir.ir.core_neuron import ANNNodeV25, CoreNeuronV25, IFNodeV25, LIFNodeV25
from paibox.paiir.ir.graph import Edge, PAIIRGraph
from paibox.paiir.ir.ir_base import InputNode, OutputNode
from paibox.paiir.ir.lut_activation import LutReLU, LutSigmoid, LutTanh
from paibox.paiir.ir.op_node import (
    AccumulateOp,
    ConcatOp,
    CPUOp,
    OfflineCoreOp,
    SequentialOp,
    SplitOp,
    StandaloneActOp,
    StandaloneCompOp,
    TensorLayout,
    TransformOp,
)
from paibox.paiir.ir.signal_domain import SignalDomain
from paibox.paiir.lowering.converter import torch_to_paiir
from paibox.paiir.pipeline.compile import compile_to_paiir
from paibox.paiir.pipeline.passes import (
    GraphCleanupWarning,
    GraphValidationError,
    assign_tick_params,
    flatten_general_add_chains,
    fuse_to_offline_cores,
    propagate_data_format,
    propagate_signal_semantics,
    specialize_general_adds,
    validate_compiled_graph,
    validate_deployable_graph,
    validate_graph,
)
from tests.paiir.conftest import (
    ANNClassifier,
    ANNConvBNReLU,
    ANNResidualSubtract,
    MultiInputMerge,
    SNNDepthwiseSeparable,
    SNNFlattenTransition,
    SNNResidualAdd,
    SNNTwoLayer,
    SNNWithMaxPool,
    SPPFBlock,
    convert_and_fuse,
    find_first,
    find_node_names,
    find_nodes,
    find_transform_nodes,
    make_img_1ch_4x4,
    make_img_3ch_8x8,
    make_img_16ch_8x8,
    make_vec_8d,
)


def _layout(shape: tuple[int, ...] | torch.Size, dims: tuple[int, ...]) -> TensorLayout:
    return TensorLayout(shape=torch.Size(shape), dims=dims)


def _set_single_layouts(
    node,
    input_shape: tuple[int, ...] | torch.Size,
    output_shape: tuple[int, ...] | torch.Size,
    input_dims: tuple[int, ...] = (),
    output_dims: tuple[int, ...] = (),
) -> None:
    node.input_layouts = (_layout(input_shape, input_dims),)
    node.output_layouts = (_layout(output_shape, output_dims),)


def _set_multi_input_single_output_layouts(
    node,
    input_shapes: list[tuple[int, ...] | torch.Size],
    output_shape: tuple[int, ...] | torch.Size,
    input_dims: list[tuple[int, ...]],
    output_dims: tuple[int, ...],
) -> None:
    node.input_layouts = tuple(
        _layout(shape, dims) for shape, dims in zip(input_shapes, input_dims)
    )
    node.output_layouts = (_layout(output_shape, output_dims),)


def _set_split_layouts(
    node: SplitOp,
    input_shape: tuple[int, ...] | torch.Size,
    output_shapes: list[tuple[int, ...] | torch.Size],
    dims: tuple[int, ...],
) -> None:
    node.input_layouts = (_layout(input_shape, dims),)
    node.output_layouts = tuple(_layout(shape, dims) for shape in output_shapes)


class TestSNNConversion:
    """Test SNN network conversion through the full pipeline."""

    def test_two_layer_snn(self):
        """Conv-LIF -> Conv-IF: two SequentialOps after fusion."""
        model = SNNTwoLayer()
        fused = convert_and_fuse(model, make_img_3ch_8x8())

        seq_nodes = find_nodes(fused, SequentialOp)
        assert len(seq_nodes) == 2

        comp_types = {type(n.comp) for n in seq_nodes}
        assert comp_types == {nn.Conv2d}
        act_types = sorted(type(n.act).__name__ for n in seq_nodes)
        assert act_types == ["IFNodeV25", "LIFNodeV25"]

        assert find_nodes(fused, StandaloneCompOp) == []
        assert find_nodes(fused, StandaloneActOp) == []

    def test_residual_add(self):
        """Two conv branches + add + LIF: AccumulateOp after fusion."""
        model = SNNResidualAdd()
        fused = convert_and_fuse(model, make_img_3ch_8x8())

        accum_nodes = find_nodes(fused, AccumulateOp)
        assert len(accum_nodes) == 1
        assert len(accum_nodes[0].comps) == 2
        assert isinstance(accum_nodes[0].act, LIFNodeV25)
        assert accum_nodes[0].signs == (1, 1)

        assert find_nodes(fused, PotentialAddOp) == []
        assert find_nodes(fused, StandaloneCompOp) == []

    def test_maxpool_chain(self):
        """Conv-LIF -> MaxPool-LIF: pooling mode inferred."""
        model = SNNWithMaxPool()
        fused = convert_and_fuse(model, make_img_3ch_8x8())

        seq_nodes = find_nodes(fused, SequentialOp)
        assert len(seq_nodes) == 2

        pool_ops = [n for n in seq_nodes if isinstance(n.comp, nn.MaxPool2d)]
        assert len(pool_ops) == 1

        assert pool_ops[0].core_params.pooling_mode == PoolingMode.MAX

    def test_depthwise_separable(self):
        """DWConv-LIF -> PWConv-LIF: grouped conv handled."""
        model = SNNDepthwiseSeparable()
        fused = convert_and_fuse(model, make_img_16ch_8x8())

        seq_nodes = find_nodes(fused, SequentialOp)
        assert len(seq_nodes) == 2

        conv_ops = [n for n in seq_nodes if isinstance(n.comp, nn.Conv2d)]
        groups = sorted([n.comp.groups for n in conv_ops])  # type: ignore
        assert groups == [1, 16]

    def test_flatten_transition(self):
        """Conv-IF -> flatten -> Linear-IF: flatten is preserved as one transform node."""
        model = SNNFlattenTransition()
        fused = convert_and_fuse(model, make_img_1ch_4x4())

        seq_nodes = find_nodes(fused, SequentialOp)
        assert len(seq_nodes) == 2
        transform_nodes = find_transform_nodes(fused)
        assert len(transform_nodes) == 1

        comp_types = sorted(type(n.comp).__name__ for n in seq_nodes)
        assert comp_types == ["Conv2d", "Linear"]


class TestANNConversion:
    """Test ANN network conversion through the full pipeline."""

    def test_classifier_head(self):
        """Conv-ReLU -> AvgPool -> flatten -> Linear-Sigmoid."""
        model = ANNClassifier()
        fused = convert_and_fuse(model, make_img_3ch_8x8())

        seq_nodes = find_nodes(fused, SequentialOp)
        assert len(seq_nodes) >= 2

        lut_types = {type(n.act.lut) for n in seq_nodes if n.act.lut is not None}
        assert LutReLU in lut_types
        assert LutSigmoid in lut_types

    def test_conv_bn_relu(self):
        """Conv-BN-ReLU chain: BN bypassed, Conv fuses with ReLU."""
        model = ANNConvBNReLU()
        model.eval()
        fused = convert_and_fuse(model, make_img_3ch_8x8())

        seq_nodes = find_nodes(fused, SequentialOp)
        assert len(seq_nodes) == 2
        assert all(isinstance(n.act.lut, LutReLU) for n in seq_nodes)

        for node in fused.nodes.values():
            if hasattr(node, "comp"):
                assert not isinstance(node.comp, (nn.BatchNorm1d, nn.BatchNorm2d))  # type: ignore

    def test_subtract_branch(self):
        """Two linear branches with subtraction -> tanh."""
        model = ANNResidualSubtract()
        fused = convert_and_fuse(model, make_vec_8d())

        accum_nodes = find_nodes(fused, AccumulateOp)
        assert len(accum_nodes) == 1
        assert accum_nodes[0].signs == (1, -1)
        assert isinstance(accum_nodes[0].act.lut, LutTanh)


class TestGeneralAddChainFlattening:
    """Test add-chain canonicalization before add specialization."""

    class LinearAddChain(nn.Module):
        def __init__(self, num_terms: int):
            super().__init__()
            self.linears = nn.ModuleList(nn.Linear(8, 4) for _ in range(num_terms))
            self.relu = nn.ReLU()

        def forward(self, x):
            terms = [linear(x) for linear in self.linears]
            result = terms[0]
            for term in terms[1:]:
                result = result + term
            return self.relu(result)

    class SubtractLeftChain(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_a = nn.Linear(8, 4)
            self.linear_b = nn.Linear(8, 4)
            self.linear_c = nn.Linear(8, 4)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.linear_a(x) - self.linear_b(x) + self.linear_c(x))

    class SubtractRightChain(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_a = nn.Linear(8, 4)
            self.linear_b = nn.Linear(8, 4)
            self.linear_c = nn.Linear(8, 4)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.linear_a(x) + (self.linear_b(x) - self.linear_c(x)))

    class ThreeConvLIFAdd(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_a = nn.Conv2d(3, 4, 3, padding=1)
            self.conv_b = nn.Conv2d(3, 4, 3, padding=1)
            self.conv_c = nn.Conv2d(3, 4, 3, padding=1)
            self.lif = neuron.LIFNode(tau=2.0)

        def forward(self, x):
            return self.lif(self.conv_a(x) + self.conv_b(x) + self.conv_c(x))

    class MultiInputThreeLinearAdd(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_a = nn.Linear(8, 4)
            self.linear_b = nn.Linear(8, 4)
            self.linear_c = nn.Linear(8, 4)
            self.relu = nn.ReLU()

        def forward(self, x, y, z):
            return self.relu(self.linear_a(x) + self.linear_b(y) + self.linear_c(z))

    class OrderPreservingTransformChain(nn.Module):
        def __init__(self, transform_kind: str):
            super().__init__()
            self.transform_kind = transform_kind
            self.linear_a = nn.Linear(8, 4)
            self.linear_b = nn.Linear(8, 4)
            self.linear_c = nn.Linear(8, 4)
            self.relu = nn.ReLU()

        def forward(self, x):
            a = self.linear_a(x)
            if self.transform_kind == "view":
                a = a.view_as(a)
            elif self.transform_kind == "reshape":
                a = a.reshape(a.shape)
            elif self.transform_kind == "flatten":
                a = a.flatten(1, 1)
            else:
                raise AssertionError(f"unknown transform_kind={self.transform_kind}")
            return self.relu(a + self.linear_b(x) + self.linear_c(x))

    class LayoutTransformOperandChain(nn.Module):
        def __init__(self, transform_kind: str):
            super().__init__()
            self.transform_kind = transform_kind
            self.conv_a = nn.Conv2d(3, 4, 1)
            self.conv_b = nn.Conv2d(3, 4, 1)
            self.conv_c = nn.Conv2d(3, 4, 1)
            self.relu = nn.ReLU()

        def forward(self, x):
            b = self.conv_b(x)
            if self.transform_kind == "transpose":
                b = b.transpose(2, 3)
            elif self.transform_kind == "permute":
                b = b.permute(0, 1, 3, 2)
            else:
                raise AssertionError(f"unknown transform_kind={self.transform_kind}")
            return self.relu(self.conv_a(x) + b + self.conv_c(x))

    class ShapeChangingFlattenOperandChain(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_a = nn.Conv2d(1, 1, 1)
            self.linear_b = nn.Linear(4, 4)
            self.linear_c = nn.Linear(4, 4)
            self.relu = nn.ReLU()

        def forward(self, x):
            flattened_x = x.flatten(1)
            return self.relu(
                self.conv_a(x).flatten(1)
                + self.linear_b(flattened_x)
                + self.linear_c(flattened_x)
            )

    class MultiConsumerMiddleAdd(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_a = nn.Linear(8, 4)
            self.linear_b = nn.Linear(8, 4)
            self.linear_c = nn.Linear(8, 4)
            self.relu = nn.ReLU()

        def forward(self, x):
            mid = self.linear_a(x) + self.linear_b(x)
            return self.relu(mid + self.linear_c(x)), mid

    class MultiConsumerTransformOperand(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_a = nn.Linear(8, 4)
            self.linear_b = nn.Linear(8, 4)
            self.linear_c = nn.Linear(8, 4)
            self.relu = nn.ReLU()

        def forward(self, x):
            a = self.linear_a(x)
            transformed = a.view_as(a)
            return (
                self.relu(transformed + self.linear_b(x) + self.linear_c(x)),
                transformed,
            )

    class ConstOperandChain(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_a = nn.Linear(8, 4)
            self.linear_b = nn.Linear(8, 4)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.linear_a(x) + (self.linear_b(x) + 1))

    class BroadcastOperandChain(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_a = nn.Linear(8, 4)
            self.linear_b = nn.Linear(8, 1)
            self.linear_c = nn.Linear(8, 4)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.linear_a(x) + (self.linear_b(x) + self.linear_c(x)))

    class RepeatedProducerChain(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_a = nn.Linear(8, 4)
            self.linear_b = nn.Linear(8, 4)
            self.relu = nn.ReLU()

        def forward(self, x):
            a = self.linear_a(x)
            return self.relu(a + self.linear_b(x) + a)

    class NonUnitCoeffChain(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_a = nn.Linear(8, 4)
            self.linear_b = nn.Linear(8, 4)
            self.linear_c = nn.Linear(8, 4)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(
                self.linear_a(x)
                + torch.add(self.linear_b(x), self.linear_c(x), alpha=2)
            )

    class NoActivationAddChain(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_a = nn.Linear(8, 4)
            self.linear_b = nn.Linear(8, 4)
            self.linear_c = nn.Linear(8, 4)

        def forward(self, x):
            return self.linear_a(x) + self.linear_b(x) + self.linear_c(x)

    class NoActivationSubtractChain(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_a = nn.Linear(8, 4)
            self.linear_b = nn.Linear(8, 4)
            self.linear_c = nn.Linear(8, 4)

        def forward(self, x):
            return self.linear_a(x) - self.linear_b(x) + self.linear_c(x)

    def _single_accumulate(self, compiled):
        accum_nodes = find_nodes(compiled, AccumulateOp)
        assert len(accum_nodes) == 1
        assert find_nodes(compiled, PotentialAddOp) == []
        return accum_nodes[0]

    def _single_potential_add(self, compiled):
        add_nodes = find_nodes(compiled, PotentialAddOp)
        assert len(add_nodes) == 1
        assert find_nodes(compiled, GeneralAddOp) == []
        assert find_nodes(compiled, AccumulateOp) == []
        return add_nodes[0]

    def _assert_binary_add_chain_not_flattened(self, model, *sample_inputs):
        flattened, add_nodes = self._flattened_add_nodes(model, *sample_inputs)

        assert len(add_nodes) == 2
        assert [node.coeffs for node in add_nodes] == [(1, 1), (1, 1)]
        return flattened

    def _flattened_add_nodes(self, model, *sample_inputs):
        graph = torch_to_paiir(model, *sample_inputs)
        flattened = flatten_general_add_chains(graph)
        return flattened, find_nodes(flattened, GeneralAddOp)

    def _assert_not_accumulated(self, model, *sample_inputs):
        compiled = compile_to_paiir(model, *sample_inputs)
        assert find_nodes(compiled, AccumulateOp) == []
        assert len(find_nodes(compiled, PotentialAddOp)) == 2
        return compiled

    @pytest.mark.parametrize(
        ("num_terms", "expected_signs"),
        [(3, (1, 1, 1)), (4, (1, 1, 1, 1))],
        ids=["three_terms", "four_terms"],
    )
    def test_linear_chain_compiles_to_single_accumulate(
        self, num_terms, expected_signs
    ):
        compiled = compile_to_paiir(self.LinearAddChain(num_terms), make_vec_8d())

        accum = self._single_accumulate(compiled)
        assert len(accum.comps) == num_terms
        assert accum.signs == expected_signs
        assert isinstance(accum.act, ANNNodeV25)

    def test_conv_lif_chain_compiles_to_single_accumulate(self):
        compiled = compile_to_paiir(self.ThreeConvLIFAdd(), make_img_3ch_8x8())

        accum = self._single_accumulate(compiled)
        assert len(accum.comps) == 3
        assert accum.signs == (1, 1, 1)
        assert isinstance(accum.act, LIFNodeV25)
        assert {type(comp) for comp in accum.comps} == {nn.Conv2d}

    def test_multi_input_chain_compiles_to_single_accumulate(self):
        compiled = compile_to_paiir(
            self.MultiInputThreeLinearAdd(), make_vec_8d(), make_vec_8d(), make_vec_8d()
        )

        accum = self._single_accumulate(compiled)
        assert len(accum.comps) == 3
        assert accum.signs == (1, 1, 1)
        assert len(compiled.input_nodes()) == 3

    @pytest.mark.parametrize(
        ("model_cls", "expected_signs"),
        [(SubtractLeftChain, (1, -1, 1)), (SubtractRightChain, (1, 1, -1))],
        ids=["a_minus_b_plus_c", "a_plus_b_minus_c"],
    )
    def test_chain_signs_are_preserved(self, model_cls, expected_signs):
        compiled = compile_to_paiir(model_cls(), make_vec_8d())

        accum = self._single_accumulate(compiled)
        assert accum.signs == expected_signs

    @pytest.mark.parametrize(
        ("model_cls", "expected_signs"),
        [
            (NoActivationAddChain, (1, 1, 1)),
            (NoActivationSubtractChain, (1, -1, 1)),
        ],
        ids=["add", "subtract"],
    )
    def test_no_activation_chain_compiles_to_nary_potential_add(
        self, model_cls, expected_signs
    ):
        compiled = compile_to_paiir(model_cls(), make_vec_8d())

        add = self._single_potential_add(compiled)
        assert add.signs == expected_signs
        assert len(find_nodes(compiled, StandaloneCompOp)) == 3

    @pytest.mark.parametrize("transform_kind", ["view", "reshape", "flatten"])
    def test_shape_preserving_order_preserving_operand_transform_flattens(
        self, transform_kind
    ):
        flattened, add_nodes = self._flattened_add_nodes(
            self.OrderPreservingTransformChain(transform_kind), make_vec_8d()
        )

        assert len(add_nodes) == 1
        assert add_nodes[0].coeffs == (1, 1, 1)
        assert find_nodes(flattened, TransformOp) == []

    @pytest.mark.parametrize("transform_kind", ["transpose", "permute"])
    def test_layout_transform_operand_does_not_flatten_or_accumulate(
        self, transform_kind
    ):
        x = torch.randn(1, 3, 4, 4)
        model = self.LayoutTransformOperandChain(transform_kind)

        flattened = self._assert_binary_add_chain_not_flattened(model, x)
        assert find_nodes(flattened, TransformOp)

        self._assert_not_accumulated(model, x)

    def test_shape_changing_order_preserving_transform_does_not_flatten(self):
        x = torch.randn(1, 1, 2, 2)
        model = self.ShapeChangingFlattenOperandChain()

        flattened = self._assert_binary_add_chain_not_flattened(model, x)
        transform_nodes = find_nodes(flattened, TransformOp)
        assert transform_nodes
        assert any(
            node.input_layouts[0].shape != node.output_layouts[0].shape
            for node in transform_nodes
        )

        self._assert_not_accumulated(model, x)

    @pytest.mark.parametrize(
        "model_cls",
        [MultiConsumerMiddleAdd, MultiConsumerTransformOperand],
        ids=["middle_add", "transform_operand"],
    )
    def test_multi_consumer_chain_member_does_not_flatten(self, model_cls):
        flattened = self._assert_binary_add_chain_not_flattened(
            model_cls(), make_vec_8d()
        )
        assert all(
            node.coeffs == (1, 1) for node in find_nodes(flattened, GeneralAddOp)
        )

    @pytest.mark.parametrize(
        "model_cls",
        [
            ConstOperandChain,
            BroadcastOperandChain,
            RepeatedProducerChain,
            NonUnitCoeffChain,
        ],
        ids=["const", "broadcast", "repeated_producer", "non_unit_coeff"],
    )
    def test_unsupported_operand_form_does_not_flatten(self, model_cls):
        _, add_nodes = self._flattened_add_nodes(model_cls(), make_vec_8d())

        assert len(add_nodes) == 2
        assert all(len(node.operands) == 2 for node in add_nodes)


class TestComplexPatterns:
    """Test complex/real-world network patterns."""

    def test_multi_input(self):
        """Two separate inputs merged by add + LIF."""
        model = MultiInputMerge()
        unfused = torch_to_paiir(model, make_img_3ch_8x8(), make_img_3ch_8x8())
        fused = fuse_to_offline_cores(specialize_general_adds(unfused))

        assert len(fused.input_nodes()) == 2

        accum_nodes = find_nodes(fused, AccumulateOp)
        assert len(accum_nodes) == 1
        assert len(accum_nodes[0].comps) == 2

    def test_general_add_preserves_constant_operand(self):
        class AddScalar(nn.Module):
            def forward(self, x):
                return x + 1

        x = make_img_3ch_8x8()
        graph = torch_to_paiir(AddScalar(), x, strict=False)

        general_add = find_first(graph, GeneralAddOp)
        assert general_add.has_const_operands
        assert [operand.kind for operand in general_add.operands] == [
            AddOperandKind.TENSOR,
            AddOperandKind.CONST,
        ]
        assert [
            general_add.is_operand_broadcasted(operand)
            for operand in general_add.operands
        ] == [False, True]

        y_ref = AddScalar()(x)
        y_ir = graph.forward(x)
        assert torch.is_tensor(y_ir)
        assert torch.allclose(y_ir, y_ref)

    def test_general_add_specializes_to_potential_add(self):
        model = MultiInputMerge()
        unfused = torch_to_paiir(model, make_img_3ch_8x8(), make_img_3ch_8x8())
        assert find_nodes(unfused, GeneralAddOp)

        fused = fuse_to_offline_cores(specialize_general_adds(unfused))
        accum_nodes = find_nodes(fused, AccumulateOp)
        assert len(accum_nodes) == 1
        assert find_nodes(fused, GeneralAddOp) == []

    def test_sppf_block(self):
        """SPPF: cascaded MaxPool with shared module, cat merge."""
        model = SPPFBlock()
        unfused = torch_to_paiir(model, make_img_16ch_8x8())

        comp_nodes = find_nodes(unfused, StandaloneCompOp)
        pool_nodes = [n for n in comp_nodes if isinstance(n.comp, nn.MaxPool2d)]
        assert len(pool_nodes) >= 3

    def test_standalone_conv(self):
        """Conv with no activation stays StandaloneCompOp."""
        model = nn.Conv2d(3, 8, 3)
        fused = convert_and_fuse(model, make_img_3ch_8x8())

        comp_nodes = find_nodes(fused, StandaloneCompOp)
        assert len(comp_nodes) == 1
        assert isinstance(comp_nodes[0].comp, nn.Conv2d)

    def test_concat_op(self):
        """torch.cat creates ConcatOp with correct dim and port ordering."""

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 4, 1)
                self.conv2 = nn.Conv2d(3, 4, 1)
                self.conv3 = nn.Conv2d(8, 2, 1)
                self.relu = nn.ReLU()

            def forward(self, x):
                a = self.conv1(x)
                b = self.conv2(x)
                c = torch.cat([a, b], dim=1)
                return self.relu(self.conv3(c))

        model = M()
        unfused = torch_to_paiir(model, make_img_3ch_8x8())

        # Check ConcatOp exists
        concat_nodes = find_nodes(unfused, ConcatOp)
        assert len(concat_nodes) == 1
        assert concat_nodes[0].dim == 1

        # Check port ordering: conv1 -> port 0, conv2 -> port 1
        concat_name = find_node_names(unfused, ConcatOp)[0]
        preds = unfused.predecessors(concat_name)
        assert len(preds) == 2

        fused = fuse_to_offline_cores(specialize_general_adds(unfused))
        propagate_signal_semantics(fused)
        propagate_data_format(fused)
        assign_tick_params(fused)

        # Verify data format propagation through ConcatOp
        assert all(
            node.core_params.input_sign is not None
            for node in fused.nodes.values()
            if isinstance(node, OfflineCoreOp)
        )


class TestShapeAndDims:
    """Test shape and dims propagation."""

    def test_shape_propagation(self):
        """Shapes propagate correctly through a two-layer SNN."""
        model = SNNTwoLayer()
        fused = convert_and_fuse(model, make_img_3ch_8x8())

        seq_nodes = find_nodes(fused, SequentialOp)
        for node in seq_nodes:
            assert node.output_layouts[0].shape != ()
            assert all(layout.shape != () for layout in node.input_layouts)

    def test_no_sample_input(self):
        """Converter works without sample input, shapes default to ()."""
        model = SNNTwoLayer()
        unfused = torch_to_paiir(model)
        fused = fuse_to_offline_cores(specialize_general_adds(unfused))

        seq_nodes = find_nodes(fused, SequentialOp)
        assert len(seq_nodes) == 2
        for node in seq_nodes:
            assert node.num_outputs == 1
            assert node.output_layouts[0].shape == torch.Size()
            assert node.output_layouts[0].dims == ()

    def test_dims_identity(self):
        """Identity dims for standard conv/linear (no transpose)."""
        model = SNNTwoLayer()
        fused = convert_and_fuse(model, make_img_3ch_8x8())

        seq_nodes = find_nodes(fused, SequentialOp)
        for node in seq_nodes:
            assert node.output_layouts[0].dims == (0, 1, 2, 3)

    def test_flatten_dims(self):
        """flatten resets dims to identity."""
        model = SNNFlattenTransition()
        fused = convert_and_fuse(model, make_img_1ch_4x4())

        seq_nodes = find_nodes(fused, SequentialOp)
        assert len(seq_nodes) == 2
        for node in seq_nodes:
            assert node.output_layouts[0].dims in [(0, 1, 2, 3), (0, 1)]


class TestAssignTickParams:
    """Test assign_tick_params pass."""

    @pytest.fixture
    def fused_snn(self):
        """Fixture: fused SNNTwoLayer graph."""
        return convert_and_fuse(SNNTwoLayer(), make_img_3ch_8x8())

    def test_tick_start_from_depth(self, fused_snn):
        """tick_start is assigned based on DAG depth."""
        assign_tick_params(fused_snn)

        seq_nodes = find_nodes(fused_snn, SequentialOp)
        assert len(seq_nodes) == 2

        for node in seq_nodes:
            assert node.core_params.tick_start is not None
            assert node.core_params.tick_start >= 1

        tick_starts = sorted([n.core_params.tick_start for n in seq_nodes])  # type: ignore
        assert tick_starts[0] < tick_starts[1]

    def test_tick_start_explicit_override(self, fused_snn):
        """User-set tick_start is preserved by the pass."""
        first_op = find_first(fused_snn, SequentialOp)
        first_op.core_params.tick_start = 42

        assign_tick_params(fused_snn)
        assert first_op.core_params.tick_start == 42

    @pytest.mark.parametrize(
        "tick_duration, expected",
        [(0, 0), (100, 100)],
        ids=["default_always_working", "global_override"],
    )
    def test_tick_duration(self, fused_snn, tick_duration, expected):
        """tick_duration: default (0) or graph-level override applied to all nodes."""
        assign_tick_params(fused_snn, tick_duration=tick_duration)

        for node in fused_snn.nodes.values():
            if isinstance(node, SequentialOp):
                assert node.core_params.tick_duration == expected

    def test_tick_duration_per_node_override_via_core_params(self, fused_snn):
        """Per-node tick_duration set on core_params takes priority."""
        first_op = find_first(fused_snn, SequentialOp)
        first_op.core_params.tick_duration = 50

        assign_tick_params(fused_snn, tick_duration=100)
        assert first_op.core_params.tick_duration == 50

    @pytest.mark.parametrize(
        "tick_duration, auto_reset, expected_initial",
        [(100, True, 100), (0, True, 0), (100, False, 0)],
        ids=["reset_with_duration", "reset_always_working", "no_reset"],
    )
    def test_auto_reset(self, fused_snn, tick_duration, auto_reset, expected_initial):
        """auto_reset controls tick_initial derivation from tick_duration."""
        assign_tick_params(
            fused_snn, tick_duration=tick_duration, auto_reset=auto_reset
        )

        for node in fused_snn.nodes.values():
            if isinstance(node, SequentialOp):
                assert node.core_params.tick_initial == expected_initial

    def test_tick_override_tick_start(self, fused_snn):
        """overrides dict can set tick_start for a specific node."""
        seq_names = find_node_names(fused_snn, SequentialOp)
        assert len(seq_names) == 2

        assign_tick_params(fused_snn, overrides={seq_names[0]: {"tick_start": 10}})

        seq_nodes = find_nodes(fused_snn, SequentialOp)
        by_name = {n.name: n for n in seq_nodes}
        assert by_name[seq_names[0]].core_params.tick_start == 10
        assert by_name[seq_names[1]].core_params.tick_start is not None
        assert by_name[seq_names[1]].core_params.tick_start != 10

    def test_tick_override_duration_and_auto_reset(self, fused_snn):
        """overrides dict can set tick_duration and auto_reset per-node."""
        seq_names = find_node_names(fused_snn, SequentialOp)

        assign_tick_params(
            fused_snn,
            tick_duration=100,
            auto_reset=True,
            overrides={seq_names[0]: {"tick_duration": 200, "auto_reset": False}},
        )

        seq_nodes = find_nodes(fused_snn, SequentialOp)
        by_name = {n.name: n for n in seq_nodes}

        cp0 = by_name[seq_names[0]].core_params
        assert cp0.tick_duration == 200
        assert cp0.tick_initial == 0

        cp1 = by_name[seq_names[1]].core_params
        assert cp1.tick_duration == 100
        assert cp1.tick_initial == 100

    def test_residual_tick_start(self):
        """Residual (AccumulateOp) gets correct tick_start."""
        fused = convert_and_fuse(SNNResidualAdd(), make_img_3ch_8x8())
        assign_tick_params(fused)

        accum_ops = find_nodes(fused, AccumulateOp)
        assert len(accum_ops) == 1
        assert accum_ops[0].core_params.tick_start == 1

    @pytest.mark.parametrize(
        "kwargs, error_match",
        [
            ({"tick_duration": -1}, "tick_duration.*must be non-negative"),
            ({"tick_start": -5}, "tick_start.*non-negative"),
        ],
        ids=["negative_graph_duration", "negative_override_start"],
    )
    def test_negative_param_raises(self, fused_snn, kwargs, error_match):
        """Negative tick parameters raise ValueError immediately."""
        if "tick_duration" in kwargs:
            with pytest.raises(ValueError, match=error_match):
                assign_tick_params(fused_snn, tick_duration=kwargs["tick_duration"])
        else:
            seq_name = find_node_names(fused_snn, SequentialOp)[0]
            with pytest.raises(ValueError, match=error_match):
                assign_tick_params(fused_snn, overrides={seq_name: kwargs})

    def test_override_negative_tick_duration_raises(self, fused_snn):
        """Negative tick_duration in overrides raises immediately."""
        seq_name = find_node_names(fused_snn, SequentialOp)[0]
        with pytest.raises(ValueError, match="tick_duration.*non-negative"):
            assign_tick_params(fused_snn, overrides={seq_name: {"tick_duration": -10}})

    def test_override_unknown_node_raises(self):
        """Override key for non-existent node raises KeyError."""
        fused = convert_and_fuse(SNNTwoLayer(), make_img_3ch_8x8())
        with pytest.raises(KeyError, match="does not match any node"):
            assign_tick_params(fused, overrides={"nonexistent_node": {"tick_start": 1}})

    def test_validate_tick_params_unassigned(self):
        """validate_tick_params raises if tick_start is still None."""
        cp = OfflineCoreParams()
        assert cp.tick_start is None
        with pytest.raises(ValueError, match="tick_start is None"):
            cp.validate_tick_params()

    @pytest.mark.parametrize(
        "params, error_match",
        [
            ({"tick_start": -1}, "tick_start"),
            ({"tick_start": 1, "tick_duration": -1}, "tick_duration"),
            ({"tick_start": 1, "tick_initial": -1}, "tick_initial"),
        ],
        ids=["negative_start", "negative_duration", "negative_initial"],
    )
    def test_validate_tick_params_out_of_range(self, params, error_match):
        """validate_tick_params raises on out-of-range values."""
        cp = OfflineCoreParams(**params)
        with pytest.raises(ValueError, match=error_match):
            cp.validate_tick_params()


class TestValidGraphs:
    """Well-formed graphs should pass validation without errors."""

    def test_simple_snn(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3, padding=1)
                self.ifn = neuron.IFNode()

            def forward(self, x):
                return self.ifn(self.conv(x))

        unfused = torch_to_paiir(Model(), torch.randn(1, 3, 8, 8))
        fused = fuse_to_offline_cores(specialize_general_adds(unfused))
        validate_graph(fused)

    def test_two_layer_ann(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3, padding=1)
                self.relu = nn.ReLU()
                self.linear = nn.Linear(16 * 8 * 8, 10)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = self.relu(self.conv(x))
                x = x.flatten(1)
                return self.sigmoid(self.linear(x))

        unfused = torch_to_paiir(Model(), torch.randn(1, 3, 8, 8))
        fused = fuse_to_offline_cores(specialize_general_adds(unfused))
        validate_graph(fused)

    def test_residual_snn(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_a = nn.Conv2d(3, 16, 3, padding=1)
                self.conv_b = nn.Conv2d(3, 16, 3, padding=1)
                self.lif = neuron.LIFNode(tau=2.0)

            def forward(self, x):
                return self.lif(self.conv_a(x) + self.conv_b(x))

        unfused = torch_to_paiir(Model(), torch.randn(1, 3, 8, 8))
        fused = fuse_to_offline_cores(specialize_general_adds(unfused))
        validate_graph(fused)


class TestUnrecoverableErrors:
    def test_no_input_node(self):
        graph = PAIIRGraph("no_input")
        op = SequentialOp(nn.Linear(8, 4), IFNodeV25())
        _set_single_layouts(op, (1, 8), (1, 4), (0, 1), (0, 1))
        out = OutputNode()
        graph.add_node(op)
        graph.add_node(out)
        graph.add_edge(op.name, out.name)

        with pytest.raises(GraphValidationError, match="no input node"):
            validate_graph(graph)

    def test_no_output_node(self):
        graph = PAIIRGraph("no_output")
        inp = InputNode(shape=torch.Size((1, 8)))
        op = SequentialOp(nn.Linear(8, 4), IFNodeV25())
        _set_single_layouts(op, (1, 8), (1, 4), (0, 1), (0, 1))
        graph.add_node(inp)
        graph.add_node(op)
        graph.add_edge(inp.name, op.name)

        with pytest.raises(GraphValidationError, match="no output node"):
            validate_graph(graph)

    def test_input_has_predecessors(self):
        """InputNode with predecessors is a structural error."""
        graph = PAIIRGraph("bad_input")
        inp = InputNode(shape=torch.Size((1, 8)))
        op = SequentialOp(nn.Linear(8, 4), IFNodeV25())
        _set_single_layouts(op, (1, 8), (1, 4), (0, 1), (0, 1))
        out = OutputNode()
        graph.add_node(inp)
        graph.add_node(op)
        graph.add_node(out)
        graph.add_edge(inp.name, op.name)
        graph.add_edge(op.name, out.name)
        # Manually add an invalid edge pointing into the InputNode
        graph.add_edge(op.name, inp.name)

        with pytest.raises(GraphValidationError, match="has predecessors"):
            validate_graph(graph)

    def test_output_has_successors(self):
        """OutputNode with successors is a structural error."""
        graph = PAIIRGraph("bad_output")
        inp = InputNode(shape=torch.Size((1, 8)))
        op = SequentialOp(nn.Linear(8, 4), IFNodeV25())
        _set_single_layouts(op, (1, 8), (1, 4), (0, 1), (0, 1))
        out = OutputNode()
        graph.add_node(inp)
        graph.add_node(op)
        graph.add_node(out)
        graph.add_edge(inp.name, op.name)
        graph.add_edge(op.name, out.name)
        # Manually add an invalid edge going out of the OutputNode
        graph.add_edge(out.name, op.name)

        with pytest.raises(GraphValidationError, match="has successors"):
            validate_graph(graph)

    def test_collects_all_errors(self):
        """Multiple unrecoverable errors reported at once."""
        graph = PAIIRGraph("multiple_errors")
        # No InputNode, no OutputNode
        op = SequentialOp(nn.Linear(8, 4), IFNodeV25())
        _set_single_layouts(op, (1, 8), (1, 4), (0, 1), (0, 1))
        graph.add_node(op)

        with pytest.raises(GraphValidationError) as exc_info:
            validate_graph(graph)

        err = exc_info.value
        assert len(err.errors) >= 2  # no input + no output

    def test_general_add_is_valid_expression_graph(self):
        class AddScalar(nn.Module):
            def forward(self, x):
                return x + 1

        graph = torch_to_paiir(AddScalar(), torch.randn(1, 3, 8, 8), strict=False)
        assert any(isinstance(node, GeneralAddOp) for node in graph.nodes.values())
        validate_graph(graph)

    def test_compile_rejects_unspecialized_general_add(self):
        class AddScalar(nn.Module):
            def forward(self, x):
                return x + 1

        with pytest.raises(GraphValidationError, match="GeneralAddOp"):
            compile_to_paiir(AddScalar(), torch.randn(1, 3, 8, 8), strict=False)

    def test_compile_rejects_broadcast_general_add(self):
        class AddBroadcast(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.ones(1, 3, 1, 1))

            def forward(self, x):
                return x + self.bias

        with pytest.raises(GraphValidationError, match="GeneralAddOp"):
            compile_to_paiir(AddBroadcast(), torch.randn(1, 3, 8, 8), strict=False)


class TestAutoCleanup:
    def test_orphan_op_removed_with_warning(self):
        """Orphan OpNode is removed and a warning is emitted."""
        graph = PAIIRGraph("orphan_op")
        inp = InputNode(shape=torch.Size((1, 8)))
        op1 = SequentialOp(nn.Linear(8, 4), IFNodeV25())
        _set_single_layouts(op1, (1, 8), (1, 4), (0, 1), (0, 1))
        op2 = SequentialOp(nn.Linear(8, 4), IFNodeV25())  # orphan
        _set_single_layouts(op2, (1, 8), (1, 4), (0, 1), (0, 1))
        out = OutputNode()
        graph.add_node(inp)
        graph.add_node(op1)
        graph.add_node(op2)
        graph.add_node(out)
        graph.add_edge(inp.name, op1.name)
        graph.add_edge(op1.name, out.name)

        with pytest.warns(GraphCleanupWarning, match="disconnected"):
            validate_graph(graph)

        assert op2.name not in graph.nodes
        assert (
            len([n for n in graph.nodes.values() if isinstance(n, SequentialOp)]) == 1
        )

    def test_dead_end_op_removed(self):
        """OpNode with input but no output is removed."""
        graph = PAIIRGraph("dead_end")
        inp = InputNode(shape=torch.Size((1, 8)))
        op1 = SequentialOp(nn.Linear(8, 4), IFNodeV25())
        _set_single_layouts(op1, (1, 8), (1, 4), (0, 1), (0, 1))
        op2 = SequentialOp(nn.Linear(4, 2), IFNodeV25())  # dead end
        _set_single_layouts(op2, (1, 4), (1, 2), (0, 1), (0, 1))
        out = OutputNode()
        graph.add_node(inp)
        graph.add_node(op1)
        graph.add_node(op2)
        graph.add_node(out)
        graph.add_edge(inp.name, op1.name)
        graph.add_edge(op1.name, op2.name)
        graph.add_edge(op1.name, out.name)

        with pytest.warns(GraphCleanupWarning):
            validate_graph(graph)

        assert op2.name not in graph.nodes
        # Edge from op1 -> op2 should also be removed
        assert all(e.dst != op2.name for e in graph.edges)

    def test_disconnected_input_removed(self):
        """Disconnected InputNode is removed with warning."""
        graph = PAIIRGraph("disconnected_input")
        inp1 = InputNode(shape=torch.Size((1, 8)))
        inp2 = InputNode(shape=torch.Size((1, 8)))  # disconnected
        op = SequentialOp(nn.Linear(8, 4), IFNodeV25())
        _set_single_layouts(op, (1, 8), (1, 4), (0, 1), (0, 1))
        out = OutputNode()
        graph.add_node(inp1)
        graph.add_node(inp2)
        graph.add_node(op)
        graph.add_node(out)
        graph.add_edge(inp1.name, op.name)
        graph.add_edge(op.name, out.name)

        with pytest.warns(GraphCleanupWarning):
            validate_graph(graph)

        assert inp2.name not in graph.nodes
        assert len(graph.input_nodes()) == 1

    def test_disconnected_output_removed(self):
        """Disconnected OutputNode is removed with warning."""
        graph = PAIIRGraph("disconnected_output")
        inp = InputNode(shape=torch.Size((1, 8)))
        op = SequentialOp(nn.Linear(8, 4), IFNodeV25())
        _set_single_layouts(op, (1, 8), (1, 4), (0, 1), (0, 1))
        out1 = OutputNode()
        out2 = OutputNode()  # disconnected
        graph.add_node(inp)
        graph.add_node(op)
        graph.add_node(out1)
        graph.add_node(out2)
        graph.add_edge(inp.name, op.name)
        graph.add_edge(op.name, out1.name)

        with pytest.warns(GraphCleanupWarning):
            validate_graph(graph)

        assert out2.name not in graph.nodes
        assert len(graph.output_nodes()) == 1

    def test_all_inputs_disconnected_raises_after_cleanup(self):
        """If all InputNodes are disconnected, cleanup leaves no inputs -> error."""
        graph = PAIIRGraph("all_disconnected")
        inp = InputNode(shape=torch.Size((1, 8)))  # disconnected
        op = SequentialOp(nn.Linear(8, 4), IFNodeV25())
        _set_single_layouts(op, (1, 8), (1, 4), (0, 1), (0, 1))
        out = OutputNode()
        graph.add_node(inp)
        graph.add_node(op)
        graph.add_node(out)
        graph.add_edge(op.name, out.name)
        # inp has no edges, op has no predecessors -> both will be removed

        # First the orphan nodes are removed, then post-cleanup check fails
        with pytest.raises(GraphValidationError, match="no input node after cleanup"):
            with pytest.warns(GraphCleanupWarning):
                validate_graph(graph)


class TestSNNModeLUTConsistency:
    """Validate that snn_mode matches LUT presence on OfflineCoreOp nodes."""

    def _build_graph(self, act: CoreNeuronV25) -> PAIIRGraph:
        """Build a minimal valid graph with a single SequentialOp."""
        graph = PAIIRGraph("test")
        inp = InputNode(shape=torch.Size((1, 8)))
        op = SequentialOp(nn.Linear(8, 4), act)
        _set_single_layouts(op, (1, 8), (1, 4), (0, 1), (0, 1))
        out = OutputNode()
        graph.add_node(inp)
        graph.add_node(op)
        graph.add_node(out)
        graph.add_edge(inp.name, op.name)
        graph.add_edge(op.name, out.name)
        return graph

    def test_snn_mode_correct(self):
        """SNN neuron with snn_mode=SNN passes validation."""
        graph = self._build_graph(IFNodeV25())
        validate_graph(graph)

    def test_ann_mode_correct(self):
        """ANN neuron (LUT) with snn_mode=ANN passes validation."""
        graph = self._build_graph(ANNNodeV25(lut=LutReLU()))
        validate_graph(graph)

    def test_lut_with_snn_mode_raises(self):
        """LUT activation but snn_mode=SNN raises GraphValidationError."""
        act = ANNNodeV25(lut=LutReLU())
        graph = self._build_graph(act)

        # Force snn_mode mismatch
        op = next(n for n in graph.nodes.values() if isinstance(n, SequentialOp))
        op.core_params.snn_mode = SNNMode.SNN

        with pytest.raises(GraphValidationError, match="LUT activation.*SNNMode.ANN"):
            validate_graph(graph)

    def test_no_lut_with_ann_mode_raises(self):
        """No LUT but snn_mode=ANN raises GraphValidationError."""
        act = IFNodeV25()
        graph = self._build_graph(act)

        # Force snn_mode mismatch
        op = next(n for n in graph.nodes.values() if isinstance(n, SequentialOp))
        op.core_params.snn_mode = SNNMode.ANN

        with pytest.raises(
            GraphValidationError, match="no LUT activation.*SNNMode.SNN"
        ):
            validate_graph(graph)


class TestValidateCompiledGraph:
    def _build_compiled_graph(self) -> tuple[PAIIRGraph, SequentialOp]:
        graph = PAIIRGraph("compiled")
        inp = InputNode(shape=torch.Size((1, 8)))
        op = SequentialOp(nn.Linear(8, 4), IFNodeV25())
        _set_single_layouts(op, (1, 8), (1, 4), (0, 1), (0, 1))
        op.core_params.set_input_format((DataSign.SIGNED, DataWidth.WIDTH_8BIT))
        op.core_params.set_output_format((DataSign.UNSIGNED, DataWidth.WIDTH_1BIT))
        op.core_params.set_weight_format((DataSign.SIGNED, DataWidth.WIDTH_8BIT))
        op.core_params.tick_start = 1
        op.core_params.tick_duration = 0
        op.core_params.tick_initial = 0
        out = OutputNode()
        inp.signal_semantics.output_domain = SignalDomain.VALUE
        op.signal_semantics.output_domain = SignalDomain.VALUE
        out.signal_semantics.output_domain = SignalDomain.VALUE
        graph.add_node(inp)
        graph.add_node(op)
        graph.add_node(out)
        graph.add_edge(inp.name, op.name)
        graph.add_edge(op.name, out.name)
        return graph, op

    def test_valid_compiled_graph_passes(self):
        graph, _ = self._build_compiled_graph()
        validate_compiled_graph(graph)

    def test_missing_data_format_assignment_raises(self):
        graph, op = self._build_compiled_graph()
        op.core_params._input_format_assigned = False

        with pytest.raises(
            GraphValidationError, match="missing propagated data format"
        ):
            validate_compiled_graph(graph)

    def test_catches_nodes_not_on_any_input_to_output_path(self):
        graph, _ = self._build_compiled_graph()
        branch = SequentialOp(nn.Linear(8, 4), IFNodeV25())
        dead_end = SequentialOp(nn.Linear(4, 2), IFNodeV25())

        _set_single_layouts(branch, (1, 8), (1, 4), (0, 1), (0, 1))
        _set_single_layouts(dead_end, (1, 4), (1, 2), (0, 1), (0, 1))

        for node in (branch, dead_end):
            node.core_params.set_input_format((DataSign.SIGNED, DataWidth.WIDTH_8BIT))
            node.core_params.set_output_format(
                (DataSign.UNSIGNED, DataWidth.WIDTH_1BIT)
            )
            node.core_params.set_weight_format((DataSign.SIGNED, DataWidth.WIDTH_8BIT))
            node.core_params.tick_start = 1
            node.core_params.tick_duration = 0
            node.core_params.tick_initial = 0
            node.signal_semantics.output_domain = SignalDomain.VALUE

        inp = graph.input_nodes()[0]
        graph.add_node(branch)
        graph.add_node(dead_end)
        graph.add_edge(inp.name, branch.name)
        graph.add_edge(branch.name, dead_end.name)

        with pytest.raises(
            GraphValidationError, match="nodes not on any input-to-output path"
        ):
            validate_compiled_graph(graph)

    def test_rejects_concat_predecessor_shape_mismatch(self):
        graph = PAIIRGraph("bad_concat_shapes")
        inp_a = InputNode(shape=torch.Size((1, 2, 4, 4)))
        inp_b = InputNode(shape=torch.Size((1, 2, 4)))
        cat = ConcatOp(dim=1)
        out = OutputNode()

        inp_a.signal_semantics.output_domain = SignalDomain.VALUE
        inp_b.signal_semantics.output_domain = SignalDomain.VALUE
        cat.signal_semantics.output_domain = SignalDomain.VALUE
        out.signal_semantics.output_domain = SignalDomain.VALUE

        _set_multi_input_single_output_layouts(
            cat,
            [(1, 32), (1, 8)],
            (1, 40),
            [(0, 1), (0, 1)],
            (0, 1),
        )

        graph.add_node(inp_a)
        graph.add_node(inp_b)
        graph.add_node(cat)
        graph.add_node(out)
        graph.add_edge(inp_a.name, cat.name, dst_port=0)
        graph.add_edge(inp_b.name, cat.name, dst_port=1)
        graph.add_edge(cat.name, out.name)

        with pytest.raises(
            GraphValidationError, match="ConcatOp .*predecessor shape mismatch"
        ):
            validate_compiled_graph(graph)

    def test_rejects_split_output_shape_mismatch(self):
        graph = PAIIRGraph("bad_split_spec")
        inp = InputNode(shape=torch.Size((1, 5)))
        split = SplitOp(sections=(2, 4), dim=1)
        out = OutputNode()

        inp.signal_semantics.output_domain = SignalDomain.VALUE
        split.signal_semantics.output_domain = SignalDomain.VALUE
        out.signal_semantics.output_domain = SignalDomain.VALUE

        split.input_layouts = (_layout((1, 5), (0, 1)),)

        graph.add_node(inp)
        graph.add_node(split)
        graph.add_node(out)
        graph.add_edge(inp.name, split.name)
        graph.add_edge(split.name, out.name, src_port=1)

        with pytest.raises(GraphValidationError, match="invalid split spec"):
            validate_compiled_graph(graph)

    def test_rejects_standalone_act_32bit_input_without_direct_add(self):
        graph = PAIIRGraph("bad_standalone_act_direct_add")
        inp = InputNode(shape=torch.Size((1, 4)))
        comp = StandaloneCompOp(nn.Linear(4, 4, bias=False))
        act = StandaloneActOp(ANNNodeV25(lut=LutReLU()))
        out = OutputNode(shape=torch.Size((1, 4)))

        _set_single_layouts(comp, (1, 4), (1, 4), (0, 1), (0, 1))
        _set_single_layouts(act, (1, 4), (1, 4), (0, 1), (0, 1))

        for node in (comp, act):
            node.core_params.tick_start = 1
            node.core_params.tick_duration = 0
            node.core_params.tick_initial = 0

        comp.core_params.set_input_format((DataSign.SIGNED, DataWidth.WIDTH_8BIT))
        comp.core_params.set_output_format((DataSign.SIGNED, DataWidth.WIDTH_32BIT))
        comp.core_params.set_weight_format((DataSign.SIGNED, DataWidth.WIDTH_8BIT))
        act.core_params.set_input_format((DataSign.SIGNED, DataWidth.WIDTH_32BIT))
        act.core_params.set_output_format((DataSign.UNSIGNED, DataWidth.WIDTH_8BIT))
        act.core_params.set_weight_format((DataSign.UNSIGNED, DataWidth.WIDTH_1BIT))

        inp.signal_semantics.output_domain = SignalDomain.VALUE
        comp.signal_semantics.output_domain = SignalDomain.POTENTIAL
        act.signal_semantics.output_domain = SignalDomain.VALUE
        out.signal_semantics.output_domain = SignalDomain.VALUE

        graph.add_node(inp)
        graph.add_node(comp)
        graph.add_node(act)
        graph.add_node(out)
        graph.add_edge(inp.name, comp.name)
        graph.add_edge(comp.name, act.name)
        graph.add_edge(act.name, out.name)

        with pytest.raises(
            GraphValidationError,
            match="receives WIDTH_32BIT input but add_potential is not AddPotentialMode.DIRECT_ADD",
        ):
            validate_compiled_graph(graph)


class TestSplitPassBehavior:
    def _build_split_routing_graph(
        self,
    ) -> tuple[PAIIRGraph, InputNode, SplitOp, StandaloneActOp]:
        graph = PAIIRGraph("split_routing")
        inp = InputNode(shape=torch.Size((1, 4)))
        split = SplitOp(sections=2, dim=1)
        act = StandaloneActOp(IFNodeV25())
        out = OutputNode(shape=torch.Size((1, 2)))

        _set_split_layouts(split, (1, 4), [(1, 2), (1, 2)], (0, 1))
        _set_single_layouts(act, (1, 2), (1, 2), (0, 1), (0, 1))

        graph.add_node(inp)
        graph.add_node(split)
        graph.add_node(act)
        graph.add_node(out)
        graph.add_edge(inp.name, split.name)
        graph.add_edge(split.name, act.name, src_port=0)
        graph.add_edge(act.name, out.name)
        return graph, inp, split, act

    def test_signal_domain_and_data_format_propagate_through_split(self):
        graph, inp, split, act = self._build_split_routing_graph()

        propagate_signal_semantics(
            graph,
            input_formats={inp.name: (DataSign.UNSIGNED, DataWidth.WIDTH_1BIT)},
        )
        propagate_data_format(
            graph, input_formats={inp.name: (DataSign.UNSIGNED, DataWidth.WIDTH_1BIT)}
        )

        assert split.signal_semantics.output_domain is SignalDomain.VALUE
        assert act.signal_semantics.output_domain is SignalDomain.VALUE
        assert act.core_params.input_sign == DataSign.UNSIGNED
        assert act.core_params.input_width == DataWidth.WIDTH_1BIT

    def test_assign_tick_params_treats_split_as_zero_depth(self):
        graph, _, _, act = self._build_split_routing_graph()

        assign_tick_params(graph)

        assert act.core_params.tick_start == 1

    def test_validate_split_rejects_out_of_range_src_port(self):
        graph, _, split, _ = self._build_split_routing_graph()
        graph.edges = [
            (
                edge
                if edge.src != split.name
                else Edge(
                    src=edge.src, dst=edge.dst, src_port=2, dst_port=edge.dst_port
                )
            )
            for edge in graph.edges
        ]

        with pytest.raises(GraphValidationError, match="src_port=2"):
            validate_graph(graph)


class TestValidateDeployableGraph:
    def test_rejects_non_backend_ready_node_type(self):
        graph = PAIIRGraph("non_backend_ready")
        inp = InputNode(shape=torch.Size((1, 8)))
        cpu = CPUOp()
        out = OutputNode()

        inp.signal_semantics.output_domain = SignalDomain.VALUE
        cpu.signal_semantics.output_domain = SignalDomain.VALUE
        out.signal_semantics.output_domain = SignalDomain.VALUE

        graph.add_node(inp)
        graph.add_node(cpu)
        graph.add_node(out)
        graph.add_edge(inp.name, cpu.name)
        graph.add_edge(cpu.name, out.name)

        with pytest.raises(GraphValidationError, match="CPUOp"):
            validate_deployable_graph(graph)

    def test_rejects_frontend_only_split_op(self):
        graph = PAIIRGraph("frontend_split")
        inp = InputNode(shape=torch.Size((1, 4)))
        split = SplitOp(sections=2, dim=1)
        out = OutputNode()

        inp.signal_semantics.output_domain = SignalDomain.VALUE
        split.signal_semantics.output_domain = SignalDomain.VALUE
        out.signal_semantics.output_domain = SignalDomain.VALUE

        graph.add_node(inp)
        graph.add_node(split)
        graph.add_node(out)
        graph.add_edge(inp.name, split.name)
        graph.add_edge(split.name, out.name, src_port=0)

        with pytest.raises(GraphValidationError, match="frontend-only IR"):
            validate_deployable_graph(graph)

    def test_rechecks_potential_add_predecessor_domains(self):
        graph = PAIIRGraph("bad_potential_add_domain")
        inp_a = InputNode(shape=torch.Size((1, 4)))
        inp_b = InputNode(shape=torch.Size((1, 4)))
        add = PotentialAddOp(op_signs=(1, 1))
        out = OutputNode()

        inp_a.signal_semantics.output_domain = SignalDomain.VALUE
        inp_b.signal_semantics.output_domain = SignalDomain.VALUE
        add.signal_semantics.output_domain = SignalDomain.POTENTIAL
        out.signal_semantics.output_domain = SignalDomain.POTENTIAL

        graph.add_node(inp_a)
        graph.add_node(inp_b)
        graph.add_node(add)
        graph.add_node(out)
        graph.add_edge(inp_a.name, add.name, dst_port=0)
        graph.add_edge(inp_b.name, add.name, dst_port=1)
        graph.add_edge(add.name, out.name)

        with pytest.raises(GraphValidationError, match="PotentialAddOp"):
            validate_deployable_graph(graph)

    def test_rechecks_concat_predecessor_domains(self):
        graph = PAIIRGraph("bad_concat_domain")
        inp_a = InputNode(shape=torch.Size((1, 4)))
        inp_b = InputNode(shape=torch.Size((1, 4)))
        cat = ConcatOp(dim=1)
        out = OutputNode()

        inp_a.signal_semantics.output_domain = SignalDomain.VALUE
        inp_b.signal_semantics.output_domain = SignalDomain.POTENTIAL
        cat.signal_semantics.output_domain = SignalDomain.VALUE
        out.signal_semantics.output_domain = SignalDomain.VALUE

        graph.add_node(inp_a)
        graph.add_node(inp_b)
        graph.add_node(cat)
        graph.add_node(out)
        graph.add_edge(inp_a.name, cat.name, dst_port=0)
        graph.add_edge(inp_b.name, cat.name, dst_port=1)
        graph.add_edge(cat.name, out.name)

        with pytest.raises(GraphValidationError, match="ConcatOp"):
            validate_deployable_graph(graph)

    def test_rechecks_accumulate_path_counts(self):
        graph = PAIIRGraph("bad_accumulate_counts")
        inp_a = InputNode(shape=torch.Size((1, 4)))
        inp_b = InputNode(shape=torch.Size((1, 4)))
        acc = AccumulateOp(
            comps=[nn.Linear(4, 4), nn.Linear(4, 4)],
            act=IFNodeV25(),
            op_signs=(1, 1),
        )
        out = OutputNode()

        inp_a.signal_semantics.output_domain = SignalDomain.VALUE
        inp_b.signal_semantics.output_domain = SignalDomain.VALUE
        acc.signal_semantics.output_domain = SignalDomain.VALUE
        out.signal_semantics.output_domain = SignalDomain.VALUE
        _set_single_layouts(acc, (1, 4), (1, 4), (0, 1), (0, 1))

        graph.add_node(inp_a)
        graph.add_node(inp_b)
        graph.add_node(acc)
        graph.add_node(out)
        graph.add_edge(inp_a.name, acc.name, dst_port=0)
        graph.add_edge(inp_b.name, acc.name, dst_port=1)
        graph.add_edge(acc.name, out.name)

        with pytest.raises(GraphValidationError, match="AccumulateOp"):
            validate_deployable_graph(graph)


class AddScalarDomain(nn.Module):
    def forward(self, x):
        return x + 1


class ValueBranchAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_a = nn.Conv2d(3, 4, 1)
        self.conv_b = nn.Conv2d(3, 4, 1)
        self.if_a = neuron.IFNode(v_threshold=1.0)
        self.if_b = neuron.IFNode(v_threshold=1.0)

    def forward(self, x):
        return self.if_a(self.conv_a(x)) + self.if_b(self.conv_b(x))


class TestSignalSemantics:
    def test_input_node_sets_known_code_range_from_effective_input_format(self):
        graph = PAIIRGraph("input_semantics")
        inp = InputNode(shape=torch.Size((1, 4)))
        out = OutputNode(shape=torch.Size((1, 4)))
        graph.add_node(inp)
        graph.add_node(out)
        graph.add_edge(inp.name, out.name)

        propagate_signal_semantics(
            graph,
            input_formats={inp.name: (DataSign.SIGNED, DataWidth.WIDTH_4BIT)},
        )

        assert inp.signal_semantics.output_domain is SignalDomain.VALUE
        assert inp.signal_semantics.known_code_range == (-8, 7)
        assert out.signal_semantics.output_domain is SignalDomain.VALUE
        assert out.signal_semantics.known_code_range == (-8, 7)

    def test_input_node_default_known_code_range_uses_fixed_signed_8bit(self):
        graph = PAIIRGraph("input_semantics_default")
        inp = InputNode(shape=torch.Size((1, 4)))
        out = OutputNode(shape=torch.Size((1, 4)))
        graph.add_node(inp)
        graph.add_node(out)
        graph.add_edge(inp.name, out.name)

        propagate_signal_semantics(graph)

        assert inp.signal_semantics.output_domain is SignalDomain.VALUE
        assert inp.signal_semantics.known_code_range == (-128, 127)
        assert out.signal_semantics.output_domain is SignalDomain.VALUE
        assert out.signal_semantics.known_code_range == (-128, 127)

    def test_concat_known_code_range_requires_all_predecessors_known(self):
        graph = PAIIRGraph("concat_known_code_range")
        inp = InputNode(shape=torch.Size((1, 4)))
        add = GeneralAddOp(
            operands=(
                AddOperandSpec(1, AddOperandKind.TENSOR, tensor_port=0),
                AddOperandSpec(1, AddOperandKind.CONST, const_value=1),
            )
        )
        cat = ConcatOp(dim=1)
        out = OutputNode(shape=torch.Size((1, 8)))
        for node in (inp, add, cat, out):
            graph.add_node(node)
        graph.add_edge(inp.name, add.name)
        graph.add_edge(inp.name, cat.name, dst_port=0)
        graph.add_edge(add.name, cat.name, dst_port=1)
        graph.add_edge(cat.name, out.name)

        propagate_signal_semantics(
            graph,
            input_formats={inp.name: (DataSign.UNSIGNED, DataWidth.WIDTH_1BIT)},
        )

        assert inp.signal_semantics.known_code_range == (0, 1)
        assert add.signal_semantics.output_domain is SignalDomain.VALUE
        assert add.signal_semantics.known_code_range is None
        assert cat.signal_semantics.output_domain is SignalDomain.VALUE
        assert cat.signal_semantics.known_code_range is None
        assert out.signal_semantics.known_code_range is None

    def test_standalone_maxpool_preserves_value_domain(self):
        class ValueMaxPool(nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = nn.ReLU()
                self.pool = nn.MaxPool2d(2)

            def forward(self, x):
                return self.pool(self.relu(x))

        unfused = torch_to_paiir(ValueMaxPool(), torch.randn(1, 3, 8, 8))
        fused = fuse_to_offline_cores(specialize_general_adds(unfused))

        propagate_signal_semantics(fused)

        pool = next(
            node
            for node in fused.nodes.values()
            if isinstance(node, StandaloneCompOp)
            and isinstance(node.comp, nn.MaxPool2d)
        )
        out = fused.output_nodes()[0]
        assert pool.signal_semantics.output_domain == SignalDomain.VALUE
        assert pool.signal_semantics.known_code_range == (0, 254)
        assert pool.neuron_params.output_type == OutputType.VALUE
        assert out.signal_semantics.output_domain == SignalDomain.VALUE
        assert out.signal_semantics.known_code_range == (0, 254)

    def test_standalone_maxpool_rejects_potential_domain(self):
        class PotentialMaxPool(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 4, 1)
                self.pool = nn.MaxPool2d(2)

            def forward(self, x):
                return self.pool(self.conv(x))

        unfused = torch_to_paiir(PotentialMaxPool(), torch.randn(1, 3, 8, 8))
        fused = fuse_to_offline_cores(specialize_general_adds(unfused))

        with pytest.raises(GraphValidationError, match="Standalone MaxPool"):
            propagate_signal_semantics(fused)

    def test_general_add_from_scalar_propagates_value_domain(self):
        graph = torch_to_paiir(AddScalarDomain(), torch.randn(1, 3, 8, 8), strict=False)

        propagate_signal_semantics(
            graph,
            input_formats={
                graph.input_nodes()[0].name: (DataSign.UNSIGNED, DataWidth.WIDTH_1BIT)
            },
        )

        add = find_first(graph, GeneralAddOp)
        out = graph.output_nodes()[0]
        assert (
            graph.input_nodes()[0].signal_semantics.output_domain == SignalDomain.VALUE
        )
        assert graph.input_nodes()[0].signal_semantics.known_code_range == (0, 1)
        assert add.signal_semantics.output_domain == SignalDomain.VALUE
        assert add.signal_semantics.known_code_range is None
        assert out.signal_semantics.output_domain == SignalDomain.VALUE
        assert out.signal_semantics.known_code_range is None

    def test_potential_add_from_residual_comp_propagates_potential_domain(self):
        class MembraneAdd(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_a = nn.Conv2d(3, 4, 1)
                self.conv_b = nn.Conv2d(3, 4, 1)

            def forward(self, x):
                return self.conv_a(x) + self.conv_b(x)

        unfused = torch_to_paiir(MembraneAdd(), torch.randn(1, 3, 8, 8))
        fused = fuse_to_offline_cores(specialize_general_adds(unfused))

        propagate_signal_semantics(fused)

        add = find_first(fused, PotentialAddOp)
        out = fused.output_nodes()[0]
        assert add.signal_semantics.output_domain == SignalDomain.POTENTIAL
        assert add.signal_semantics.known_code_range is None
        assert out.signal_semantics.output_domain == SignalDomain.POTENTIAL
        assert out.signal_semantics.known_code_range is None

    def test_accumulate_with_activation_propagates_value_domain(self):
        unfused = torch_to_paiir(SNNResidualAdd(), make_img_3ch_8x8())
        fused = fuse_to_offline_cores(specialize_general_adds(unfused))

        propagate_signal_semantics(fused)

        accum = find_first(fused, AccumulateOp)
        out = fused.output_nodes()[0]
        assert accum.signal_semantics.output_domain == SignalDomain.VALUE
        assert accum.signal_semantics.known_code_range == (0, 1)
        assert out.signal_semantics.output_domain == SignalDomain.VALUE
        assert out.signal_semantics.known_code_range == (0, 1)

    def test_potential_add_rejects_value_domain_inputs(self):
        unfused = torch_to_paiir(ValueBranchAdd(), torch.randn(1, 3, 8, 8))
        fused = fuse_to_offline_cores(specialize_general_adds(unfused))

        with pytest.raises(GraphValidationError, match="PotentialAddOp"):
            propagate_signal_semantics(fused)

    def test_rechecks_accumulate_signs(self):
        graph = PAIIRGraph("bad_accumulate_signs")
        inp_a = InputNode(shape=torch.Size((1, 4)))
        inp_b = InputNode(shape=torch.Size((1, 4)))
        acc = AccumulateOp(
            comps=[nn.Linear(4, 4), nn.Linear(4, 4)], act=IFNodeV25(), op_signs=(1, 1)
        )
        out = OutputNode()

        inp_a.signal_semantics.output_domain = SignalDomain.VALUE
        inp_b.signal_semantics.output_domain = SignalDomain.VALUE
        acc.signal_semantics.output_domain = SignalDomain.VALUE
        out.signal_semantics.output_domain = SignalDomain.VALUE
        _set_multi_input_single_output_layouts(
            acc, [(1, 4), (1, 4)], (1, 4), [(0, 1), (0, 1)], (0, 1)
        )
        acc.signs = (1, 0)

        graph.add_node(inp_a)
        graph.add_node(inp_b)
        graph.add_node(acc)
        graph.add_node(out)
        graph.add_edge(inp_a.name, acc.name, dst_port=0)
        graph.add_edge(inp_b.name, acc.name, dst_port=1)
        graph.add_edge(acc.name, out.name)

        with pytest.raises(GraphValidationError, match="AccumulateOp"):
            validate_deployable_graph(graph)
