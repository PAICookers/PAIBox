import operator

import torch
from torch import nn

from paibox.paiir.lowering.shape_analysis import analyze_shape_helpers
from tests.paiir.tracing import trace_for_lowering as _trace_for_shape_analysis


class TestShapeAnalysis:
    def test_flatten_module_is_recorded_as_reshape_sink(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.flatten = nn.Flatten(1)

            def forward(self, x):
                return self.flatten(x)

        gm = _trace_for_shape_analysis(M(), torch.randn(1, 3, 4, 5))
        analysis = analyze_shape_helpers(gm)

        flatten_node = next(node for node in gm.graph.nodes if node.op == "call_module")
        sink = analysis.sink_for(flatten_node)
        assert sink is not None

        assert sink.kind == "flatten"
        assert sink.data_input.op == "placeholder"
        assert sink.start_dim == 1
        assert sink.end_dim == -1
        assert sink.shape_seed_nodes == ()
        assert sink.output_shape == (1, 60)
        assert analysis.aux_nodes == set()

    def test_function_flatten_is_recorded_as_reshape_sink(self):
        class M(nn.Module):
            def forward(self, x):
                return torch.flatten(x, 1)

        gm = _trace_for_shape_analysis(M(), torch.randn(1, 3, 4, 5))
        analysis = analyze_shape_helpers(gm)

        flatten_node = next(
            node
            for node in gm.graph.nodes
            if node.op == "call_function" and node.target is torch.flatten
        )
        sink = analysis.sink_for(flatten_node)
        assert sink is not None

        assert sink.kind == "flatten"
        assert sink.data_input.op == "placeholder"
        assert sink.start_dim == 1
        assert sink.end_dim == -1
        assert sink.shape_seed_nodes == ()
        assert sink.output_shape == (1, 60)

    def test_function_unsqueeze_is_recorded_as_reshape_sink(self):
        class M(nn.Module):
            def forward(self, x):
                return torch.unsqueeze(x, 1)

        gm = _trace_for_shape_analysis(M(), torch.randn(1, 2, 3))
        analysis = analyze_shape_helpers(gm)

        unsqueeze_node = next(
            node
            for node in gm.graph.nodes
            if node.op == "call_function" and node.target is torch.unsqueeze
        )
        sink = analysis.sink_for(unsqueeze_node)
        assert sink is not None

        assert sink.kind == "reshape"
        assert sink.data_input.op == "placeholder"
        assert sink.shape_seed_nodes == ()
        assert sink.output_shape == (1, 1, 2, 3)

    def test_method_squeeze_is_recorded_as_reshape_sink(self):
        class M(nn.Module):
            def forward(self, x):
                return x.squeeze(1)

        gm = _trace_for_shape_analysis(M(), torch.randn(1, 1, 2, 3))
        analysis = analyze_shape_helpers(gm)

        squeeze_node = next(
            node
            for node in gm.graph.nodes
            if node.op == "call_method" and node.target == "squeeze"
        )
        sink = analysis.sink_for(squeeze_node)
        assert sink is not None

        assert sink.kind == "reshape"
        assert sink.data_input.op == "placeholder"
        assert sink.shape_seed_nodes == ()
        assert sink.output_shape == (1, 2, 3)

    def test_function_squeeze_is_recorded_as_reshape_sink(self):
        class M(nn.Module):
            def forward(self, x):
                return torch.squeeze(x, 1)

        gm = _trace_for_shape_analysis(M(), torch.randn(1, 1, 2, 3))
        analysis = analyze_shape_helpers(gm)

        squeeze_node = next(
            node
            for node in gm.graph.nodes
            if node.op == "call_function" and node.target is torch.squeeze
        )
        sink = analysis.sink_for(squeeze_node)
        assert sink is not None

        assert sink.kind == "reshape"
        assert sink.data_input.op == "placeholder"
        assert sink.shape_seed_nodes == ()
        assert sink.output_shape == (1, 2, 3)

    def test_function_reshape_is_recorded_as_reshape_sink(self):
        class M(nn.Module):
            def forward(self, x):
                return torch.reshape(x, (1, 6))

        gm = _trace_for_shape_analysis(M(), torch.randn(1, 2, 3))
        analysis = analyze_shape_helpers(gm)

        reshape_node = next(
            node
            for node in gm.graph.nodes
            if node.op == "call_function" and node.target is torch.reshape
        )
        sink = analysis.sink_for(reshape_node)
        assert sink is not None

        assert sink.kind == "reshape"
        assert sink.data_input.op == "placeholder"
        assert sink.shape_seed_nodes == ()
        assert sink.output_shape == (1, 6)

    def test_tuple_repeat_all_ones_is_recorded_as_reshape_sink(self):
        class M(nn.Module):
            def forward(self, x):
                return x.repeat((1, 1, 1))

        gm = _trace_for_shape_analysis(M(), torch.randn(1, 2, 3))
        analysis = analyze_shape_helpers(gm)

        repeat_node = next(
            node
            for node in gm.graph.nodes
            if node.op == "call_method" and node.target == "repeat"
        )
        sink = analysis.sink_for(repeat_node)
        assert sink is not None

        assert sink.kind == "reshape"
        assert sink.data_input.op == "placeholder"
        assert sink.shape_seed_nodes == ()
        assert sink.output_shape == (1, 2, 3)

    def test_view_shape_arithmetic_marks_only_shape_helpers(self):
        class M(nn.Module):
            def forward(self, x):
                batch = x.size(0) + x.size(1) - x.size(1)
                features = (x.size(1) * x.size(2) * x.size(3)) // batch
                return x.view(batch, features)

        gm = _trace_for_shape_analysis(M(), torch.randn(1, 2, 3, 4))
        analysis = analyze_shape_helpers(gm)

        view_node = next(
            node
            for node in gm.graph.nodes
            if node.op == "call_method" and node.target == "view"
        )
        sink = analysis.sink_for(view_node)
        assert sink is not None

        assert sink.kind == "reshape"
        assert sink.data_input.op == "placeholder"
        assert sink.output_shape == (1, 24)
        assert sink.shape_seed_nodes

        size_nodes = [
            node
            for node in gm.graph.nodes
            if node.op == "call_method" and node.target == "size"
        ]
        assert size_nodes
        assert all(analysis.is_aux(node) for node in size_nodes)
        assert not analysis.is_aux(sink.data_input)

    def test_shape_getattr_and_getitem_feed_shape_expression_semantics(self):
        class M(nn.Module):
            def forward(self, x):
                shape = x.shape
                batch = shape[0]
                features = shape[1] * shape[2] * shape[3]
                return x.view(batch, features)

        gm = _trace_for_shape_analysis(M(), torch.randn(1, 2, 3, 4))
        analysis = analyze_shape_helpers(gm)

        view_node = next(
            node
            for node in gm.graph.nodes
            if node.op == "call_method" and node.target == "view"
        )
        sink = analysis.sink_for(view_node)
        assert sink is not None

        shape_getattrs = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function"
            and node.target is getattr
            and len(node.args) >= 2
            and node.args[1] == "shape"
        ]
        getitems = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function" and node.target is operator.getitem
        ]

        assert sink.output_shape == (1, 24)
        assert shape_getattrs
        assert getitems
        assert all(analysis.is_aux(node) for node in shape_getattrs)
        assert all(analysis.is_aux(node) for node in getitems)

    def test_user_closure_excludes_size_node_shared_with_non_shape_user(self):
        class M(nn.Module):
            def forward(self, x):
                batch = x.size(0)
                y = x.view(batch, -1)
                return y, batch

        gm = _trace_for_shape_analysis(M(), torch.randn(1, 2, 3, 4))
        analysis = analyze_shape_helpers(gm)

        view_node = next(
            node
            for node in gm.graph.nodes
            if node.op == "call_method" and node.target == "view"
        )
        sink = analysis.sink_for(view_node)
        assert sink is not None
        size_node = next(
            node
            for node in gm.graph.nodes
            if node.op == "call_method" and node.target == "size"
        )

        assert sink.shape_seed_nodes == (size_node,)
        assert not analysis.is_aux(size_node)

    def test_flatten_then_fixed_reshape_records_two_sinks(self):
        class M(nn.Module):
            def forward(self, x):
                x = x.flatten(1)
                return x.reshape(1, 2, 12)

        gm = _trace_for_shape_analysis(M(), torch.randn(1, 2, 3, 4))
        analysis = analyze_shape_helpers(gm)

        flatten_node = next(
            node
            for node in gm.graph.nodes
            if node.op == "call_method" and node.target == "flatten"
        )
        reshape_node = next(
            node
            for node in gm.graph.nodes
            if node.op == "call_method" and node.target == "reshape"
        )

        flatten_sink = analysis.sink_for(flatten_node)
        reshape_sink = analysis.sink_for(reshape_node)
        assert flatten_sink is not None
        assert reshape_sink is not None

        assert flatten_sink.kind == "flatten"
        assert flatten_sink.data_input.op == "placeholder"
        assert flatten_sink.output_shape == (1, 24)
        assert flatten_sink.shape_seed_nodes == ()

        assert reshape_sink.kind == "reshape"
        assert reshape_sink.data_input is flatten_node
        assert reshape_sink.output_shape == (1, 2, 12)
        assert reshape_sink.shape_seed_nodes == ()
        assert len(analysis.reshape_sinks) == 2

    def test_flatten_then_reshape_shape_arithmetic_marks_flatten_size_users_aux(self):
        class M(nn.Module):
            def forward(self, x):
                x = x.flatten(1)
                return x.reshape(x.size(0), 2, x.size(1) // 2)

        gm = _trace_for_shape_analysis(M(), torch.randn(1, 2, 3, 4))
        analysis = analyze_shape_helpers(gm)

        flatten_node = next(
            node
            for node in gm.graph.nodes
            if node.op == "call_method" and node.target == "flatten"
        )
        reshape_node = next(
            node
            for node in gm.graph.nodes
            if node.op == "call_method" and node.target == "reshape"
        )
        reshape_sink = analysis.sink_for(reshape_node)
        assert reshape_sink is not None

        size_nodes = [
            node
            for node in gm.graph.nodes
            if node.op == "call_method"
            and node.target == "size"
            and node.args
            and node.args[0] is flatten_node
        ]

        assert analysis.sink_for(flatten_node) is not None
        assert reshape_sink.data_input is flatten_node
        assert reshape_sink.output_shape == (1, 2, 12)
        assert size_nodes
        assert all(analysis.is_aux(node) for node in size_nodes)
