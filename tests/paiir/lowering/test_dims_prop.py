import torch
from torch import fx, nn

from tests.paiir.tracing import trace_with_fx_shape_and_dims as _propagate


def _output_dims(gm: fx.GraphModule) -> tuple[int, ...]:
    """Return the dims of the output node."""
    for node in reversed(list(gm.graph.nodes)):
        if node.op == "output":
            return node.meta.get("dims", ())
    return ()


def _node_dims(gm: fx.GraphModule, target_name: str) -> tuple[int, ...]:
    """Return the dims of the first node whose name contains *target_name*."""
    for node in gm.graph.nodes:
        if target_name in node.name:
            return node.meta.get("dims", ())
    raise KeyError(f"No node with name containing '{target_name}'")


class TestDimsProp:
    def test_identity_dims(self):
        """Placeholder nodes get identity ordering matching their ndim."""

        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        gm = _propagate(M(), torch.randn(2, 3, 4))
        dims = _node_dims(gm, "x")
        assert dims == (0, 1, 2)

    def test_scalar_placeholder(self):
        """0-d input gets empty dims."""

        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        gm = _propagate(M(), torch.tensor(1.0))
        dims = _node_dims(gm, "x")
        assert dims == ()

    def test_call_module_resets_dims(self):
        """call_module nodes get identity ordering (modules define their own layout)."""

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3, padding=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.conv(x)

        gm = _propagate(M(), torch.randn(1, 3, 8, 8))
        dims = _node_dims(gm, "conv")
        assert dims == (0, 1, 2, 3)

    def test_transpose_by_args(self):
        """torch.transpose(x, dim0, dim1) swaps two axes."""

        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.transpose(x, 1, 2)

        gm = _propagate(M(), torch.randn(2, 3, 4))
        assert _output_dims(gm) == (0, 2, 1)

    def test_method_transpose(self):
        """x.transpose(dim0, dim1) method form."""

        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.transpose(0, 2)

        gm = _propagate(M(), torch.randn(2, 3, 4))
        assert _output_dims(gm) == (2, 1, 0)

    def test_double_transpose_restores_identity(self):
        """Transposing the same pair twice restores identity."""

        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.transpose(torch.transpose(x, 0, 1), 0, 1)

        gm = _propagate(M(), torch.randn(2, 3))
        assert _output_dims(gm) == (0, 1)

    def test_transpose_4d(self):
        """Transpose on a 4-d tensor."""

        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.transpose(1, 3)

        gm = _propagate(M(), torch.randn(1, 3, 4, 5))
        assert _output_dims(gm) == (0, 3, 2, 1)

    def test_permute_tuple_arg(self):
        """torch.permute(x, (dims...)) with a tuple."""

        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.permute(x, (2, 0, 1))

        gm = _propagate(M(), torch.randn(2, 3, 4))
        assert _output_dims(gm) == (2, 0, 1)

    def test_method_permute(self):
        """x.permute(d0, d1, ...) method form with separate int args."""

        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.permute(1, 2, 0)

        gm = _propagate(M(), torch.randn(2, 3, 4))
        assert _output_dims(gm) == (1, 2, 0)

    def test_permute_identity(self):
        """Permuting to identity ordering is a no-op."""

        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.permute(0, 1, 2)

        gm = _propagate(M(), torch.randn(2, 3, 4))
        assert _output_dims(gm) == (0, 1, 2)

    def test_permute_after_transpose(self):
        """Permute applied after a transpose composes correctly."""

        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                y = x.transpose(0, 2)  # (2, 1, 0)
                return y.permute(2, 1, 0)  # input_dims[2]=0, [1]=1, [0]=2 -> (0, 1, 2)

        gm = _propagate(M(), torch.randn(2, 3, 4))
        assert _output_dims(gm) == (0, 1, 2)

    def test_permute_4d(self):
        """4-d permutation (NCHW -> NHWC)."""

        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.permute(0, 2, 3, 1)

        gm = _propagate(M(), torch.randn(1, 3, 4, 5))
        assert _output_dims(gm) == (0, 2, 3, 1)

    def test_flatten_resets_to_identity(self):
        """Reshape-like ops (flatten, view, reshape) reset to identity."""

        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.flatten(1)

        gm = _propagate(M(), torch.randn(2, 3, 4))
        assert _output_dims(gm) == (0, 1)

    def test_function_flatten_resets_to_identity(self):
        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.flatten(x, 1)

        gm = _propagate(M(), torch.randn(2, 3, 4))
        assert _output_dims(gm) == (0, 1)
        assert _node_dims(gm, "flatten") == (0, 1)

    def test_view_resets_to_identity(self):
        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.view(2, 12)

        gm = _propagate(M(), torch.randn(2, 3, 4))
        assert _output_dims(gm) == (0, 1)

    def test_reshape_resets_to_identity(self):
        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.reshape(6, 4)

        gm = _propagate(M(), torch.randn(2, 3, 4))
        assert _output_dims(gm) == (0, 1)

    def test_function_reshape_resets_to_identity(self):
        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.reshape(x, (6, 4))

        gm = _propagate(M(), torch.randn(2, 3, 4))
        assert _output_dims(gm) == (0, 1)
        assert _node_dims(gm, "reshape") == (0, 1)

    def test_view_as_resets_to_identity(self):
        class M(nn.Module):
            def forward(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
                return x.view_as(ref)

        gm = _propagate(M(), torch.randn(1, 3, 2, 2), torch.randn(1, 12))
        assert _output_dims(gm) == (0, 1)
        assert _node_dims(gm, "view_as") == (0, 1)

    def test_function_unsqueeze_resets_to_identity(self):
        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.unsqueeze(x, 1)

        gm = _propagate(M(), torch.randn(1, 2, 3))
        assert _output_dims(gm) == (0, 1, 2, 3)
        assert _node_dims(gm, "unsqueeze") == (0, 1, 2, 3)

    def test_method_squeeze_resets_to_identity(self):
        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.squeeze(1)

        gm = _propagate(M(), torch.randn(1, 1, 2, 3))
        assert _output_dims(gm) == (0, 1, 2)
        assert _node_dims(gm, "squeeze") == (0, 1, 2)

    def test_function_squeeze_resets_to_identity(self):
        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.squeeze(x, 1)

        gm = _propagate(M(), torch.randn(1, 1, 2, 3))
        assert _output_dims(gm) == (0, 1, 2)
        assert _node_dims(gm, "squeeze") == (0, 1, 2)

    def test_contiguous_inherits_dims(self):
        """contiguous() preserves dims from input (no layout change)."""

        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.transpose(0, 1).contiguous()

        gm = _propagate(M(), torch.randn(2, 3))
        assert _output_dims(gm) == (1, 0)

    def test_add_inherits_from_first_input(self):
        """Binary ops inherit dims from the first input node."""

        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                y = x.transpose(0, 1)
                return y + y

        gm = _propagate(M(), torch.randn(2, 3))
        assert _output_dims(gm) == (1, 0)

    def test_relu_inherits(self):
        """Unary element-wise ops inherit dims."""

        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.relu(x.transpose(0, 2))

        gm = _propagate(M(), torch.randn(2, 3, 4))
        assert _output_dims(gm) == (2, 1, 0)

    def test_output_inherits_from_return_value(self):
        """Output node gets dims from the returned node."""

        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.transpose(0, 1)

        gm = _propagate(M(), torch.randn(2, 3))
        assert _output_dims(gm) == (1, 0)

    def test_output_identity_passthrough(self):
        """Identity model: output dims match input dims."""

        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        gm = _propagate(M(), torch.randn(2, 3, 4))
        assert _output_dims(gm) == (0, 1, 2)

    def test_get_attr_identity(self):
        """get_attr nodes get identity ordering."""

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.randn(3, 4))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + self.buf  # type: ignore

        gm = _propagate(M(), torch.randn(3, 4))
        for node in gm.graph.nodes:
            if node.op == "get_attr":
                assert node.meta["dims"] == (0, 1)
                break

    def test_transpose_then_flatten(self):
        """Transpose followed by flatten: flatten resets to identity."""

        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.transpose(1, 2).flatten(1)

        gm = _propagate(M(), torch.randn(2, 3, 4))
        assert _output_dims(gm) == (0, 1)

    def test_transpose_then_function_flatten(self):
        """Function-form flatten must also reset non-identity dims."""

        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.flatten(x.transpose(1, 2), 1)

        gm = _propagate(M(), torch.randn(2, 3, 4))
        assert _output_dims(gm) == (0, 1)
        assert _node_dims(gm, "flatten") == (0, 1)

    def test_permute_then_function_reshape(self):
        """Function-form reshape must reset non-identity dims."""

        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                y = x.permute(0, 2, 3, 1)
                return torch.reshape(y, (y.size(0), -1))

        gm = _propagate(M(), torch.randn(1, 2, 3, 4))
        assert _output_dims(gm) == (0, 1)
        assert _node_dims(gm, "reshape") == (0, 1)

    def test_conv_transpose_permute(self):
        """Conv (identity) -> transpose -> permute chain."""

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3, padding=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                y = self.conv(x)  # identity (0,1,2,3)
                y = y.transpose(1, 3)  # (0,3,2,1)
                y = y.permute(
                    0, 3, 2, 1
                )  # input[0]=0, input[3]=1, input[2]=2, input[1]=3 -> (0,1,2,3)
                return y

        gm = _propagate(M(), torch.randn(1, 3, 8, 8))
        assert _output_dims(gm) == (0, 1, 2, 3)

    def test_chained_permute(self):
        """Two permutations compose correctly."""

        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                y = x.permute(2, 0, 1)  # (2, 0, 1)
                return y.permute(
                    1, 2, 0
                )  # input[1]=0, input[2]=1, input[0]=2 -> (0, 1, 2)

        gm = _propagate(M(), torch.randn(2, 3, 4))
        assert _output_dims(gm) == (0, 1, 2)

    def test_skip_connection_same_dims(self):
        """Residual add where both branches have the same dims."""

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(16, 16, 3, padding=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + self.conv(x)

        gm = _propagate(M(), torch.randn(1, 16, 8, 8))
        assert _output_dims(gm) == (0, 1, 2, 3)

    def test_skip_connection_with_transpose(self):
        """Branch applies transpose; add inherits dims from first input."""

        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                branch = x.transpose(1, 2)  # (0, 2, 1)
                return branch + branch

        gm = _propagate(M(), torch.randn(2, 3, 4))
        assert _output_dims(gm) == (0, 2, 1)

    def test_multi_layer_cnn(self):
        """Multi-layer CNN: each conv resets dims to identity."""

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
                self.conv2 = nn.Conv2d(8, 16, 3, padding=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.conv1(x)
                x = x.transpose(2, 3)  # (0, 1, 3, 2) — only swap H/W, keep C=8
                x = self.conv2(x)  # call_module -> identity (0, 1, 2, 3)
                return x

        gm = _propagate(M(), torch.randn(1, 3, 8, 8))
        assert _output_dims(gm) == (0, 1, 2, 3)
        assert _node_dims(gm, "transpose") == (0, 1, 3, 2)

    def test_nchw_to_nhwc_pipeline(self):
        """Realistic NCHW -> NHWC conversion followed by flatten."""

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3, padding=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.conv(x)  # (0, 1, 2, 3) NCHW
                x = x.permute(0, 2, 3, 1)  # (0, 2, 3, 1) NHWC
                x = x.contiguous()  # inherits (0, 2, 3, 1)
                x = x.flatten(1)  # resets to (0, 1)
                return x

        gm = _propagate(M(), torch.randn(1, 3, 8, 8))
        assert _output_dims(gm) == (0, 1)
        assert _node_dims(gm, "permute") == (0, 2, 3, 1)
        assert _node_dims(gm, "contiguous") == (0, 2, 3, 1)
