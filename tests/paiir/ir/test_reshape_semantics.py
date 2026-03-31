import torch

from paibox.paiir.ir.reshape_semantics import (
    canonicalize_layout_view,
    materialize_logical_layout,
)


class TestReshapeSemantics:
    def test_canonicalizes_singleton_axis_layout_view(self):
        logical_shape, canonical_dims = canonicalize_layout_view(
            torch.Size((1, 3, 4)), (1, 0, 2)
        )

        assert logical_shape == (3, 1, 4)
        assert canonical_dims == (0, 1, 2)

    def test_keeps_non_singleton_permutation(self):
        logical_shape, canonical_dims = canonicalize_layout_view(
            torch.Size((2, 3, 4)), (1, 0, 2)
        )

        assert logical_shape == (2, 3, 4)
        assert canonical_dims == (1, 0, 2)

    def test_materializes_singleton_axis_layout_with_reshape_equivalent(self):
        x = torch.arange(12).reshape(1, 3, 4)

        result = materialize_logical_layout(x, (1, 0, 2))

        assert result.shape == (3, 1, 4)
        assert torch.equal(result, x.permute(1, 0, 2))

    def test_materializes_non_singleton_layout_with_permute_semantics(self):
        x = torch.arange(24).reshape(2, 3, 4)

        result = materialize_logical_layout(x, (1, 0, 2))

        assert result.shape == (3, 2, 4)
        assert torch.equal(result, x.permute(1, 0, 2))
