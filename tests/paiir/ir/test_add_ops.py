import pytest
import torch

from paibox.paiir.ir.add_ops import PotentialAddOp


class TestPotentialAddOp:
    def test_elementwise_add_forward(self):
        op = PotentialAddOp(op_signs=(1, -1))
        x1 = torch.randn(1, 4, 8, 8)
        x2 = torch.randn(1, 4, 8, 8)

        out = op(x1, x2)

        assert out.shape == x1.shape
        assert torch.allclose(out, x1 - x2)

    def test_nary_elementwise_add_forward(self):
        op = PotentialAddOp(op_signs=(1, -1, 1))
        x1 = torch.randn(1, 4, 8, 8)
        x2 = torch.randn(1, 4, 8, 8)
        x3 = torch.randn(1, 4, 8, 8)

        out = op(x1, x2, x3)

        assert out.shape == x1.shape
        assert torch.allclose(out, x1 - x2 + x3)

    def test_requires_two_paths(self):
        with pytest.raises(ValueError, match="at least two signed input paths"):
            PotentialAddOp(op_signs=(1,))
