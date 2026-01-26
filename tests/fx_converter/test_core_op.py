import unittest
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from torch import fx

from paibox.fx_converter.core_op import CoreOpNode, OpLoc
from paibox.fx_converter.lut_activation import LutReLU


class TestCoreOpNode(unittest.TestCase):
    def test_init_and_forward(self):
        # Create dummy inputs
        linear = nn.Linear(10, 5)
        # Initialize weights to known values for deterministic testing if needed
        nn.init.constant_(linear.weight, 1.0)
        nn.init.constant_(linear.bias, 0.0)

        # simple mock activation that just returns input + 1
        class MockActivation(nn.Module):
            def forward(self, x):
                return x + 1

            def get_attrs(self):
                return {}

        # We can't easily mock isinstance checks inside CoreOpNode without patching,
        # but CoreOpNode checks specific bases.
        # Using a real LutReLU is safer for structural tests.
        act = LutReLU()
        # Monkey patch forward of this instance for testing flow
        act.forward = MagicMock(side_effect=lambda x: x + 1.0)

        core_op = CoreOpNode([linear], act)

        x = torch.ones(1, 10)  # Sum = 10 * 1 = 10
        y = core_op(x)

        # Linear(ones) -> sum(1.0*1.0)*10 = 10.0
        # CoreOpNode sums inputs (here only one) -> 10.0
        # Act(10.0) -> 11.0

        self.assertTrue(torch.allclose(y, torch.full_like(y, 11.0)))

    def test_implicit_sum_signs(self):
        # op1 and op2
        op1 = nn.Identity()
        op2 = nn.Identity()

        # Mock activation
        act = LutReLU()
        act.forward = MagicMock(side_effect=lambda x: x)

        # Signs: 1, -1
        core_op = CoreOpNode([op1, op2], act, implicit_sum_signs=[1, -1])

        x1 = torch.tensor([10.0])
        x2 = torch.tensor([4.0])

        y = core_op(x1, x2)
        # 10 * 1 + 4 * -1 = 6
        self.assertEqual(y.item(), 6.0)

    def test_build(self):
        # Construct a simple graph
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5)
                self.relu = nn.ReLU()

            def forward(self, x):
                return self.relu(self.fc(x))

        model = SimpleModel()
        gm = fx.symbolic_trace(model)

        nodes = list(gm.graph.nodes)
        fc_node = next(n for n in nodes if n.target == "fc")
        relu_node = next(n for n in nodes if n.target == "relu")
        module_dict = dict(gm.named_modules())

        core_op = CoreOpNode.build(
            [fc_node], relu_node, module_dict, OpLoc.OFFLINE_CORE
        )

        self.assertIsInstance(core_op, CoreOpNode)
        self.assertEqual(len(core_op.op1), 1)
        self.assertIsInstance(core_op.op1[0], nn.Linear)
        self.assertIsInstance(core_op.op2, LutReLU)
