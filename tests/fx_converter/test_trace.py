import torch
from torch import nn

from paibox._logging import DEFAULT_LOG_SETTINGS, set_logs
from paibox.fx_converter.trace import (
    trace_spikingjelly_model,
)

set_logs(**DEFAULT_LOG_SETTINGS)


class TestShapeAnnotation:
    def test_propagate_tensor_shape(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.li1 = nn.Linear(128, 64)
                self.li2 = nn.Linear(256, 64)

            def forward(self, x, y):
                x = self.li1(x)
                y = self.li2(y)

                # call_method    transpose
                z = x @ y
                return z.permut(1, 0)

        m = M()
        gm = trace_spikingjelly_model(m)
        gm.graph.print_tabular()

        found = gm.graph.find_nodes(
            op="call_function", target=torch.ops.aten.transpose.int
        )
        print(found)
