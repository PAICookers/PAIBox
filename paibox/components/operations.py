from .neuron import Neuron
from .projection import InputProj
from typing import Union
import numpy as np
from paibox.utils import as_shape, shape2num
from paibox.base import PAIBoxObject
from paibox.types import NEUOUT_U8_DTYPE, DataType, NeuOutType, Shape
__all__ = ["Transpose", "VirtualNode", "Concat"]


class VirtualNode(PAIBoxObject):
    def __init__(self, name: str):
        super().__init__(name)
    
    @property
    def shape_out(self) -> tuple[int, ...]:
        """Actual shape of output."""
        raise NotImplementedError
    
    @property
    def num_out(self) -> int:
        raise NotImplementedError
    
    def output(self, index: int) -> NeuOutType:
        raise NotImplementedError
    
    @property
    def delay_relative(self) -> int:
        raise TypeError("Virtual_Node does not have delay_relative")
    
    
    def get_output(self, source: Union[Neuron, InputProj], index: int) -> NeuOutType:
        if isinstance(source, Neuron):
            return source.delay_registers[index]
        elif isinstance(source, InputProj):
            return source.output
        else:
            raise TypeError("source must be an instance of Neuron or InputProj")

class Transpose(VirtualNode):
    def __init__(self, source: Union[Neuron, InputProj], axis: tuple[int, ...] = (1, 0)):
        super().__init__(f"Transposed_{source.name}")
        if (not isinstance(source, Neuron)) and (not isinstance(source, InputProj)):
            raise TypeError("source must be an instance of Neuron or InputProj")
        
        if (len(source.shape_out) != 2):
            raise ValueError("source to be transposed must have 2 dimensions")
        n_nuerons = source.num_out
        shape = source.shape_out
        order = np.arange(n_nuerons)
        order = order.reshape(shape)
        order_transposed = np.transpose(order, axis)
        
        self.in_order = order_transposed.ravel()
        self.weight_order = self.in_order.argsort()
        self._shape_out = order_transposed.shape
        self.source = source
        
    @property
    def shape_out(self) -> tuple[int, ...]:
        return self._shape_out
        
    @property
    def num_out(self) -> int:
        return shape2num(self.shape_out)
    
    def output(self, idx: int) -> NeuOutType:
        output = self.get_output(self.source, idx)
        output = output.ravel()
        output = output[self.in_order]
        output = output.reshape(self.shape_out)
        return output
            
class Concat(VirtualNode):
    def __init__(self, sources: list[Union[Neuron, InputProj]], axis: int = 0):
        # Check if whether the shape of the sources are the same, except for the axis
        if len(sources) < 2:
            raise ValueError("Concatenation must have at least two source")
        
        super().__init__(f"Concat_{'_'.join([source.name for source in sources])}")
        
        shapes_without_axis: list[tuple[int, ...]] = list()
        orders: list[np.ndarray] = list()
        axis_length: int = 0
        offset: int = 0
        for source in sources:
            shapes_without_axis.append(source.shape_out[:axis] + source.shape_out[axis+1:])
            axis_length += source.shape_out[axis]
            n_neurons = source.num_out
            shape = source.shape_out
            order = np.arange(offset, offset + n_neurons)
            order = order.reshape(shape)
            orders.append(order)
            offset += n_neurons

        if len(set(shapes_without_axis)) != 1:
            raise ValueError("Shapes of sources must be the same, except for the axis")

        self.in_order = np.concatenate(orders, axis).ravel()
        self.weight_order = self.in_order.argsort()
        self._shape_out = sources[0].shape_out[:axis] + (axis_length,) + sources[0].shape_out[axis+1:]
        self.sources = sources
        
    @property
    def shape_out(self) -> tuple[int, ...]:
        return self._shape_out
        
    @property
    def num_out(self) -> int:
        return shape2num(self.shape_out)
    
    def output(self, idx: int) -> NeuOutType:
        outputs: list[NeuOutType] = list()
        for source in self.sources:
            outputs.append(self.get_output(source, idx).ravel())
        output = np.concatenate(outputs, axis=0)
        output = output[self.in_order]
        output = output.reshape(self.shape_out)
        return output




