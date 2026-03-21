from typing import Optional

import torch
from paicorelib import (
    OfflineNeuFullAttrsV2Part2,
    OutputType,
    SNNMode,
    WeightCompressType,
)
from torch import Tensor, nn

from ..paiir import (
    AccumulateOp,
    InputNode,
    LutData,
    OfflineCoreOp,
    OfflineCoreParams,
    OutputNode,
    PAIIRGraph,
    SequentialOp,
    StandaloneActOp,
    StandaloneCompOp,
)
from .core_config import Frontend_Core_Config


class CustomIndex:
    def __init__(self, idx: int):
        self.idx = idx

    def __hash__(self) -> int:
        return hash(self.idx)

    def __eq__(self, value: "CustomIndex") -> bool:
        return self.idx == value.idx


def get_frontend_core_conf(
    core_params: OfflineCoreParams, lut_data: Optional[LutData]
) -> Frontend_Core_Config:
    assert core_params.tick_start is not None
    if core_params.snn_mode == SNNMode.ANN:
        assert core_params.tick_initial == 1, "ANN mode requires tick_initial=1"
        # assert lut_data is not None, "lut_data must be provided for ANN mode"
    elif core_params.snn_mode == SNNMode.SNN:
        assert lut_data is None, "lut_data should not be provided for SNN mode"

    return Frontend_Core_Config(
        snn_ann=core_params.snn_mode,
        max_pooling=core_params.pooling_mode,
        zero_output=core_params.zero_output,
        input_sign=core_params.input_sign,
        input_width=core_params.input_width,
        output_sign=core_params.output_sign,
        output_width=core_params.output_width,
        weight_sign=core_params.weight_sign,
        weight_width=core_params.weight_width,
        tick_start=core_params.tick_start,
        tick_duration=core_params.tick_duration,
        tick_initial=core_params.tick_initial,
        lut_data=lut_data,
    )


class InNode:
    def __init__(self, name: str, shape: tuple[int, ...]):
        self.name = name
        self.shape = torch.Size(shape)
        self.successors: list[CoreOpNode] = []

        # Note: InNodes don't have predecessors since they represent external inputs, but we keep the attribute for uniformity
        self.predecessors: list[CoreOpNode | InNode] = []

    def __hash__(self) -> int:
        return hash(id(self))

    def __str__(self) -> str:
        return f"InputNode({self.name})"

    def __repr__(self) -> str:
        return self.__str__()


class CoreOpNode:
    def __init__(self, name: str, raw_node: OfflineCoreOp, shape: tuple[int, ...]):
        self.name = name
        self.raw_node = raw_node
        self.shape = torch.Size(shape)
        self.successors: list[CoreOpNode] = []
        self.predecessors: list[CoreOpNode | InNode] = []
        self.comps: list[Optional[nn.Module]] = []
        self.weights: list[Optional[Tensor]] = []
        self.set_comps_and_weights()

        self.frontend_core_config: Frontend_Core_Config = get_frontend_core_conf(
            raw_node.core_params, raw_node.lut_data
        )

    def set_comps_and_weights(self) -> None:
        """Set self.comps based on the type of raw_node."""
        if isinstance(self.raw_node, SequentialOp):
            self.comps = [self.raw_node.comp]
        elif isinstance(self.raw_node, AccumulateOp):
            self.comps = list(self.raw_node.comps)
        elif isinstance(self.raw_node, StandaloneCompOp):
            self.comps = [self.raw_node.comp]
        elif isinstance(self.raw_node, StandaloneActOp):
            self.comps = [None]
        else:
            raise NotImplementedError(f"Unsupported node type: {type(self.raw_node)}")

        weights = self.raw_node.weights
        print(f"Setting comps and weights for node {self.name} \n\tcomps: {self.comps} \n\traw weights: {weights}")
        if weights is not None:
            self.weights = list(weights)
        else:
            self.weights = [None]

    def __hash__(self) -> int:
        return hash(id(self))

    def __str__(self) -> str:
        return f"CoreOpNode({self.name})"

    def __repr__(self) -> str:
        return self.__str__()

    def attrs_part2(self, idx=0) -> OfflineNeuFullAttrsV2Part2:
        neu_attrs = self.raw_node.neuron_params

        if isinstance(neu_attrs.leak_v, torch.Tensor):
            assert self.shape[0] == 1, "Batch size > 1 not supported for tensor leak_v"
            out_channel = self.shape[1]
            assert (
                neu_attrs.leak_v.numel() == out_channel
            ), "leak_v tensor size must match output channels"
            cur_channel = idx // (self.shape.numel() // self.shape[1])
            leak_v = neu_attrs.leak_v[cur_channel].item()
        else:
            leak_v = neu_attrs.leak_v

        return OfflineNeuFullAttrsV2Part2(
            reset_mode=neu_attrs.reset_mode,
            reset_v=round(neu_attrs.reset_v),
            threshold_neg_mode=neu_attrs.thres_neg_mode,
            threshold_pos_mode=neu_attrs.thres_pos_mode,
            threshold_neg=round(neu_attrs.thres_neg),
            threshold_pos=round(neu_attrs.thres_pos),
            lateral_inhibition=neu_attrs.lateral_inhi,
            leak_multi_sequence=neu_attrs.leak_multi_sequence,
            leak_multi_input=neu_attrs.leak_multi_input,
            leak_multi_mode=neu_attrs.leak_multi_mode,
            leak_add_mode=neu_attrs.leak_add_mode,
            leak_tau=neu_attrs.leak_tau,
            leak_v=round(leak_v),
            weight_compress=WeightCompressType.DENSE,
            vjt_initial=round(neu_attrs.init_v),
        )

    def output_type(self) -> OutputType:
        neu_attrs = self.raw_node.neuron_params
        return neu_attrs.output_type

    def core_config(self) -> Frontend_Core_Config:
        return self.frontend_core_config


def build_nodes(graph: PAIIRGraph) -> list[CoreOpNode | InNode]:
    nodes: list[CoreOpNode | InNode] = []
    nodes_map: dict[str, CoreOpNode | InNode] = {}
    for raw_node in graph.nodes.values():
        if isinstance(raw_node, OfflineCoreOp):
            node = CoreOpNode(raw_node.name, raw_node, raw_node.output_shape)
        elif isinstance(raw_node, InputNode):
            node = InNode(raw_node.name, raw_node.shape)
        elif isinstance(raw_node, OutputNode):
            pass
        else:
            raise NotImplementedError(f"Unsupported node type: {type(raw_node)}")
        nodes_map[raw_node.name] = node
        nodes.append(node)

    for node_name, cur_node in nodes_map.items():
        succ_node_names = graph.successors(node_name)
        for succ_name in succ_node_names:
            succ_node = nodes_map[succ_name]
            assert isinstance(
                succ_node, (CoreOpNode)
            ), "Only CoreOpNode can be a successor of other node"
            cur_node.successors.append(succ_node)
        if isinstance(cur_node, CoreOpNode):
            pred_node_names = graph.predecessors(node_name)
            for pred_name in pred_node_names:
                pred_node = nodes_map[pred_name]
                cur_node.predecessors.append(pred_node)

    return nodes
