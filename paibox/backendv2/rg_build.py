from .op_node import (
    AllNode,
    CoreOpNode,
    CustomIndex,
    InNode,
    InputElem,
    Neuron,
    OutNode,
    RemapElem,
    ReorderNode,
    SourceElem,
    SourceNode,
    get_elem,
)
from .routing import InputGroup, OutputGroup, RemapGroup, RoutingGroup


def gen_elems(node: SourceNode) -> list[SourceElem]:
    elems: list[SourceElem] = []
    # flatten the shape to get total number of neurons
    num_elem = node.shape.numel()
    for idx in range(num_elem):
        elem: SourceElem = get_elem(node, idx)
        elems.append(elem)
    return elems


def gen_neurons(node: CoreOpNode) -> list[Neuron]:
    neurons: list[Neuron] = []
    # flatten the shape to get total number of neurons
    num_neu = node.shape.numel()
    for idx in range(num_neu):
        neu: Neuron = Neuron(target=node, index=CustomIndex(idx))
        neurons.append(neu)
    return neurons


def gen_ioelements(node: InNode) -> list[InputElem]:
    ioelements: list[InputElem] = []
    # flatten the shape to get total number of neurons
    num_elem = node.shape.numel()
    for idx in range(num_elem):
        ioelem: InputElem = InputElem(target=node, index=CustomIndex(idx))
        ioelements.append(ioelem)
    return ioelements


def gen_reorder_elems(node: ReorderNode) -> list[RemapElem]:
    elems: list[RemapElem] = []
    # flatten the shape to get total number of neurons
    num_elem = node.shape.numel()
    for idx in range(num_elem):
        elem: RemapElem = RemapElem(target=node, index=CustomIndex(idx))
        elems.append(elem)
    return elems


def build_routing_group(
    nodes: set[CoreOpNode], input_nodes: set[SourceNode]
) -> RoutingGroup:
    raw_neus: list[Neuron] = []
    for node in nodes:
        raw_neus.extend(gen_neurons(node))
    input_list: list[SourceElem] = []
    for in_node in input_nodes:
        input_list.extend(gen_elems(in_node))
    rg = RoutingGroup(raw_neus, input_list, nodes, input_nodes)
    return rg


def build_reorder_group(
    nodes: set[ReorderNode], input_nodes: set[SourceNode]
) -> RemapGroup:
    raw_neus: list[RemapElem] = []
    for node in nodes:
        raw_neus.extend(gen_reorder_elems(node))
    input_list: list[SourceElem] = []
    for in_node in input_nodes:
        input_list.extend(gen_elems(in_node))
    rg = RemapGroup(raw_neus, input_list, nodes, input_nodes)
    return rg


def build_input_group(nodes: set[InNode], input_nodes: set[SourceNode]) -> InputGroup:
    assert len(input_nodes) == 0, "Input group should not have any input nodes."
    raw_neus: list[InputElem] = []
    for node in nodes:
        raw_neus.extend(gen_ioelements(node))
    in_grp = InputGroup(raw_neus, nodes)
    return in_grp


def build_output_group(
    nodes: set[OutNode], input_nodes: set[SourceNode]
) -> OutputGroup:
    assert len(nodes) == 1, "Output group should have exactly one output node."
    input_list: list[SourceElem] = []
    for input_node in input_nodes:
        input_list.extend(gen_elems(input_node))
    out_grp = OutputGroup(input_list, input_nodes)
    return out_grp


def build_groups(
    nodes: list[AllNode],
) -> tuple[list[RemapGroup | RoutingGroup], list[InputGroup], list[OutputGroup]]:
    # Placeholder implementation

    # Logic to build routing groups from nodes goes here
    node_sets: list[set[AllNode]] = []
    for node in nodes:
        if len(node.successors) > 0:
            node_sets.append(set(node.successors))
        node_sets.append({node})

    while True:
        merged = False
        for i in range(len(node_sets)):
            for j in range(i + 1, len(node_sets)):
                if node_sets[i] & node_sets[j]:
                    node_sets[i] |= node_sets[j]
                    node_sets.pop(j)
                    merged = True
                    break
            if merged:
                break
        if not merged:
            break
    print(f"Identified {len(node_sets)} groups.")
    print(node_sets)
    groups: list[RemapGroup | RoutingGroup] = []
    input_groups: list[InputGroup] = []
    output_groups: list[OutputGroup] = []

    for node_set in node_sets:
        input_nodes: set[SourceNode] = set()
        reorder_node_set: set[ReorderNode] = set()
        routing_node_set: set[CoreOpNode] = set()
        input_node_set: set[InNode] = set()
        output_node_set: set[OutNode] = set()
        print(f"\nProcessing node set {node_set} for group building:")
        for node in node_set:
            print(f"\tProcessing node {node} in group building:")
            print(f"\t\tPredecessors: {node.predecessors}")
            print(f"\t\tSuccessors: {node.successors}")
            if isinstance(node, ReorderNode):
                reorder_node_set.add(node)
            elif isinstance(node, CoreOpNode):
                routing_node_set.add(node)
            elif isinstance(node, InNode):
                input_node_set.add(node)
            elif isinstance(node, OutNode):
                output_node_set.add(node)
            input_nodes.update(node.predecessors)

        group_node_sets = {
            "reorder_node_set": reorder_node_set,
            "routing_node_set": routing_node_set,
            "input_node_set": input_node_set,
            "output_node_set": output_node_set,
        }

        non_empty_sets = {name: s for name, s in group_node_sets.items() if len(s) > 0}
        assert (
            len(non_empty_sets) == 1
        ), f"Expected exactly one non-empty node set for group building, but got: {non_empty_sets}"
        group_type, _ = non_empty_sets.popitem()

        if group_type == "reorder_node_set":
            group = build_reorder_group(reorder_node_set, input_nodes)
            groups.append(group)
        elif group_type == "routing_node_set":
            group = build_routing_group(routing_node_set, input_nodes)
            groups.append(group)
        elif group_type == "input_node_set":
            in_group = build_input_group(input_node_set, input_nodes)
            input_groups.append(in_group)
        elif group_type == "output_node_set":
            out_group = build_output_group(output_node_set, input_nodes)
            output_groups.append(out_group)

    return groups, input_groups, output_groups
