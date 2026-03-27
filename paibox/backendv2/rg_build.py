from .op_node import (
    BaseNode,
    CoreOpNode,
    CustomIndex,
    DestNode,
    InNode,
    InputElem,
    Neuron,
    ReorderElem,
    ReorderNode,
    SourceElem,
    SourceNode,
    get_elem,
)
from .routing import ReorderGroup, RoutingGroup


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


def gen_reorder_elems(node: ReorderNode) -> list[ReorderElem]:
    elems: list[ReorderElem] = []
    # flatten the shape to get total number of neurons
    num_elem = node.shape.numel()
    for idx in range(num_elem):
        elem: ReorderElem = ReorderElem(target=node, index=CustomIndex(idx))
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
) -> ReorderGroup:
    raw_neus: list[ReorderElem] = []
    for node in nodes:
        raw_neus.extend(gen_reorder_elems(node))
    input_list: list[SourceElem] = []
    for in_node in input_nodes:
        input_list.extend(gen_elems(in_node))
    rg = ReorderGroup(raw_neus, input_list, nodes, input_nodes)
    return rg


def build_groups(nodes: list[SourceNode]) -> list[ReorderGroup | RoutingGroup]:
    # Placeholder implementation
    groups: list[ReorderGroup | RoutingGroup] = []
    # Logic to build routing groups from nodes goes here
    node_sets: list[set[DestNode]] = []
    for node in nodes:
        if len(node.successors) > 0:
            node_sets.append(set(node.successors))
        if isinstance(node, DestNode):
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
    print(f"Identified {len(node_sets)} routing groups.")
    print(node_sets)

    for node_set in node_sets:
        input_nodes: set[SourceNode] = set()
        reorder_node_set: set[ReorderNode] = set()
        routing_node_set: set[CoreOpNode] = set()
        print(f"\nProcessing node set {node_set} for group building:")
        for node in node_set:
            print(f"\tProcessing node {node} in group building:")
            print(f"\t\tPredecessors: {node.predecessors}")
            print(f"\t\tSuccessors: {node.successors}")
            if isinstance(node, ReorderNode):
                reorder_node_set.add(node)
            elif isinstance(node, CoreOpNode):
                routing_node_set.add(node)
            input_nodes.update(node.predecessors)
        print("Group input nodes:", input_nodes)
        is_reorder = len(reorder_node_set) == 1 and len(routing_node_set) == 0
        is_routing = len(reorder_node_set) == 0 and len(routing_node_set) > 0

        assert is_reorder ^ is_routing, "Invalid group type."

        if is_reorder:
            group = build_reorder_group(reorder_node_set, input_nodes)
        else:
            group = build_routing_group(routing_node_set, input_nodes)

        groups.append(group)

    return groups
