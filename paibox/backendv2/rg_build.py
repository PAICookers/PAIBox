from .neuron import InputElem, Neuron
from .op_node import CoreOpNode, CustomIndex, InputNode
from .routing import RoutingGroup


def gen_neurons(node: CoreOpNode) -> list[Neuron]:
    neurons: list[Neuron] = []
    # flatten the shape to get total number of neurons
    num_neu = node.shape.numel()
    for idx in range(num_neu):
        neu: Neuron = Neuron(target=node, index=CustomIndex(idx))
        neurons.append(neu)
    return neurons


def gen_ioelements(node: InputNode) -> list[InputElem]:
    ioelements: list[InputElem] = []
    # flatten the shape to get total number of neurons
    num_elem = node.shape.numel()
    for idx in range(num_elem):
        ioelem: InputElem = InputElem(target=node, index=CustomIndex(idx))
        ioelements.append(ioelem)
    return ioelements


def build_routing_group(
    nodes: set[CoreOpNode], input_nodes: set[CoreOpNode | InputNode]
) -> RoutingGroup:
    # Placeholder implementation

    raw_neus: list[Neuron] = []
    for node in nodes:
        raw_neus.extend(gen_neurons(node))
    input_list: list[Neuron | InputElem] = []
    for in_node in input_nodes:
        if isinstance(in_node, CoreOpNode):
            input_list.extend(gen_neurons(in_node))
        elif isinstance(in_node, InputNode):
            input_list.extend(gen_ioelements(in_node))
    rg = RoutingGroup(raw_neus, input_list)
    # Logic to build a routing group from nodes goes here
    return rg


def build_routing_groups(nodes: list[CoreOpNode | InputNode]) -> list[RoutingGroup]:
    # Placeholder implementation
    routing_groups: list[RoutingGroup] = []
    # Logic to build routing groups from nodes goes here
    node_sets: list[set[CoreOpNode]] = []
    for node in nodes:
        if len(node.successors) > 0:
            node_sets.append(set(node.successors))
        if isinstance(node, CoreOpNode):
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

    routing_groups: list[RoutingGroup] = []

    for node_set in node_sets:
        input_nodes: set[CoreOpNode | InputNode] = set()
        for node in node_set:
            input_nodes.update(node.predecessors)
        rg = build_routing_group(node_set, input_nodes)
        routing_groups.append(rg)

    return routing_groups
