from typing import Dict, Union, List, Tuple

class NeuDyn:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"NeuDyn Node: {self.name}"

class InputProj:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"InputProj Node: {self.name}"

class SynSys:
    def __init__(self, source, dest):
        self.source = source
        self.dest = dest

    def __str__(self):
        return f"Edge from {self.source.name} to {self.dest.name}"

def separate_network(nodes: Dict[str, Union[NeuDyn, InputProj]],
                     edges: Dict[str, SynSys]) -> \
                     List[Tuple[Dict[str, Union[NeuDyn, InputProj]], Dict[str, SynSys]]]:

    def dfs(node, component_nodes):
        component_nodes.append(node)
        for edge_name, edge in edges.items():
            if(edge_name in edges_visited):
                continue
            source = edge.source.name
            dest = edge.dest.name
            next = None
            
            if source == node and dest in nodes_remaining:
                next = dest
            elif dest == node and source in nodes_remaining:
                next = source
                
            if next is not None:
                nodes_remaining.remove(next)
                edges_visited.add(edge_name)
                dfs(next, component_nodes)
            

    components = []
    nodes_remaining = set(nodes.keys())
    edges_visited   = set()

    while nodes_remaining:
        current_node = nodes_remaining.pop()
        component_nodes = []
        dfs(current_node, component_nodes)
        component_edges = {edge_name: edge for edge_name, edge in edges.items() if
                           edge.source.name in component_nodes and edge.dest.name in component_nodes}
        component = ({node_name: nodes[node_name] for node_name in component_nodes}, component_edges)
        edges_visited |= set(component_edges.keys())
        components.append(component)

    return components

# Example usage:
nodes = {
    'A': NeuDyn('A'),
    'B': NeuDyn('B'),
    'C': NeuDyn('C'),
    'D': NeuDyn('D'),
    'E': NeuDyn('E')
}

edges = {
    '1': SynSys(nodes['A'], nodes['B']),
    '2': SynSys(nodes['B'], nodes['C']),
    '3': SynSys(nodes['C'], nodes['A']),
    '4': SynSys(nodes['D'], nodes['E'])
}

result = separate_network(nodes, edges)
for component in result:
    print("Nodes:")
    for node in component[0].values():
        print(node)
    print("Edges:")
    for edge in component[1].values():
        print(edge)
    print("-------------------")