from .operations import Concat, Transpose, VirtualNode
from .synapses import ConnType, FullConnectedSyn, FullConnSyn


def process_edge(edge: FullConnectedSyn) -> list[FullConnectedSyn]:
    processed_edges: list[FullConnectedSyn] = list()
    if isinstance(edge.source, Transpose):
        processed_edge = FullConnSyn(
            edge.source.source,
            edge.dest,
            edge.connectivity[edge.source.weight_order],
            ConnType.All2All,
            f"{edge.name}_transpose",
        )
        processed_edges.append(processed_edge)
    elif isinstance(edge.source, Concat):
        connectivity = edge.connectivity[edge.source.weight_order]
        offset = 0
        for i, source in enumerate(edge.source.sources):
            processed_edge = FullConnSyn(
                source,
                edge.dest,
                connectivity[offset : offset + source.num_out],
                ConnType.All2All,
                f"{edge.name}_concat_{i}",
            )
            processed_edges.append(processed_edge)
            offset = offset + source.num_out
    else:
        processed_edges.append(edge)

    return processed_edges
