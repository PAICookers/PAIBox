import paibox as pb


def test_paiboxobject_find_nodes():
    n1 = pb.neuron.TonicSpikingNeuron(2, 3)
    n2 = pb.neuron.TonicSpikingNeuron(2, 3)
    s1 = pb.synapses.NoDecay(n1, n2, pb.synapses.All2All())
    n3 = pb.neuron.TonicSpikingNeuron(2, 4)

    seq1 = pb.network.Sequential(n1, s1, n2)

    network = pb.network.Network(seq1, n3)

    nodes = seq1.nodes(method="relative", level=-1, include_self=False)

    print("nodes:")
    for k, v in nodes.items():
        print(k, v)

    nodes_2 = network.nodes()

    print("nodes_2:")
    for k, v in nodes_2.items():
        print(k, v)

    print("nodedict in network")
    for k, v in network.children.items():
        print(k, v)
