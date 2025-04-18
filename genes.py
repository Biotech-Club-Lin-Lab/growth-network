import random


class NodeGene:
    def __init__(
        self,
        node_id: int,
        layer: str | int,
        bias: float = None,
        activation_function: str = "ReLU",
    ):
        """
        Node class represents the genetic encoding for a single neuron in a network.

        :param node_id: Unique identifier for the node
        :param bias: Bias value for the node
        :param layer: Type of node (e.g., 'input', 'output' or a number for hidden layers)
        :param activation_function: Activation function for the node (default is 'ReLU')
        """

        self.id = node_id  # Unique identifier for the node
        self.layer = layer
        self.bias = (
            bias if bias is not None else random.uniform(-1, 1)
        )  # Bias value for the node
        self.af = activation_function  # define code later


class EdgeGene:
    def __init__(
        self,
        edge_id: int,
        node: int,
        out_edge_to: int,
        weight: float,
        enabled: bool = True,
    ):
        """
        Edge class is the genetic encoding for a single connection in a network.
        Each edge_gene defines a connection between two neurons with a weight.

        :param edge_id: Unique identifier for the gene
        :param out_edge_to: Output neuron (target node)
        :param weight: Weight of the connection (can be positive or negative)
        :param enabled: Boolean indicating if the gene is enabled or disabled
        """
        self.id = edge_id  # Unique identifier for the gene
        self.node = node
        self.out_edge_to = out_edge_to  # Output neuron
        self.weight = weight  # weight
        self.enabled = enabled  # Boolean indicating if the gene is enabled or disabled
