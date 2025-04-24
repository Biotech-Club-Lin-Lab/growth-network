from genes import *
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os


class Chromosome:
    def __init__(
        self,
        id: int,
        nodes: dict[int, NodeGene] = None,
        edges: dict[int, EdgeGene] = None,
        inputs: int = None,
        outputs: int = None,
        hidden: int = None,
        num_layers: int = 0
    ):
        self.id = id

        if nodes is None and edges is None:
            connectivity_ratio = 0.5
            nodes = {}
            edges = {}

            node_id = 0
            edge_id = 0

            # Create input, hidden, and output node IDs
            input_ids = []
            hidden_ids = []
            output_ids = []

            for _ in range(inputs):
                nodes[node_id] = NodeGene(node_id, "input")
                input_ids.append(node_id)
                node_id += 1

            for _ in range(hidden):
                nodes[node_id] = NodeGene(node_id, 1)
                hidden_ids.append(node_id)
                node_id += 1

            for _ in range(outputs):
                nodes[node_id] = NodeGene(node_id, "output")
                output_ids.append(node_id)
                node_id += 1

            # Calculate number of layers (input, hidden, output = 3)
            self.num_layers = 3

            # Create edges between input and hidden nodes
            for i in input_ids:
                connectable_hidden = random.sample(
                    hidden_ids, random.randint(int(hidden * connectivity_ratio), hidden)
                )
                for h in connectable_hidden:
                    weight = np.random.uniform(-1, 1)
                    edges[edge_id] = EdgeGene(edge_id, i, h, weight)
                    edge_id += 1

            # Create edges between hidden and output nodes
            for h in hidden_ids:
                connectable_output = random.sample(
                    output_ids, random.randint(int(outputs * connectivity_ratio), outputs)
                )
                for o in connectable_output:
                    weight = np.random.uniform(-1, 1)
                    edges[edge_id] = EdgeGene(edge_id, h, o, weight)
                    edge_id += 1

        self.nodes = nodes
        self.edges = edges
        if not hasattr(self, "num_layers"):
            self.num_layers = num_layers

    def show(
        self, width_scale: float = 3.0, min_width: float = 0.5, save: bool = False
    ):
        def create_directed_graph(c: Chromosome):
            g = nx.DiGraph()
            for node in c.nodes.values():
                g.add_node(node.id, layer=node.layer)
            for edge in c.edges.values():
                g.add_edge(edge.source, edge.target, weight=edge.weight)
            return g

        def plot_chromosome(
            c: Chromosome,
            width_scale: float = 3.0,
            min_width: float = 0.5,
            save: bool = False,
        ):
            g = create_directed_graph(c)
            pos = nx.multipartite_layout(
                g, subset_key="layer", align="vertical", scale=1, center=None
            )
            edge_weights = [g[u][v]["weight"] for u, v in g.edges()]
            edge_widths = [
                max(min_width, abs(weight) * width_scale) for weight in edge_weights
            ]
            edge_colors = ["green" if weight >= 0 else "red" for weight in edge_weights]
            nx.draw_networkx_nodes(g, pos, node_size=400)
            nx.draw_networkx_labels(g, pos, font_size=10)
            nx.draw_networkx_edges(
                g, pos, edgelist=g.edges(), width=edge_widths, edge_color=edge_colors
            )

            if save:
                path_name = "figs"
                os.makedirs(path_name, exist_ok=True)
                fig_name = f"chromosome-{self.id}.svg"
                full_path = os.path.join(path_name, fig_name)
                plt.savefig(full_path, format="svg", dpi=1200)
                plt.close()
            else:
                plt.show()

        plot_chromosome(self, width_scale, min_width, save)
