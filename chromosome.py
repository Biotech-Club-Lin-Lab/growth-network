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
        nodes: list[NodeGene] = None, #make a dict where the key is the id and the value is the node
        edges: list[EdgeGene] = None, #make a dict where the key is the id and the value is the edge
        inputs: int = None,
        outputs: int = None,
        hidden: int = None,
    ):
        self.id = id

        if nodes is None and edges is None:
            connectivity_ratio = 0.5
            nodes = []
            edges = []
            # Create iterables
            input_ids = range(inputs)
            hidden_ids = range(inputs, inputs + hidden)
            output_ids = range(inputs + hidden, inputs + hidden + outputs)
            # Create nodes per the number of inputs, outputs and hidden nodes
            for _ in range(inputs):
                nodes.append(NodeGene(len(nodes), "input"))
            for _ in range(hidden):
                nodes.append(NodeGene(len(nodes), 1))
            for _ in range(outputs):
                nodes.append(NodeGene(len(nodes), "output"))
            # Create edges between input and hidden nodes
            for i in input_ids:
                connectable_hidden = random.sample(
                    hidden_ids, random.randint(int(hidden * connectivity_ratio), hidden)
                )
                for h in connectable_hidden:
                    weight = np.random.uniform(-1, 1)
                    edges.append(EdgeGene(len(edges), i, h, weight))
            # Create edges between hidden and output nodes
            for h in hidden_ids:
                connectable_output = random.sample(
                    output_ids,
                    random.randint(int(outputs * connectivity_ratio), outputs),
                )
                for o in connectable_output:
                    weight = np.random.uniform(-1, 1)
                    edges.append(EdgeGene(len(edges), h, o, weight))

        self.nodes = nodes
        self.edges = edges

    def show(
        self, width_scale: float = 3.0, min_width: float = 0.5, save: bool = False
    ):
        def create_directed_graph(c: Chromosome):
            g = nx.DiGraph()
            for node in c.nodes:
                g.add_node(node.id, layer=node.layer)
            for edge in c.edges:
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
