from genes import *
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import pandas as pd


class Chromosome:
    def __init__(
        self,
        id: int,
        nodes: dict[int, NodeGene] = None,
        edges: dict[int, EdgeGene] = None,
        inputs: int = None,
        outputs: int = None,
        hidden: int = None,
    ):
        self.id = id

        if nodes is None and edges is None:
            connectivity_ratio = 0.5
            nodes = {}
            edges = {}

            # Create iterables
            input_ids = range(inputs)
            hidden_ids = range(inputs, inputs + hidden)
            output_ids = range(inputs + hidden, inputs + hidden + outputs)
            count = 0

            # Create nodes
            for _ in range(inputs):
                nodes[_] = NodeGene(len(nodes), "input")
            for _ in range(hidden):
                nodes[inputs + _] = NodeGene(len(nodes), 1)
            for _ in range(outputs):
                nodes[inputs + hidden + _] = NodeGene(len(nodes), "output")

            # Create edges between input and hidden nodes
            for i in input_ids:
                connectable_hidden = random.sample(
                    hidden_ids, random.randint(int(hidden * connectivity_ratio), hidden)
                )
                for h in connectable_hidden:
                    weight = np.random.uniform(-1, 1)
                    edges[count] = EdgeGene(len(edges), i, h, weight)
                    count += 1

            # Create edges between hidden and output nodes
            for h in hidden_ids:
                connectable_output = random.sample(
                    output_ids,
                    random.randint(int(outputs * connectivity_ratio), outputs),
                )
                for o in connectable_output:
                    weight = np.random.uniform(-1, 1)
                    edges[count] = EdgeGene(len(edges), h, o, weight)
                    count += 1

        self.nodes = nodes
        self.edges = edges

        # Convert nodes to DataFrame for efficient querying
        nodes_data = []
        for node_id, node in nodes.items():
            nodes_data.append({
                "node_id":node_id,
                "layer":node.layer
            })
        self.nodes_df = pd.DataFrame(nodes_data)

        # Convert edges to DataFrame for efficient querying
        edges_data = []
        for edge_id, edge in edges.items():
            edges_data.append({
                'edge_id':edge_id,
                'source':edge.source,
                'target':edge.target,
                'weight':edge.weight,
                'enabled':edge.enabled
            })
        self.edges_df = pd.DataFrame(edges_data)

        # Create indices for faster lookups - don't set inplace to avoid modifying original df
        self.nodes_by_id = self.nodes_df.set_index('node_id')

        # Create multiple indexing structures for edges
        self.edges_by_id = self.edges_df.set_index('edge_id')
        self.edges_by_source = self.edges_df.set_index('source')
        self.edges_by_target = self.edges_df.set_index('target')

        # Calculate number of layers dynamically
        layers = set(node.layer for node in self.nodes.values())
        self.num_layers = len(layers)
    
    def get_node_by_id(self, node_id):
        """Get node gene by its ID"""
        return self.nodes.get(node_id)
    
    def get_node_by_layer(self, layer):
        """Get all nodes in a specific layer"""
        #Query the data frame
        node_ids = self.nodes_df[self.nodes_df['layer']==layer]['node_id'].tolist()
        #Return the actual NodeGene objects
        return {node_id: self.nodes[node_id] for node_id in node_ids}

    def get_edge_by_id(self, edge_id):
        """Get edge gene by its ID"""
        return self.edges.get(edge_id)
    
    def get_edges_by_source(self, source_id):
        """Get all edges from a specific source node"""
        try:
            edge_ids = self.edges_df[self.edges_df['source']==source_id]['edge_id']
            return {edge_id: self.edges[edge_id] for edge_id in edge_ids}
        except KeyError:
            return {}
    
    def get_edges_by_target(self, target_id):
        """Get all edges to a specific target node"""
        try:
            edge_ids = self.edges_df[self.edges_df['target']==target_id]['edge_id']
            return {edge_id: self.edges[edge_id] for edge_id in edge_ids}
        except KeyError:
            return {}

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
