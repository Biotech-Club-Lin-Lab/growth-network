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
        nodes_df: pd.DataFrame = None,
        edges_df: pd.DataFrame = None,
        inputs: int = None,
        outputs: int = None,
        hidden: int = None,
    ):
        self.id = id

        if nodes_df is None and edges_df is None:
            # Create empty DataFrames with the required columns
            self.nodes = pd.DataFrame(columns=['id', 'layer', 'layer_id', 'bias'])
            self.edges = pd.DataFrame(columns=['id', 'source', 'target', 'weight', 'enabled'])
            
            connectivity_ratio = 0.5
            node_count = 0
            edge_count = 0
            
            # Create iterables
            input_ids = range(inputs)
            hidden_ids = range(inputs, inputs + hidden)
            output_ids = range(inputs + hidden, inputs + hidden + outputs)
            
            # Create input nodes
            for i in range(inputs):
                self.nodes.loc[node_count] = {
                    'id': node_count,
                    'layer': 'input',
                    'layer_id': i,
                    'bias': 0.0  # Input nodes typically don't have bias
                }
                node_count += 1
            
            # Create hidden nodes
            for h in range(hidden):
                self.nodes.loc[node_count] = {
                    'id': node_count,
                    'layer': 1,  # Using integer for hidden layers
                    'layer_id': h,
                    'bias': np.random.uniform(-1, 1)
                }
                node_count += 1
            
            # Create output nodes
            for o in range(outputs):
                self.nodes.loc[node_count] = {
                    'id': node_count,
                    'layer': 'output',
                    'layer_id': o,
                    'bias': np.random.uniform(-1, 1)
                }
                node_count += 1
            
            # Create edges between input and hidden nodes
            for i in input_ids:
                connectable_hidden = random.sample(
                    list(hidden_ids), 
                    random.randint(int(hidden * connectivity_ratio), hidden)
                )
                for h in connectable_hidden:
                    self.edges.loc[edge_count] = {
                        'id': edge_count,
                        'source': i,
                        'target': h,
                        'weight': np.random.uniform(-1, 1),
                        'enabled': True
                    }
                    edge_count += 1
            
            # Create edges between hidden and output nodes
            for h in hidden_ids:
                connectable_output = random.sample(
                    list(output_ids),
                    random.randint(int(outputs * connectivity_ratio), outputs)
                )
                for o in connectable_output:
                    self.edges.loc[edge_count] = {
                        'id': edge_count,
                        'source': h,
                        'target': o,
                        'weight': np.random.uniform(-1, 1),
                        'enabled': True
                    }
                    edge_count += 1
        else:
            # Use provided DataFrames
            self.nodes = nodes_df
            self.edges = edges_df
        
        # Calculate number of layers dynamically
        self.num_layers = len(self.nodes['layer'].unique())
    
    # Node-related methods
    def get_node_by_id(self, node_id):
        """Get node by its ID"""
        result = self.nodes[self.nodes['id'] == node_id]
        return result.iloc[0] if not result.empty else None
    
    def get_nodes_by_layer(self, layer):
        """Get all nodes in a specific layer"""
        return self.nodes[self.nodes['layer'] == layer]
    
    def get_node_by_layer_and_layer_id(self, layer, layer_id):
        """Get node by its layer and layer_id"""
        result = self.nodes[(self.nodes['layer'] == layer) & (self.nodes['layer_id'] == layer_id)]
        return result.iloc[0] if not result.empty else None
    
    def get_input_nodes(self):
        """Get all input nodes"""
        return self.nodes[self.nodes['layer'] == 'input']
    
    def get_output_nodes(self):
        """Get all output nodes"""
        return self.nodes[self.nodes['layer'] == 'output']
    
    def get_hidden_nodes(self):
        """Get all hidden nodes (anything not input or output)"""
        return self.nodes[(self.nodes['layer'] != 'input') & (self.nodes['layer'] != 'output')]
    
    # Edge-related methods
    def get_edge_by_id(self, edge_id):
        """Get edge by its ID"""
        result = self.edges[self.edges['id'] == edge_id]
        return result.iloc[0] if not result.empty else None
    
    def get_edges_by_source(self, source_id):
        """Get all edges from a specific source node"""
        return self.edges[self.edges['source'] == source_id]
    
    def get_edges_by_target(self, target_id):
        """Get all edges to a specific target node"""
        return self.edges[self.edges['target'] == target_id]
    
    def get_edges_between(self, source_id, target_id):
        """Get all edges between specific source and target nodes"""
        mask = (self.edges['source'] == source_id) & (self.edges['target'] == target_id)
        return self.edges[mask]
    
    def get_enabled_edges(self):
        """Get all enabled edges"""
        return self.edges[self.edges['enabled'] == True]
    
    # Network operations
    def add_node(self, layer, bias=0.0):
        """Add a new node to the network"""
        # Get new node_id
        new_node_id = int(self.nodes['id'].max() + 1) if len(self.nodes) > 0 else 0
        
        # Get new layer_id
        layer_nodes = self.get_nodes_by_layer(layer)
        new_layer_id = int(layer_nodes['layer_id'].max() + 1) if len(layer_nodes) > 0 else 0
        
        # Create new node row
        new_node = pd.DataFrame({
            'id': [new_node_id],
            'layer': [layer],
            'layer_id': [new_layer_id],
            'bias': [bias]
        })
        
        # Append to nodes DataFrame
        self.nodes = pd.concat([self.nodes, new_node], ignore_index=True)
        
        return new_node_id
    
    def add_edge(self, source, target, weight=None, enabled=True):
        """Add a new edge to the network"""
        # Generate weight if not provided
        if weight is None:
            weight = np.random.uniform(-1, 1)
        
        # Get new edge_id
        new_edge_id = int(self.edges['id'].max() + 1) if len(self.edges) > 0 else 0
        
        # Create new edge row
        new_edge = pd.DataFrame({
            'id': [new_edge_id],
            'source': [source],
            'target': [target],
            'weight': [weight],
            'enabled': [enabled]
        })
        
        # Append to edges DataFrame
        self.edges = pd.concat([self.edges, new_edge], ignore_index=True)
        
        return new_edge_id
    
    def disable_edge(self, edge_id):
        """Disable an edge"""
        idx = self.edges.index[self.edges['id'] == edge_id].tolist()
        if idx:
            self.edges.loc[idx[0], 'enabled'] = False
            return True
        return False
    
    def update_node_bias(self, node_id, new_bias):
        """Update a node's bias"""
        idx = self.nodes.index[self.nodes['id'] == node_id].tolist()
        if idx:
            self.nodes.loc[idx[0], 'bias'] = new_bias
            return True
        return False
    
    def update_edge_weight(self, edge_id, new_weight):
        """Update an edge's weight"""
        idx = self.edges.index[self.edges['id'] == edge_id].tolist()
        if idx:
            self.edges.loc[idx[0], 'weight'] = new_weight
            return True
        return False
    
    def to_networkx(self):
        """Convert chromosome to a NetworkX graph for visualization"""
        G = nx.DiGraph()
        
        # Add nodes with their attributes
        for _, node in self.nodes.iterrows():
            G.add_node(
                node['id'], 
                layer=node['layer'],
                layer_id=node['layer_id'],
                bias=node['bias']
            )
        
        # Add edges with their attributes
        for _, edge in self.edges.iterrows():
            if edge['enabled']:
                G.add_edge(
                    edge['source'],
                    edge['target'],
                    weight=edge['weight'],
                    id=edge['id']
                )
        
        return G
    
    def visualize(self, ax=None, figsize=(10, 8)):
        """Visualize the network using NetworkX"""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        G = self.to_networkx()
        
        # Position nodes in layers
        pos = {}
        layers = {}
        
        # Group nodes by layer
        for _, node in self.nodes.iterrows():
            layer = node['layer']
            if layer not in layers:
                layers[layer] = []
            layers[layer].append(node['id'])
        
        # Sort layers
        sorted_layers = []
        if 'input' in layers:
            sorted_layers.append(('input', layers['input']))
        
        # Add numeric layers in order
        numeric_layers = [layer for layer in layers if isinstance(layer, (int, float))]
        for layer in sorted(numeric_layers):
            sorted_layers.append((layer, layers[layer]))
        
        if 'output' in layers:
            sorted_layers.append(('output', layers['output']))
        
        # Position nodes
        for i, (layer, nodes_in_layer) in enumerate(sorted_layers):
            layer_size = len(nodes_in_layer)
            for j, node_id in enumerate(nodes_in_layer):
                pos[node_id] = (i, (j - layer_size/2)/max(1, layer_size-1)*2)
        
        # Draw the network
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', node_size=500)
        
        edge_colors = []
        edge_widths = []
        
        for u, v, data in G.edges(data=True):
            weight = data['weight']
            edge_colors.append('red' if weight < 0 else 'green')
            edge_widths.append(abs(weight) * 2)
        
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=edge_widths, 
                               arrowsize=15, connectionstyle='arc3,rad=0.1')
        
        nx.draw_networkx_labels(G, pos, ax=ax)
        
        if ax is None:
            plt.title(f"Chromosome {self.id}")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        return ax
