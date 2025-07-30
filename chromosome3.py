import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd


class Chromosome:
    def __init__(
        self,
        id: int,
        nodes_df: pd.DataFrame = None,
        edges_df: pd.DataFrame = None,
        inputs: int = None,
        outputs: int = None,
        hidden_nodes: int = None,  # Number of hidden nodes to create
        connectivity_ratio: float = 0.5,  # Ratio of possible connections to create
        generation: int = 0,       # Track the generation of this chromosome
    ):
        self.id = id
        self.generation = generation

        if nodes_df is None and edges_df is None:
            # Create empty DataFrames with the required columns
            self.nodes = pd.DataFrame(columns=['id', 'type', 'bias', 'birth_generation', 'division_count'])
            self.edges = pd.DataFrame(columns=['id', 'source', 'target', 'weight', 'enabled'])
            
            if connectivity_ratio is None:
                connectivity_ratio = 0.5
                
            node_count = 0
            edge_count = 0
            
            # If hidden_nodes not specified, default to a reasonable number
            if hidden_nodes is None:
                hidden_nodes = 10  # Default number of hidden nodes
            
            # Create input nodes
            input_start_id = node_count
            for i in range(inputs):
                self.nodes.loc[node_count] = {
                    'id': node_count,
                    'type': 'input',
                    'bias': 0.0,  # Input nodes typically don't have bias
                    'birth_generation': generation,
                    'division_count': 0
                }
                node_count += 1
            
            # Create hidden nodes
            hidden_start_id = node_count
            for h in range(hidden_nodes):
                self.nodes.loc[node_count] = {
                    'id': node_count,
                    'type': 'hidden',
                    'bias': np.random.uniform(-1, 1),
                    'birth_generation': generation,
                    'division_count': 0
                }
                node_count += 1
            
            # Create output nodes
            output_start_id = node_count
            for o in range(outputs):
                self.nodes.loc[node_count] = {
                    'id': node_count,
                    'type': 'output',
                    'bias': np.random.uniform(-1, 1),
                    'birth_generation': generation,
                    'division_count': 0
                }
                node_count += 1
            
            # Create random edges ensuring no cycles
            input_ids = list(range(input_start_id, hidden_start_id))
            hidden_ids = list(range(hidden_start_id, output_start_id))
            output_ids = list(range(output_start_id, node_count))
            
            # Function to check if adding an edge would create a cycle
            def would_create_cycle(source, target, existing_edges):
                """Check if adding edge from source to target would create a cycle"""
                # Create a directed graph with existing edges
                G = nx.DiGraph()
                for _, edge in existing_edges.iterrows():
                    if edge['enabled']:
                        G.add_edge(edge['source'], edge['target'])
                
                # Add the proposed edge
                G.add_edge(source, target)
                
                # Check for cycles
                try:
                    nx.find_cycle(G)
                    return True  # Found a cycle
                except nx.NetworkXNoCycle:
                    return False  # No cycle found
            
            # Connect inputs to some hidden nodes
            for i in input_ids:
                # Determine how many connections to create
                num_connections = max(1, int(connectivity_ratio * len(hidden_ids)))
                targets = random.sample(hidden_ids, num_connections)
                
                for t in targets:
                    self.edges.loc[edge_count] = {
                        'id': edge_count,
                        'source': i,
                        'target': t,
                        'weight': np.random.uniform(-1, 1),
                        'enabled': True
                    }
                    edge_count += 1
            
            # Connect hidden nodes to output nodes
            for h in hidden_ids:
                # Determine how many output connections to create
                num_connections = max(1, int(connectivity_ratio * len(output_ids)))
                targets = random.sample(output_ids, num_connections)
                
                for t in targets:
                    self.edges.loc[edge_count] = {
                        'id': edge_count,
                        'source': h,
                        'target': t,
                        'weight': np.random.uniform(-1, 1),
                        'enabled': True
                    }
                    edge_count += 1
            
            # Add some additional hidden-to-hidden connections
            possible_h2h = [(src, tgt) for src in hidden_ids for tgt in hidden_ids if src != tgt]
            random.shuffle(possible_h2h)
            
            # Try to add some random hidden-to-hidden connections
            for src, tgt in possible_h2h[:int(connectivity_ratio * len(possible_h2h))]:
                # Check if this edge would create a cycle
                temp_df = pd.DataFrame(self.edges)
                if not would_create_cycle(src, tgt, temp_df):
                    self.edges.loc[edge_count] = {
                        'id': edge_count,
                        'source': src,
                        'target': tgt,
                        'weight': np.random.uniform(-1, 1),
                        'enabled': True
                    }
                    edge_count += 1
        else:
            # Use provided DataFrames
            self.nodes = nodes_df.copy()
            self.edges = edges_df.copy()
        
        # Initialize the NN instance for running the network
        self.NN = self.NN(self)
    
    # Node-related methods
    def get_node_by_id(self, node_id):
        """Get node by its ID"""
        result = self.nodes[self.nodes['id'] == node_id]
        return result.iloc[0] if not result.empty else None
    
    def get_nodes_by_type(self, node_type):
        """Get all nodes of a specific type"""
        return self.nodes[self.nodes['type'] == node_type]
    
    def get_input_nodes(self):
        """Get all input nodes"""
        return self.nodes[self.nodes['type'] == 'input']
    
    def get_output_nodes(self):
        """Get all output nodes"""
        return self.nodes[self.nodes['type'] == 'output']
    
    def get_hidden_nodes(self):
        """Get all hidden nodes"""
        return self.nodes[self.nodes['type'] == 'hidden']
    
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
    
    def would_create_cycle(self, source, target):
        """Check if adding edge from source to target would create a cycle"""
        # Create a directed graph with existing edges
        G = nx.DiGraph()
        for _, edge in self.edges.iterrows():
            if edge['enabled']:
                G.add_edge(edge['source'], edge['target'])
        
        # Add the proposed edge
        G.add_edge(source, target)
        
        # Check for cycles
        try:
            nx.find_cycle(G)
            return True  # Found a cycle
        except nx.NetworkXNoCycle:
            return False  # No cycle found
            
    # Network operations
    def add_node(self, node_type, bias=None, birth_generation=None):
        """Add a new node to the network"""
        # Get new node_id
        new_node_id = int(self.nodes['id'].max() + 1) if len(self.nodes) > 0 else 0
        
        # Set bias based on node type
        if bias is None:
            if node_type == 'input':
                bias = 0.0
            else:
                bias = np.random.uniform(-1, 1)
        
        # If birth_generation not provided, use current generation
        if birth_generation is None:
            birth_generation = self.generation
        
        # Create new node row
        new_node = pd.DataFrame({
            'id': [new_node_id],
            'type': [node_type],
            'bias': [bias],
            'birth_generation': [birth_generation],
            'division_count': [0]
        })
        
        # Append to nodes DataFrame
        self.nodes = pd.concat([self.nodes, new_node], ignore_index=True)
        
        return new_node_id
    
    def add_edge(self, source, target, weight=None, enabled=True):
        """
        Add a new edge to the network if it doesn't create a cycle
        
        Returns:
            int or None: edge_id if added, None if would create cycle
        """
        # Check if adding this edge would create a cycle
        if self.would_create_cycle(source, target):
            return None
            
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
    
    def get_connection_count(self, node_id):
        """Get the total number of connections for a node (both incoming and outgoing)"""
        incoming = len(self.get_edges_by_target(node_id))
        outgoing = len(self.get_edges_by_source(node_id))
        return incoming + outgoing
        
    def evaluate_division(self, node_id, division_model):
        """
        Evaluate whether a node should divide based on input from another neural network
        with added randomness to ensure diverse behavior across nodes
        """
        node = self.get_node_by_id(node_id)
        
        # Only hidden nodes can divide
        if node['type'] != 'hidden':
            return False
        
        # Calculate node age
        node_age = self.generation - node['birth_generation']
        
        # Get division count
        division_count = node['division_count']
        
        # Get connection count
        connection_count = self.get_connection_count(node_id)
        
        # Normalize inputs
        max_age = 20  # Arbitrary maximum age for normalization
        max_division = 5  # Arbitrary maximum divisions for normalization
        max_connections = 20  # Arbitrary maximum connections for normalization
        max_node_id = 100  # Arbitrary maximum node ID for normalization
        
        normalized_age = node_age / max_age
        normalized_division = division_count / max_division
        normalized_connections = connection_count / max_connections
        normalized_node_id = node_id / max_node_id  # Normalize node ID
        
        # Add random noise to inputs to create variability across nodes
        noise_range = 0.3  # Controls how much randomness to add
        noisy_age = normalized_age + random.uniform(-noise_range, noise_range)
        noisy_division = normalized_division + random.uniform(-noise_range, noise_range)
        noisy_connections = normalized_connections + random.uniform(-noise_range, noise_range)
        noisy_node_id = normalized_node_id + random.uniform(-noise_range, noise_range)
        
        # Use both original inputs and noisy inputs
        inputs = [noisy_age, noisy_division, noisy_connections, noisy_node_id]
        output = division_model.NN.run(inputs)
        
        # Add some node-specific randomness to the output to further diversify results
        node_specific_factor = random.uniform(0.7, 1.3)
        modified_output = output[0] * node_specific_factor
        
        # Use a random threshold
        threshold = random.uniform(0.2, 0.5)  # Balanced range to get mixed results
        
        # Decision is based on the modified output value with random threshold
        return modified_output > threshold
    
    def perform_division(self, node_id, bias_func=None):
        """
        Perform division of a node, creating a single daughter node
        
        Args:
            node_id: ID of the parent node to divide
            bias_func: Function to calculate bias of daughter node based on parent node
                    If None, uses a small random variation of parent's bias
                    
        Returns:
            list: ID of the newly created daughter node in a list
        """
        parent_node = self.get_node_by_id(node_id)
        
        # Only hidden nodes can divide
        if parent_node['type'] != 'hidden':
            return []
        
        # Increment division count of parent node
        parent_idx = self.nodes.index[self.nodes['id'] == node_id].tolist()[0]
        self.nodes.at[parent_idx, 'division_count'] += 1
        
        # Default bias function: random variation of parent bias
        if bias_func is None:
            def bias_func(parent_bias):
                return parent_bias + np.random.uniform(-0.2, 0.2)
        
        # Create a daughter node of the same type
        daughter_id = self.add_node(
            node_type='hidden',
            bias=bias_func(parent_node['bias']),
            birth_generation=self.generation
        )
        
        # Connect parent to daughter if it doesn't create a cycle
        if not self.would_create_cycle(node_id, daughter_id):
            self.add_edge(source=node_id, target=daughter_id)
        
        # Connect incoming edges to daughter
        for _, edge in self.get_edges_by_target(node_id).iterrows():
            # Skip self-loops and connections from parent (to avoid creating loops)
            if edge['source'] != node_id and edge['source'] != daughter_id:
                if not self.would_create_cycle(edge['source'], daughter_id):
                    self.add_edge(source=edge['source'], target=daughter_id, weight=edge['weight'])
        
        # Connect daughter to outgoing edges
        for _, edge in self.get_edges_by_source(node_id).iterrows():
            # Skip self-loops and connections to parent (to avoid creating loops)
            if edge['target'] != node_id and edge['target'] != daughter_id:
                if not self.would_create_cycle(daughter_id, edge['target']):
                    self.add_edge(source=daughter_id, target=edge['target'], weight=edge['weight'])
        
        return [daughter_id] 
    
    def evaluate_all_nodes_for_division(self, division_model, bias_func=None):
        """
        Evaluate all hidden nodes for division and perform division if criteria are met
        
        Args:
            division_model: A Chromosome instance that acts as the division decision model
            bias_func: Function to calculate bias of daughter nodes based on parent node
            
        Returns:
            list: IDs of all newly created nodes
        """
        hidden_nodes = self.get_hidden_nodes()
        new_nodes = []
        
        # Get nodes to be evaluated (make a copy to avoid iteration issues when adding nodes)
        nodes_to_evaluate = hidden_nodes['id'].tolist()
        
        for node_id in nodes_to_evaluate:
            if self.evaluate_division(node_id, division_model):
                daughter_nodes = self.perform_division(node_id, bias_func)
                new_nodes.extend(daughter_nodes)
        
        return new_nodes
    
    def to_networkx(self):
        """Convert chromosome to a NetworkX graph for visualization"""
        G = nx.DiGraph()
        
        # Add nodes with their attributes
        for _, node in self.nodes.iterrows():
            G.add_node(
                node['id'], 
                type=node['type'],
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
        
        # Use a hierarchical layout
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        except:
            # Fall back to another layout if graphviz is not available
            pos = nx.spring_layout(G)
        
        # Draw nodes by type with different colors
        node_colors = []
        for node in G.nodes():
            node_type = G.nodes[node]['type']
            if node_type == 'input':
                node_colors.append('lightblue')
            elif node_type == 'hidden':
                node_colors.append('lightgreen')  
            elif node_type == 'output':
                node_colors.append('lightcoral')
        
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=500)
        
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
    
    class NN:
        def __init__(self, C):
            self.edges = C.edges
            self.nodes = C.nodes
            
            # Results is a dataframe that will hold the output of each node
            self.results = pd.DataFrame({'node id': C.nodes['id'].values, 'result': None})
        
        def run(self, X):
            # Set input vals
            inp_ids = self.nodes.loc[self.nodes['type'] == 'input', 'id'].values
            
            # Ensure X has the correct length
            if len(X) != len(inp_ids):
                raise ValueError(f"Input size {len(X)} doesn't match network input nodes {len(inp_ids)}")
                
            self.results.loc[self.results['node id'].isin(inp_ids), 'result'] = X
            
            # Sort nodes for efficient execution (topological sort)
            G = nx.DiGraph()
            for _, edge in self.edges.iterrows():
                if edge['enabled']:
                    G.add_edge(edge['source'], edge['target'])
            
            try:
                # Get execution order through topological sort
                execution_order = list(nx.topological_sort(G))
                
                # Skip input nodes as they're already set
                execution_order = [node_id for node_id in execution_order 
                                   if node_id not in inp_ids]
            except nx.NetworkXUnfeasible:
                # If there's a cycle, fall back to old execution method
                execution_order = self.nodes.loc[~self.nodes['id'].isin(inp_ids), 'id'].values
            
            def ReLU(x):
                return np.maximum(0, x)
            
            # Execute nodes in order
            for node_id in execution_order:
                source_edges = self.edges.loc[(self.edges['target'] == node_id) & (self.edges['enabled'] == True)]
                source_vals = self.results.loc[self.results['node id'].isin(source_edges['source'].values)]
                
                # If we have incoming edges with values
                if not source_edges.empty and not source_vals.empty:
                    if not source_vals['result'].isnull().any():
                        x = source_vals['result'].values
                        w = source_edges['weight'].values
                        b = self.nodes.loc[self.nodes['id'] == node_id, 'bias'].values[0]
                        output = ReLU(np.dot(x, w) + b)
                        self.results.loc[self.results['node id'] == node_id, 'result'] = output
            
            # Return output values
            outputs = self.results.loc[self.results['node id'].isin(
                self.nodes.loc[self.nodes['type'] == 'output', 'id'].values), 'result'].values
            # After computing outputs, fill None with 0.0
            outputs = self.results.loc[self.results['node id'].isin(
                self.nodes.loc[self.nodes['type'] == 'output', 'id'].values), 'result'].values

            # Convert to list and replace None values
            outputs = [0.0 if v is None or pd.isna(v) else v for v in outputs]

            return outputs
