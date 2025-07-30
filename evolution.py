import numpy as np
import random
import gymnasium as gym
from chromosome3 import Chromosome  # Import the Chromosome3 class
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import networkx as nx


class Evolution:
    def __init__(self, population_size=20, inputs=4, outputs=1, hidden_nodes=15, connectivity_ratio=0.5):
        """
        Initialize the evolutionary process with the updated chromosome structure.
        
        Args:
            population_size: Number of chromosomes in each generation
            inputs: Number of input nodes for each neural network
            outputs: Number of output nodes for each neural network
            hidden_nodes: Number of hidden nodes to start with
            connectivity_ratio: Ratio of possible connections to create
        """
        self.population_size = population_size
        self.inputs = inputs
        self.outputs = outputs
        self.hidden_nodes = hidden_nodes
        self.connectivity_ratio = connectivity_ratio
        self.generation = 0
        self.population = []
        self.best_fitness = 0
        self.best_chromosome = None
        self.fitness_history = []
        
        # Create initial population
        for i in range(self.population_size):
            chromosome = Chromosome(
                id=i,
                inputs=self.inputs,
                outputs=self.outputs,
                hidden_nodes=self.hidden_nodes,
                connectivity_ratio=self.connectivity_ratio,
                generation=self.generation
            )
            self.population.append(chromosome)
    
    def _validate_network(self, chromosome):
        """Validate that the neural network structure is valid"""
        # Check that all input nodes have outgoing connections
        input_nodes = chromosome.get_input_nodes()['id'].values
        
        for input_id in input_nodes:
            if len(chromosome.get_edges_by_source(input_id)) == 0:
                return False
                
        # Check network can handle expected input size
        try:
            test_input = np.zeros(self.inputs)
            chromosome.NN.run(test_input)  # Should not raise errors
            return True
        except Exception as e:
            print(f"Network validation failed: {e}")
            return False
    
    def evaluate_fitness(self, chromosome, render=False):
        """
        Evaluate fitness of a chromosome using the CartPole-v1 environment.
        
        Returns:
            float: Fitness score
        """
        if not self._validate_network(chromosome):
            print(f"Warning: Invalid network structure in chromosome {chromosome.id}")
            return 0  # Return minimum fitness for invalid networks
        
        env = gym.make('CartPole-v1', render_mode='human' if render else None)
        observation, _ = env.reset()
        
        done = False
        total_reward = 0
        max_steps = 500  # Maximum episode length
        
        while not done and total_reward < max_steps:
            # Use neural network to determine action
            action = self._get_action(chromosome, observation)
            
            # Take step in environment
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        env.close()
        return total_reward
    
    def _get_action(self, chromosome, observation):
        """
        Use the neural network to determine the action.
        
        Args:
            chromosome: The neural network chromosome
            observation: The current state observation
            
        Returns:
            int: 0 or 1 (left or right)
        """
        # Normalize observation values to proper range for neural network
        norm_obs = observation  # CartPole values are already in a reasonable range
        
        # Get output from neural network
        output = chromosome.NN.run(norm_obs)
        
        # Convert output to discrete action (0 or 1)
        return 1 if output[0] > 0.5 else 0
    
    def evaluate_population(self):
        """
        Evaluate fitness of entire population.
        
        Returns:
            list: List of (chromosome, fitness) tuples
        """
        fitness_scores = []
        
        for chromosome in tqdm(self.population, desc=f"Generation {self.generation}"):
            fitness = self.evaluate_fitness(chromosome)
            fitness_scores.append((chromosome, fitness))
            
        # Sort by fitness (descending)
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Update best chromosome if found
        if fitness_scores[0][1] > self.best_fitness:
            self.best_fitness = fitness_scores[0][1]
            self.best_chromosome = fitness_scores[0][0]
        
        # Record average fitness
        avg_fitness = sum(score for _, score in fitness_scores) / len(fitness_scores)
        self.fitness_history.append({
            'generation': self.generation,
            'max_fitness': fitness_scores[0][1],
            'avg_fitness': avg_fitness
        })
        
        return fitness_scores
    
    def select_parents(self, fitness_scores, selection_method='tournament', tournament_size=3):
        """
        Select parents for reproduction based on fitness.
        
        Args:
            fitness_scores: List of (chromosome, fitness) tuples
            selection_method: 'tournament' or 'roulette'
            tournament_size: Size of tournament for tournament selection
            
        Returns:
            list: Selected parent chromosomes
        """
        parents = []
        
        if selection_method == 'tournament':
            # Tournament selection
            for _ in range(self.population_size):
                tournament = random.sample(fitness_scores, tournament_size)
                winner = max(tournament, key=lambda x: x[1])
                parents.append(winner[0])
                
        elif selection_method == 'roulette':
            # Roulette wheel selection
            total_fitness = sum(score for _, score in fitness_scores)
            
            # Handle case where all fitnesses are zero
            if total_fitness == 0:
                return [chromosome for chromosome, _ in random.sample(fitness_scores, self.population_size)]
            
            for _ in range(self.population_size):
                pick = random.uniform(0, total_fitness)
                current = 0
                for chromosome, score in fitness_scores:
                    current += score
                    if current > pick:
                        parents.append(chromosome)
                        break
                # Ensure we always select someone
                if len(parents) < _ + 1:
                    parents.append(fitness_scores[0][0])
        
        return parents
    
    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parent chromosomes, focusing on edges.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            
        Returns:
            Chromosome: Child chromosome
        """
        # First, get all nodes from both parents
        # We'll use parent1's nodes as the base and add any unique nodes from parent2 that are needed
        all_nodes = parent1.nodes.copy()
        
        # Create empty DataFrame for child edges
        edge_records = []
        
        # Combine edges from both parents with probability
        parent1_edges = parent1.edges.copy()
        parent2_edges = parent2.edges.copy()
        
        # Group edges by target node for easier handling
        p1_edges_by_target = parent1_edges.groupby('target')
        p2_edges_by_target = parent2_edges.groupby('target')
        
        # Get all target nodes from both parents
        all_targets = set(parent1_edges['target'].unique()) | set(parent2_edges['target'].unique())
        
        # For each target node, randomly choose which parent's incoming edges to use
        for target in all_targets:
            # Choose parent with probability based on their fitness ratio (here we use 50/50)
            chosen_parent = random.choice([1, 2])
            
            if chosen_parent == 1 and target in p1_edges_by_target.groups:
                # Use parent1's edges to this target
                for _, edge in p1_edges_by_target.get_group(target).iterrows():
                    edge_records.append({
                        'id': len(edge_records),
                        'source': edge['source'],
                        'target': edge['target'],
                        'weight': edge['weight'],
                        'enabled': edge['enabled']
                    })
            elif chosen_parent == 2 and target in p2_edges_by_target.groups:
                # Use parent2's edges to this target
                for _, edge in p2_edges_by_target.get_group(target).iterrows():
                    # Ensure that the source node exists in our node list
                    source_node = edge['source']
                    
                    # If the source node is not in all_nodes, add it from parent2
                    if source_node not in all_nodes['id'].values:
                        source_node_data = parent2.nodes[parent2.nodes['id'] == source_node]
                        if not source_node_data.empty:
                            source_node_row = source_node_data.iloc[0].copy()
                            # Make sure we're not duplicating IDs
                            if source_node in all_nodes['id'].values:
                                source_node = max(all_nodes['id']) + 1
                                source_node_row['id'] = source_node
                            all_nodes = pd.concat([all_nodes, pd.DataFrame([source_node_row])], ignore_index=True)
                    
                    edge_records.append({
                        'id': len(edge_records),
                        'source': source_node,
                        'target': edge['target'],
                        'weight': edge['weight'],
                        'enabled': edge['enabled']
                    })
        
        # Create the child edges DataFrame
        child_edges = pd.DataFrame(edge_records) if edge_records else pd.DataFrame(columns=['id', 'source', 'target', 'weight', 'enabled'])
        
        # Ensure the child has all the necessary node types
        # Check for input nodes
        input_nodes = all_nodes[all_nodes['type'] == 'input']
        if len(input_nodes) < self.inputs:
            # Add missing input nodes from parent2
            parent2_input_nodes = parent2.nodes[parent2.nodes['type'] == 'input']
            for i in range(self.inputs - len(input_nodes)):
                if i < len(parent2_input_nodes):
                    node_row = parent2_input_nodes.iloc[i].copy()
                    # Make sure we're not duplicating IDs
                    if node_row['id'] in all_nodes['id'].values:
                        node_row['id'] = max(all_nodes['id']) + 1
                    all_nodes = pd.concat([all_nodes, pd.DataFrame([node_row])], ignore_index=True)
        
        # Check for output nodes
        output_nodes = all_nodes[all_nodes['type'] == 'output']
        if len(output_nodes) < self.outputs:
            # Add missing output nodes from parent2
            parent2_output_nodes = parent2.nodes[parent2.nodes['type'] == 'output']
            for i in range(self.outputs - len(output_nodes)):
                if i < len(parent2_output_nodes):
                    node_row = parent2_output_nodes.iloc[i].copy()
                    # Make sure we're not duplicating IDs
                    if node_row['id'] in all_nodes['id'].values:
                        node_row['id'] = max(all_nodes['id']) + 1
                    all_nodes = pd.concat([all_nodes, pd.DataFrame([node_row])], ignore_index=True)
        
        # Ensure all input nodes have at least one outgoing connection
        input_nodes = all_nodes[all_nodes['type'] == 'input']
        for _, input_node in input_nodes.iterrows():
            input_id = input_node['id']
            if not any(child_edges['source'] == input_id):
                # Find a suitable target node (preferably hidden)
                hidden_nodes = all_nodes[all_nodes['type'] == 'hidden']
                if not hidden_nodes.empty:
                    target_id = hidden_nodes.sample(1).iloc[0]['id']
                else:
                    # If no hidden nodes, connect to an output node
                    output_nodes = all_nodes[all_nodes['type'] == 'output']
                    target_id = output_nodes.sample(1).iloc[0]['id']
                
                # Add a new edge with random weight
                child_edges = pd.concat([child_edges, pd.DataFrame([{
                    'id': len(child_edges),
                    'source': input_id,
                    'target': target_id,
                    'weight': np.random.uniform(-1, 1),
                    'enabled': True
                }])], ignore_index=True)
        
        # Ensure all output nodes have at least one incoming connection
        output_nodes = all_nodes[all_nodes['type'] == 'output']
        for _, output_node in output_nodes.iterrows():
            output_id = output_node['id']
            if not any(child_edges['target'] == output_id):
                # Find a suitable source node (preferably hidden)
                hidden_nodes = all_nodes[all_nodes['type'] == 'hidden']
                if not hidden_nodes.empty:
                    source_id = hidden_nodes.sample(1).iloc[0]['id']
                else:
                    # If no hidden nodes, connect from an input node
                    input_nodes = all_nodes[all_nodes['type'] == 'input']
                    source_id = input_nodes.sample(1).iloc[0]['id']
                
                # Add a new edge with random weight
                child_edges = pd.concat([child_edges, pd.DataFrame([{
                    'id': len(child_edges),
                    'source': source_id,
                    'target': output_id,
                    'weight': np.random.uniform(-1, 1),
                    'enabled': True
                }])], ignore_index=True)
        
        # Create new chromosome ID
        new_id = max(parent1.id, parent2.id) + 1
        
        # Create and return the child chromosome
        child = Chromosome(
            id=new_id,
            nodes_df=all_nodes,
            edges_df=child_edges,
            generation=self.generation + 1
        )
        
        return child
    
    def mutate(self, chromosome, mutation_rate=0.1, weight_mutation_scale=0.5, bias_mutation_scale=0.2):
        """
        Mutate a chromosome.
        
        Args:
            chromosome: Chromosome to mutate
            mutation_rate: Probability of each weight/bias being mutated
            weight_mutation_scale: Scale of weight mutations
            bias_mutation_scale: Scale of bias mutations
            
        Returns:
            Chromosome: Mutated chromosome
        """
        # Mutate edge weights
        for index, edge in chromosome.edges.iterrows():
            if random.random() < mutation_rate:
                new_weight = edge['weight'] + np.random.normal(0, weight_mutation_scale)
                chromosome.update_edge_weight(edge['id'], new_weight)
        
        # Mutate node biases (except input nodes)
        non_input_nodes = chromosome.nodes[chromosome.nodes['type'] != 'input']
        for index, node in non_input_nodes.iterrows():
            if random.random() < mutation_rate:
                new_bias = node['bias'] + np.random.normal(0, bias_mutation_scale)
                chromosome.update_node_bias(node['id'], new_bias)
        
        # Structural mutation: enable/disable edges
        if random.random() < mutation_rate * 0.5:  # Lower probability for structural mutations
            # Get a small subset of edges to potentially toggle
            if not chromosome.edges.empty:
                num_edges_to_toggle = max(1, int(len(chromosome.edges) * 0.1))
                edges_to_toggle = chromosome.edges.sample(min(num_edges_to_toggle, len(chromosome.edges)))
                
                for _, edge in edges_to_toggle.iterrows():
                    # Toggle the edge (but ensure we don't disable critical paths)
                    if edge['enabled']:
                        # Only disable if it won't disconnect input->output path
                        # Simple heuristic: make sure target node has other incoming edges
                        target_edges = chromosome.get_edges_by_target(edge['target'])
                        if len(target_edges) > 1:
                            chromosome.disable_edge(edge['id'])
                    else:
                        # Re-enable disabled edge
                        idx = chromosome.edges.index[chromosome.edges['id'] == edge['id']].tolist()
                        if idx:
                            chromosome.edges.loc[idx[0], 'enabled'] = True
        
        # Structural mutation: add new edge
        if random.random() < mutation_rate * 0.3:  # Even lower probability for adding edges
            # Get available nodes
            node_ids = chromosome.nodes['id'].values
            
            # Try multiple times to find a valid connection
            for _ in range(5):  # Try up to 5 times
                # Select random source and target
                source = random.choice(node_ids)
                target = random.choice(node_ids)
                
                # Skip if source is output or target is input (invalid connection)
                source_type = chromosome.nodes[chromosome.nodes['id'] == source]['type'].values[0]
                target_type = chromosome.nodes[chromosome.nodes['id'] == target]['type'].values[0]
                
                if source_type == 'output' or target_type == 'input' or source == target:
                    continue
                
                # Check if adding this edge would create a cycle
                if not chromosome.would_create_cycle(source, target):
                    # Add the edge with random weight
                    chromosome.add_edge(
                        source=source,
                        target=target,
                        weight=np.random.uniform(-1, 1)
                    )
                    break  # Successfully added an edge
        
        # Structural mutation: add new hidden node
        if random.random() < mutation_rate * 0.1:  # Very low probability for adding nodes
            # Add a new hidden node
            new_node_id = chromosome.add_node(
                node_type='hidden',
                bias=np.random.uniform(-1, 1),
                birth_generation=self.generation
            )
            
            # Connect it to the network
            # 1. Connect from a random existing node
            existing_nodes = chromosome.nodes[chromosome.nodes['id'] != new_node_id]['id'].values
            if len(existing_nodes) > 0:
                source_id = random.choice(existing_nodes)
                source_type = chromosome.nodes[chromosome.nodes['id'] == source_id]['type'].values[0]
                
                if source_type != 'output':  # Outputs cannot be sources
                    chromosome.add_edge(
                        source=source_id,
                        target=new_node_id,
                        weight=np.random.uniform(-1, 1)
                    )
            
            # 2. Connect to a random existing node
            if len(existing_nodes) > 0:
                target_id = random.choice(existing_nodes)
                target_type = chromosome.nodes[chromosome.nodes['id'] == target_id]['type'].values[0]
                
                if target_type != 'input':  # Inputs cannot be targets
                    chromosome.add_edge(
                        source=new_node_id,
                        target=target_id,
                        weight=np.random.uniform(-1, 1)
                    )
        
        return chromosome
    
    def evolve(self, generations=10, mutation_rate=0.2, selection_method='tournament'):
        """
        Run the evolutionary process.
        
        Args:
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation for each gene
            selection_method: Method for parent selection
            
        Returns:
            Chromosome: Best chromosome found
        """
        for _ in range(generations):
            # Evaluate current population
            fitness_scores = self.evaluate_population()
            
            # Print generation stats
            best_fitness = fitness_scores[0][1]
            avg_fitness = sum(score for _, score in fitness_scores) / len(fitness_scores)
            print(f"Generation {self.generation}: Best fitness = {best_fitness}, Avg fitness = {avg_fitness:.2f}")
            
            # Select parents
            parents = self.select_parents(fitness_scores, selection_method)
            
            # Create new population
            new_population = []
            
            # Elitism: Keep the best chromosome
            new_population.append(fitness_scores[0][0])
            
            # Create offspring through crossover and mutation
            while len(new_population) < self.population_size:
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)
                
                # Ensure parents are different
                attempts = 0
                while parent2.id == parent1.id and len(parents) > 1 and attempts < 10:
                    parent2 = random.choice(parents)
                    attempts += 1
                
                # Crossover
                child = self.crossover(parent1, parent2)
                
                # Mutation
                child = self.mutate(child, mutation_rate)
                
                new_population.append(child)
            
            # Update population
            self.population = new_population
            self.generation += 1
        
        # Return best chromosome
        return self.best_chromosome
    
    def visualize_fitness_history(self):
        """
        Visualize the fitness history over generations.
        """
        df = pd.DataFrame(self.fitness_history)
        
        plt.figure(figsize=(10, 6))
        plt.plot(df['generation'], df['max_fitness'], 'b-', label='Max Fitness')
        plt.plot(df['generation'], df['avg_fitness'], 'r-', label='Average Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness History')
        plt.legend()
        plt.grid(True)
        plt.show()


def run_evolution_experiment():
    """Run an evolutionary experiment for the pole balancing task."""
    # Parameters
    POPULATION_SIZE = 30
    GENERATIONS = 20
    MUTATION_RATE = 0.2
    
    # Initialize evolution
    evolution = Evolution(
        population_size=POPULATION_SIZE,
        inputs=4,  # CartPole has 4 inputs: position, velocity, angle, angular velocity
        outputs=1,  # We need 1 output for left/right decision
        hidden_nodes=15,
        connectivity_ratio=0.6
    )
    
    # Run evolution
    best_chromosome = evolution.evolve(
        generations=GENERATIONS,
        mutation_rate=MUTATION_RATE,
        selection_method='tournament'
    )
    
    # Visualize fitness history
    evolution.visualize_fitness_history()
    
    # Test the best chromosome
    print(f"\nTesting best chromosome (Generation {best_chromosome.generation}, ID {best_chromosome.id})")
    final_fitness = evolution.evaluate_fitness(best_chromosome, render=True)
    print(f"Final fitness: {final_fitness}")
    
    # Visualize the best network
    plt.figure(figsize=(12, 10))
    best_chromosome.visualize()
    plt.title(f"Best Chromosome: Gen {best_chromosome.generation}, ID {best_chromosome.id}, Fitness {final_fitness}")
    plt.show()
    
    return best_chromosome, evolution


if __name__ == "__main__":
    best_solution, evo = run_evolution_experiment()