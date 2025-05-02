from chromosome3 import *
import matplotlib.pyplot as plt

# Create a division model with 4 inputs (age, division count, connection count, node_id) with more randomized weights
division_model = Chromosome(
    id=999,
    inputs=4,  # [age, division_count, connection_count, node_id]
    outputs=1,  # [should_divide]
    hidden_layers=[5, 3]
)

# Randomize some of the weights in the division model to ensure varied responses
for _ in range(10):
    edge_id = np.random.choice(division_model.edges['id'].values)
    division_model.update_edge_weight(edge_id, np.random.uniform(-2, 2))

# Create our main chromosome to be evaluated
chromosome = Chromosome(id=1, inputs=3, outputs=2, hidden_layers=[5, 4, 3]) 

# Define a bias function with increased variability
def my_bias_function(parent_bias):
    return parent_bias + np.random.uniform(-1.0, 1.0)  # Wider range of variability

# Run the division evaluation
new_nodes = chromosome.evaluate_all_nodes_for_division(
    division_model, 
    bias_func=my_bias_function
)

# Print division results
if len(new_nodes) > 0:
    print(f"Created {len(new_nodes)} new nodes from division: {new_nodes}")
    print(f"Division ratio: {len(new_nodes)}/{len(chromosome.get_hidden_nodes()) - len(new_nodes)} "
          f"({(len(new_nodes)/(len(chromosome.get_hidden_nodes()) - len(new_nodes))) * 100:.2f}%)")
else:
    print("No divisions occurred")

# Visualize the network after division
plt.figure(figsize=(12, 10))
chromosome.visualize()
plt.title(f"Chromosome {chromosome.id} with {len(chromosome.nodes)} nodes")
plt.show()

def test_thresholds():
    print("\nTesting variable thresholds:")
    # Save the random thresholds for each node in a dictionary to display
    node_thresholds = {}
    
    for node_id in chromosome.get_hidden_nodes()['id'].values:
        # Generate a random threshold just like in evaluate_division
        threshold = random.random()
        node_thresholds[node_id] = threshold
        
        # Call evaluate_division (this will generate a new random threshold internally)
        result = chromosome.evaluate_division(node_id, division_model)
        
        # Display the threshold we generated (not the one used in evaluate_division)
        print(f"Node {node_id}: random threshold = {threshold:.4f}, will divide = {result}")

# Uncomment to test thresholds
test_thresholds()