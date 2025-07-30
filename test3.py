from chromosome3 import *
import matplotlib.pyplot as plt

# Create a division model with 4 inputs (age, division count, connection count, node_id) with more randomized weights
division_model = Chromosome(
    id=999,
    inputs=4,  # [age, division_count, connection_count, node_id]
    outputs=1,  # [should_divide]
    hidden_layers=[5, 3]
)

# Randomize ALL of the weights in the division model with a wider range
for _, edge in division_model.edges.iterrows():
    division_model.update_edge_weight(edge['id'], np.random.uniform(-1.5, 1.5))

# Set a moderate bias for the output node
output_nodes = division_model.get_output_nodes()
for _, node in output_nodes.iterrows():
    division_model.update_node_bias(node['id'], 0.2)

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

# Visualize the network after division
plt.figure(figsize=(12, 10))
chromosome.visualize()
plt.title(f"Chromosome {chromosome.id} with {len(chromosome.nodes)} nodes")
plt.show()

def test_thresholds():
    print("\nTesting division model with randomized inputs and outputs:")
    division_counts = {"will_divide": 0, "wont_divide": 0}
    
    for node_id in chromosome.get_hidden_nodes()['id'].values:
        # Make multiple tests for each node to see variability
        results = []
        for _ in range(3):  # Test each node 3 times
            # Run the evaluate_division to get a result
            result = chromosome.evaluate_division(node_id, division_model)
            results.append(result)
            
            if result:
                division_counts["will_divide"] += 1
            else:
                division_counts["wont_divide"] += 1
        
        # Display results
        print(f"Node {node_id}: division results from 3 tests = {results}")
    
    # Print overall statistics
    total_tests = division_counts["will_divide"] + division_counts["wont_divide"]
    divide_percentage = (division_counts["will_divide"] / total_tests) * 100 if total_tests > 0 else 0
    print(f"\nOverall division statistics:")
    print(f"Will divide: {division_counts['will_divide']} ({divide_percentage:.1f}%)")
    print(f"Won't divide: {division_counts['wont_divide']} ({100-divide_percentage:.1f}%)")

# Uncomment to test thresholds
test_thresholds()