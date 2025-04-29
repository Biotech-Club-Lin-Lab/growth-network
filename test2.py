from chromosome import *

# Create a new network
chromosome = Chromosome(id=1, inputs=3, outputs=2, hidden_layers=[5, 8, 3])

# Get node by ID
node = chromosome.get_node_by_id(3)
print(node)

# Get all edges from a source node
edges = chromosome.get_edges_by_source(0)
print(edges)

# Add a new node to hidden layer
new_node_id = chromosome.add_node(layer=1, bias=0.5)

# Connect the new node to existing nodes
chromosome.add_edge(source=0, target=new_node_id)
chromosome.add_edge(source=new_node_id, target=5)

# Visualize the network
chromosome.visualize()
plt.show()

outputs = chromosome.NN.run([.1, .2, .3])
print(outputs)