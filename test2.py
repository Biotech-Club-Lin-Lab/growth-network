from chromosome import *

# Create a new network
chromosome = Chromosome(0, inputs=3, outputs=2, hidden=2)

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