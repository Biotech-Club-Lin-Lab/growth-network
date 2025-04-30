from chromosome3 import *
division_model = Chromosome(
    id=999,
    inputs=3,  # [age, division_count, connection_count]
    outputs=1,  # [should_divide]
    hidden_layers=[4]
)
chromosome = Chromosome(id=1, inputs=3, outputs=2, hidden_layers=[5, 4, 3]) 


def my_bias_function(parent_bias):
    return parent_bias + np.random.uniform(-0.5, 0.5)
    
new_nodes = chromosome.evaluate_all_nodes_for_division(
    division_model, 
    bias_func=my_bias_function
)

chromosome.visualize()
plt.show()
    