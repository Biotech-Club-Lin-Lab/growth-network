import numpy as np
import pandas as pd
from chromosome import * 

def get_dfs(C:Chromosome):
    """
    Generate a nodes and edges dataframe from a chromosome object
    nodes df also contains a layer id column, refering to the coordinate pos of the node in the layer
    :param C: Chromosome object
    :return: nodes and edges dataframes"""

    #nodes df
    ids =[]
    layers = []
    biases = []
    for node in C.nodes:
        ids.append(node.id)
        layers.append(node.layer)
        biases.append(node.bias)
    nodes = pd.DataFrame({'id':ids,'layer':layers,'bias':biases})

    #add layer id col to nodes df
    keys=nodes['layer'].unique().tolist()
    vals = [0]*len(keys)
    counter = dict(zip(keys,vals))
    newcol = []
    for layer in nodes['layer']:
        newcol.append(counter[layer])
        counter[layer] += 1
    newcol
    nodes.insert(2,'layer id',newcol)

    #edges df
    ids = []
    sources = []
    targets = []
    weights = []
    enabled = []
    for edge in C.edges:
        ids.append(edge.id)
        sources.append(edge.source)
        targets.append(edge.target)
        weights.append(edge.weight)
        enabled.append(edge.enabled)
    edges = pd.DataFrame({'id':ids,'source':sources,'target':targets,'weight':weights,'enabled':enabled})
    return nodes, edges

def get_wb(source, target, nodes, edges):
    
    s_nodes = nodes.loc[nodes['layer'] == source]
    t_nodes = nodes.loc[nodes['layer'] == target]
    layer_edges = edges.loc[edges['source'].isin(s_nodes['id']) & edges['target'].isin(t_nodes['id'])]

    #create adjacency matrix
    w = np.zeros((len(s_nodes),len(t_nodes)))
    for row in layer_edges.iterrows():
        edge = row[1]
        s_id = edge['source']
        t_id = edge['target']
        s = s_nodes.loc[s_nodes['id'] == s_id, 'layer id'].values[0]
        t = t_nodes.loc[t_nodes['id'] == t_id, 'layer id'].values[0]
        w[s,t] = edge['weight']
    
    b = t_nodes['bias'].values 
    return w,b

class Layer:
    def __init__(self, weights, biases):
        """
        Layer class represents a hidden/output layer in the network.
        :param weights: adjacency matrix between source and target nodes containing weights of the layer
        :param biases: biases of the target layer"""
        
        self.weights = weights
        self.biases = biases
        
    def forward(self, inputs):
        self.output = inputs @ self.weights + self.biases # matrix multiplication of inputs and weights + biases

class ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0,inputs)

class NN:
    def __init__(self, C, af='relu'):
        """
        :param C: Chromosome object
        :param af: activation function, default is relu
        """
        #define nodes and edges dfs
        nodes, edges = get_dfs(C)
        
        #define list of layers
        uniq_layers = nodes['layer'].unique()
        layers = []
        for i in range(len(uniq_layers) - 1): # -1 to stop at output layer
            s_layer = uniq_layers[i]
            t_layer = uniq_layers[i+1]
            w,b = get_wb(s_layer, t_layer, nodes, edges)
            layers.append(Layer(weights=w, biases=b))
        
        #define activation function
        if af == 'relu':
            self.af = ReLU()
        else:
            raise ValueError("Activation function not supported")

        self.nodes = nodes
        self.edges = edges
        self.layers = layers

    def result(self, inputs:np.array, verbose=False):
        """
        :param inputs: input array
        :return: output array
        """
        #forward pass
        for i in range(len(self.layers)):
            layer = self.layers[i]
            layer.forward(inputs)
            self.af.forward(layer.output)
            inputs = self.af.output
            if verbose:
                print(f'layer {i+1}: ', inputs)

        outputs = inputs
        return outputs
    


        