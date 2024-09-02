import numpy as np
import math
from neuron_class import *
from layer_class import *

def relu_activation(z): return max(z, 0)
def softmax_activation(z): return max(z, 0)

class Network:
    def __init__(self, structure, input_neurons):
        structure = np.insert(structure, 0, input_neurons)
        self.layers = [Layer(neuron_number=structure[i], input_size=structure[i - 1], activation_function=relu_activation) for i in range(1, len(structure))]
        self.layers[-1].activation_function = softmax_activation
    
    def feedforward(self, input_vector):
        for layer in self.layers:
            input_vector = layer.push(input_vector=input_vector)
        return input_vector

if __name__ == "__main__":
    net = Network(structure=np.array([784, 16, 16, 10]), input_neurons=2)
    output = net.feedforward([3, 8])
    print(output)