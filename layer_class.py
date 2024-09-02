import numpy as np
from neuron_class import *

class Layer:
    def __init__ (self, neuron_number, input_size, activation_function):
        self.neuron_number = neuron_number
        self.input_size = input_size
        self.neurons = [Neuron(input_size=input_size) for _ in range(neuron_number)]
        self.activation_function = activation_function
    
    def push(self, input_vector):
        activated_values = np.array([neuron.calculate(input_vector=input_vector, activation_function=self.activation_function) for neuron in self.neurons])
        return activated_values

if __name__ == "__main__":
    layer = Layer(neuron_number=3, input_size=2)
    print(layer.calculate(np.array([2, 3])))
