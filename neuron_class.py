import numpy as np
import math

id_counter = 0

np.random.seed(0)

class Neuron:
    def __init__ (self, input_size):
        global id_counter
        self.id = id_counter + 1
        id_counter += 1
        
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        
        # if input_size == 2: self.weights = np.array([1, 1])
        # if input_size == 3: self.weights = np.array([1, 1, 1])
        
        self.bias = 0
        
        self.hold_value = None
    
    def calculate(self, input_vector, activation_function):
        linear_sum =  np.dot(input_vector, self.weights) + self.bias
        self.hold_value = activation_function(z=linear_sum)
        return self.hold_value


    
if __name__ == "__main__":
    n = Neuron(2)
    input_vector = np.array([1, 2])
    print(n.calculate(input_vector))