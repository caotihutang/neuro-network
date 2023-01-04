import numpy as np


class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)

    def forward(self,x):
        self.weights = x


x = Layer(2,2)
x.forward(np.array([[1,2],[3,4]]))
print(x.forward())