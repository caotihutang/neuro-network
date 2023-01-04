import numpy as np


class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.random.randn(1, output_size)
        self.propagate = np.random.randn(output_size, input_size)
        # self.inv_grad = np.random.randn(output_size * input_size, output_size)
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, input_data):
        return sigmoid(np.matmul(np.transpose(self.weights), input_data) +  self.bias)

    def get_grad(self, input_data, next_layer = None, is_final_layer=False):
        # layer output
        out_data = self.forward(input_data)
        # factor of activation function
        factor = np.matmul(out_data, np.transpose(out_data - 1))
        grad = np.multiply(factor, np.eye(self.output_size) * input_data[0][0])
        # inv_grad = np.linalg.inv(grad)
        for i in range(self.input_size-1):
            temp = np.multiply(factor, np.eye(self.output_size) * input_data[0][i+1])
            # concat grad in order of row
            grad = np.r_([grad, temp])
            # inv_grad = np.c_([inv_grad, np.linalg.inv(temp)])
        # self.inv_grad = inv_grad
        if is_final_layer:
            weights_grad = grad
            bias_grad = np.matmul(factor, np.eye(self.output_size))
            self.propagate = np.eye(self.output_size)
        else:
            if not next_layer:
                raise Exception('get_grad() missing 1 required positional argument: %s This layer is not the final layer!'.format('next_layer'))
            # self.weights_grad = np.matmul(np.matmul(np.matmul(next_layer.weights_grad, next_layer.inv_grad), next_layer.weights), grad)
            weights_grad = np.matmul(next_layer.propagate, grad)
            bias_grad = np.matmul(next_layer.propagate, np.matmul(factor, np.eye(self.output_size)))
            self.propagate = np.matmul(next_layer.propagate, np.transpose(self.weights))

        return weights_grad, bias_grad

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Net:
    def __init__(self, net_shape):
        self.layers = []
        self.output = []
        length = len(net_shape)
        for i in range(length-1):
            self.layers.append(Layer(net_shape[i], net_shape[i+1]))

    def forward(self, input_data):
        x = self.layers[0].forward(input_data)
        self.output.append(x)
        for i in range(1, len(self.layers)-1):
            x = self.layers[i].forward(x)
            self.output.append(x)

        return x

    def binary_loss(self,):
    def backward(self, loss):
        weights_grad, bias_grad = self.layers[-1].get_grad(self.output[-2], is_final_layer=True)
        self.layers[-1].weights = self.layers[-1].weights -
        for i in reversed(range(1, len(self.layers) - 1)):
