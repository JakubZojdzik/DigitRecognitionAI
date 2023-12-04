import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_deriv(z):
    return sigmoid(z)*(1-sigmoid(z))

class Network:
    def __init__(self, sizes, activation, a_derivative):
        self.activation = activation
        self.a_derivative = a_derivative
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.biases_updates = np.zeros_like(self.biases)
        self.weights_updates = np.zeros_like(self.weights)

    def train_step(self, input, expected_out):
        anow = input
        a = [input]
        z = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, anow) + b
            z.append(z)
            anow = self.activation(z)
            a.append(anow)

        # Backpropagation
        dera = np.multiply((a[-1] - expected_out) * 2)
        self.biases_updates[-1] = dera * self.a_derivative(z[-1])
        self.weights_updates[-1] = np.dot(dera * self.a_derivative(z[-1]), a[-2].transpose())

        for l in range(self.num_layers-3, -1, -1):
            dera = np.dot(self.weights[l+1].transpose(), dera)
            self.biases_updates[l] = dera * self.a_derivative(z[l])
            self.weights_updates[l] = np.dot(dera * self.a_derivative(z[l]), a[l].transpose())

n = Network([2, 3, 2], sigmoid, sigmoid_deriv)
print(n.biases)