import numpy as np
from reader import Reader

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_deriv(z):
    return sigmoid(z)*(1-sigmoid(z))

class Network:
    def __init__(self, sizes, activation, a_derivative, batch_size=10):
        self.activation = activation
        self.a_derivative = a_derivative
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.biases_updates = self.biases
        self.batch_size = batch_size
        for i in range(len(self.biases_updates)):
            self.biases_updates[i] = np.zeros_like(self.biases_updates[i])
        self.weights_updates = self.weights
        for i in range(len(self.weights_updates)):
            self.weights_updates[i] = np.zeros_like(self.weights_updates[i])

    def train_step(self, input, expected_out):
        anow = np.frombuffer(input, dtype=np.uint8)
        anow = anow.reshape((784, 1))
        anow = anow / 255.0
        a = [anow]
        ztab = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, anow) + b
            ztab.append(z)
            anow = self.activation(z)
            a.append(anow)

        dera = (a[-1] - expected_out) * 2
        self.biases_updates[-1] -= (dera * self.a_derivative(ztab[-1])) / self.batch_size
        self.weights_updates[-1] -= np.dot(dera * self.a_derivative(ztab[-1]), a[-2].transpose()) / self.batch_size

        for l in range(self.num_layers-3, -1, -1):
            dera = np.dot(self.weights[l+1].transpose(), dera)
            self.biases_updates[l] -= dera * self.a_derivative(ztab[l])
            self.weights_updates[l] -= np.dot(dera * self.a_derivative(ztab[l]), a[l].transpose())

    def train(self, eta):
        r = Reader('tests/train-images-idx3-ubyte', 'tests/train-labels-idx1-ubyte')
        # for i in range(r.set_size // self.batch_size):
        for i in range(200):
            print("Epoch {0}".format(i))
            for _ in range(self.batch_size):
                test = r.next_test()
                expected = np.zeros((10, 1))
                expected[np.frombuffer(test[1], dtype=np.uint8)] = 1.0
                self.train_step(test[0], expected)
            self.update(eta, r.set_size)
        print(self.biases)

    def update(self, eta, batch_size):
        for i in range(len(self.biases)):
            self.biases[i] -= eta * self.biases_updates[i]
        for i in range(len(self.weights)):
            self.weights[i] -= eta * self.weights_updates[i]

        for i in range(len(self.biases_updates)):
            self.biases_updates[i] = np.zeros_like(self.biases_updates[i])

        for i in range(len(self.weights_updates)):
            self.weights_updates[i] = np.zeros_like(self.weights_updates[i])

n = Network([784, 16, 16, 10], sigmoid, sigmoid_deriv)
n.train(0.1)