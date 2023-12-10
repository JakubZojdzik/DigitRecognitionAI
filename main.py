import numpy as np
from network import Network
from reader import Reader

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def d_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))

def relu(z):
    return np.maximum(0, z)

def d_relu(z):
    return np.where(z > 0, 1, 0)

n = Network([784, 16, 16, 10], relu, d_relu)
r = Reader('tests/train-images-idx3-ubyte', 'tests/train-labels-idx1-ubyte')

t = r.all_tests()
n.train(t, 15, 64, 0.1)