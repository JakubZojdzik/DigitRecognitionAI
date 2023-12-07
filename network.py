import numpy as np
from reader import Reader

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def d_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))

def relu(z):
    return np.maximum(0, z)

def d_relu(z):
    return np.where(z > 0, 1, 0)

class Network:
    def __init__(self, sizes, activation, d_activation):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.activation = activation
        self.d_activation = d_activation

        self.biases = [np.random.randn(size, 1) for size in sizes[1:]]
        self.weights = [np.random.randn(size2, size1) / np.sqrt(size1) for size1, size2 in zip(sizes[:-1], sizes[1:])]

        self.biases_upd = [np.zeros(b.shape) for b in self.biases]
        self.weights_upd = [np.zeros(w.shape) for w in self.weights]

        self.a_arr = []
        self.z_arr = []

    def feedforward(self, a):
        self.a_arr = [a]
        self.z_arr = [self.activation(a)]
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            a = self.activation(z)
            self.z_arr.append(z)
            self.a_arr.append(a)
        return a

    def backprop(self, y):
        for i in range(self.num_layers - 1, 0, -1):
            if i == self.num_layers - 1:
                delta = (self.a_arr[i] - y) * self.d_activation(self.z_arr[i])
            else:
                delta = np.dot(self.weights[i].T, delta) * self.d_activation(self.z_arr[i])
            self.biases_upd[i-1] = delta
            self.weights_upd[i-1] = np.dot(delta, self.a_arr[i-1].T)

    def update(self, lr, batch_size):
        self.biases = [b - (lr / batch_size) * b_upd for b, b_upd in zip(self.biases, self.biases_upd)]
        self.weights = [w - (lr / batch_size) * w_upd for w, w_upd in zip(self.weights, self.weights_upd)]

        self.biases_upd = [np.zeros(b.shape) for b in self.biases]
        self.weights_upd = [np.zeros(w.shape) for w in self.weights]


    def onehot(self, y):
        onehot = np.zeros((self.sizes[-1], 1))
        onehot[y] = 1.0
        return onehot

    def train(self, train_data, epochs, batch_size, lr):
        n = len(train_data)
        for i in range(epochs):
            correct = 0
            np.random.shuffle(train_data)
            batches = [train_data[k:k+batch_size] for k in range(0, n, batch_size)]
            itr = 0
            for batch in batches:
                itr += 1
                mini_correct = 0
                for x, y in batch:
                    if(np.argmax(self.feedforward(x)) == y):
                        correct += 1
                        mini_correct += 1
                    self.backprop(self.onehot(y))
                self.update(lr, batch_size)
                if(itr % 100 == 0):
                    print(f"Mini batch: {mini_correct}/{len(batch)}")

            print(f"Epoch {i}:  {correct}/{len(train_data)}")


n = Network([784, 16, 16, 10], relu, d_relu)
r = Reader('tests/train-images-idx3-ubyte', 'tests/train-labels-idx1-ubyte')

t = r.all_tests()
print(len(t))
n.train(t, 4, 64, 0.1)