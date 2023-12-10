import numpy as np

class Network:
    def __init__(self, sizes, activation, d_activation):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.activation = activation
        self.d_activation = d_activation

        self.biases = [np.random.randn(size, 1) / np.sqrt(size) for size in sizes[1:]]
        self.weights = [np.random.randn(size2, size1) / np.sqrt(size1) for size1, size2 in zip(sizes[:-1], sizes[1:])]

        self.biases_upd = [np.zeros(b.shape) for b in self.biases]
        self.weights_upd = [np.zeros(w.shape) for w in self.weights]

        self.a_arr = []
        self.z_arr = []

    def feedforward(self, a):
        self.a_arr = [a]
        self.z_arr = [self.activation(a)]
        layer_itr = 1
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            if layer_itr != self.num_layers - 1:
                a = self.activation(z)
            else:
                a = self.softmax(z)
            self.z_arr.append(z)
            self.a_arr.append(a)
            layer_itr += 1
        return a

    def backprop(self, y):
        for i in range(self.num_layers - 1, 0, -1):
            # print(i, self.num_layers - 1)
            if i == self.num_layers - 1:
                delta = self.d_softmax_cross_entropy(self.a_arr[i], y)
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

    @staticmethod
    def softmax(z):
        return np.exp(z) / np.sum(np.exp(z), axis=0)

    @staticmethod
    def d_softmax_cross_entropy(z, y):
        return z - y
