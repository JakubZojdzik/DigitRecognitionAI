import numpy as np

class Reader:
    def __init__(self, file_images, file_labels):
        self.file_images = open(file_images, 'rb')
        self.file_labels = open(file_labels, 'rb')
        self.index = 0
        self.set_size = 0
        self.rows = 0
        self.cols = 0
        self.read_meta()

    def read_meta(self):
        self.file_images.seek(4)
        self.set_size = int.from_bytes(self.file_images.read(4), 'big')
        self.rows = int.from_bytes(self.file_images.read(4), 'big')
        self.cols = int.from_bytes(self.file_images.read(4), 'big')

    def all_tests(self):
        res = []
        for _ in range(self.set_size):
            self.file_images.seek(16 + self.index * self.rows * self.cols)
            self.file_labels.seek(8 + self.index)
            self.index += 1
            pixels = np.frombuffer(self.file_images.read(self.rows * self.cols), dtype=np.uint8)
            pixels = pixels.reshape((784, 1))
            pixels = pixels / 255.0
            label = int.from_bytes(self.file_labels.read(1), 'big')
            res.append((pixels, label))

            pixels = pixels.reshape((28, 28))

            pixels = np.roll(pixels, 1, axis=1)
            res.append((pixels.reshape((784, 1)), label))

            pixels = np.roll(pixels, -2, axis=1)
            res.append((pixels.reshape((784, 1)), label))

            pixels = np.roll(pixels, 1, axis=0)
            res.append((pixels.reshape((784, 1)), label))

            pixels = np.roll(pixels, -2, axis=0)
            res.append((pixels.reshape((784, 1)), label))
        return res