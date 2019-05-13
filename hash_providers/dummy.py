import numpy as np
import random


class DummyProvider:
    def __init__(self, n):
        self.n = n
        self.m = 2 ** n
        self.seed = self.random()

    def digest(self, x):
        return self.seed ^ x

    def random(self):
        return random.randint(0, self.m - 1)

    def int_to_01(self, num, dtype=np.float):
        return np.array(list(bin(num)[2:].zfill(self.n)), dtype=dtype)

    def training_pair(self, input_dtype=np.float, output_dtype=np.float):
        message = self.random()
        y = message
        x = self.digest(message)
        y_bits = self.int_to_01(y, dtype=output_dtype)
        x_bits = self.int_to_01(x, dtype=input_dtype)
        return x_bits, y_bits
