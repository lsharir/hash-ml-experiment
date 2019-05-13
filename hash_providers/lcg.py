import numpy as np
import random


class LcgProvider:
    def __init__(self, a, c, m, n):
        self.a = a
        self.c = c
        self.m = m
        self.n = n

    def digest(self, x):
        return (self.a * x + self.c) % self.m

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
