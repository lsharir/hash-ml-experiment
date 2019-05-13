import numpy as np
import utils

from hash_libs.md2 import MD2


class Md2Provider:
    def __init__(self, rounds=18):
        self.rounds = rounds

    def md2(self, message):
        md2_value = MD2(message, self.rounds).hexdigest()
        return md2_value

    def training_pair(self, input_dtype=np.float, output_dtype=np.float):
        y_hex = utils.make_n_bit_hex_string(128, '032x')
        x_hex = self.md2(y_hex)
        y_bits = utils.convert_hex_2_bits(y_hex, output_dtype)
        x_bits = utils.convert_hex_2_bits(x_hex, input_dtype)
        return x_bits, y_bits
