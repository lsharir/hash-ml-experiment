import random
import torch
from bitarray import bitarray
import numpy as np
from . import analysis


def make_n_bit_hex_string(n, hex_format):
    bits = random.getrandbits(n)
    return format(bits, hex_format)


def convert_hex_2_bits(hex_string, dtype):
    _bitarray = bitarray()
    _bitarray.frombytes(bytes.fromhex(hex_string))
    return np.array(_bitarray.tolist(), dtype=dtype)


def convert_bits_2_hex(bits_array, hex_format='032x'):
    bits = bits_array
    if type(bits_array) is torch.Tensor:
        bits = bits_array.tolist()
    _bitarray = bitarray(bits)
    return format(int(_bitarray.to01(), 2), hex_format)


def model_vs_random(model, data_loader):
    correct = 0
    correct_at_random = 0
    total = 0
    for x, y in data_loader:
        num_samples = y.size()[0]
        correct += torch.sum(y == torch.argmax(model(x), dim=1)).item()
        correct_at_random += np.sum(np.random.choice(2, num_samples))
        total += num_samples

    return correct / total, correct_at_random / total


def count_parameters(model):
    return sum(torch.numel(p) for p in model.parameters() if p.requires_grad)
