import time
import math
import random
import string
import torch
from bitarray import bitarray
import numpy as np
from Cryptodome.Hash import MD2

letters = list(string.ascii_letters)

def toBinary(text):
    a_bytes = bytes(text, "ascii")
    b_list = []
    for x in a_bytes:
        for b in "{0:b}".format(x):
            b_list.append(int(b))

    return b_list


def letterHashStupid(letter):
    return chr(ord(letter) + 1)


def randomTrainingPair():
    output_ascii = randomChoice(string.ascii_letters)
    input_ascii = letterHashStupid(output_ascii)
    output_binary = toBinary(output_ascii)
    input_binary = toBinary(input_ascii)
    return input_binary, output_binary


def randomTrainingExample():
    input_binary, output_binary = randomTrainingPair()
    return torch.FloatTensor(input_binary), torch.FloatTensor(output_binary)


def make_n_bit_hex_string(n, hex_format):
    bits = random.getrandbits(n)
    return format(bits, hex_format)


def convert_hex_2_bits(hex_string):
    _bitarray = bitarray()
    _bitarray.frombytes(bytes.fromhex(hex_string))
    return np.array(_bitarray.tolist(), dtype=np.float)


def convert_bits_2_hex(bits_array, hex_format='032x'):
    bits = bits_array
    if type(bits_array) is torch.Tensor:
        bits = bits_array.tolist()
    _bitarray = bitarray(bits)
    return format(int(_bitarray.to01(), 2), hex_format)


def md2TrainingPair():
    y_hex = make_n_bit_hex_string(128, '032x')
    x_hex = md2(y_hex)
    y_bits = convert_hex_2_bits(y_hex)
    x_bits = convert_hex_2_bits(x_hex)
    return x_bits, y_bits


def md2TrainingExample():
    input_binary, output_binary = md2TrainingPair()
    return torch.Tensor(input_binary), torch.Tensor(output_binary)


def stabilize_prediction(x):
    return x - .5 > 0


def toAscii(x):
    x_text = ""
    x_bytes = zip(*(iter(x),) * 7)
    for x_byte in x_bytes:
        n = int('0b{}'.format("".join([str(int(b)) for b in x_byte])), 2)
        x_text += n.to_bytes((n.bit_length() + 7) // 8, 'big').decode()
    return x_text


def md2(hex_y):
    bytes_y = bytes.fromhex(hex_y)
    h = MD2.new()
    h.update(bytes_y)
    return h.hexdigest()
