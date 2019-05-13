from torch.utils import data
from data import data_utils
from hash_providers.lcg import LcgProvider
from hash_providers.md2 import Md2Provider
from hash_providers.dummy import DummyProvider


class DummyBitDataset(data.Dataset):
    def __init__(self, bit_index, n=7, num_batches=128, batch_size=128):
        self.bit_index = bit_index
        self.size = num_batches * batch_size
        self.num_batches = num_batches
        self.provider = DummyProvider(n)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        x, y = data_utils.dummy_training_example(self.provider)
        return x, y[-(self.bit_index+1)]

class Md2BitDataset(data.Dataset):
    def __init__(self, bit_index, rounds=18, num_batches=32, batch_size=16384):
        self.bit_index = bit_index
        self.size = num_batches * batch_size
        self.num_batches = num_batches
        self.provider = Md2Provider(rounds)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        x, y = data_utils.md2_training_example(self.provider)
        return x, y[-(self.bit_index+1)]


class LcgBitDataset(data.Dataset):
    def __init__(self, bit_index, a, c, m, n, num_batches=128, batch_size=128):
        self.a = a
        self.c = c
        self.m = m
        self.n = n
        self.provider = LcgProvider(a, c, m, n)
        self.bit_index = bit_index
        self.size = num_batches * batch_size
        self.num_batches = num_batches

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        x, y = data_utils.lcg_training_example(self.provider)
        return x, y[-(self.bit_index + 1)]
