import torch
import numpy as np
from hash_providers.lcg import LcgProvider
from hash_providers.dummy import DummyProvider
from hash_providers.md2 import Md2Provider


def dummy_training_example(provider: DummyProvider):
    input_binary, output_binary = provider.training_pair(input_dtype=np.float, output_dtype=np.int)
    return torch.Tensor(input_binary), output_binary

def md2_training_example(provider: Md2Provider):
    input_binary, output_binary = provider.training_pair(input_dtype=np.float, output_dtype=np.int)
    return torch.Tensor(input_binary), output_binary


def lcg_training_example(provider: LcgProvider):
    input_binary, output_binary = provider.training_pair(input_dtype=np.float, output_dtype=np.int)
    return torch.Tensor(input_binary), output_binary
