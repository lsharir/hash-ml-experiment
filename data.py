import torch
from torch.utils import data
import utils


class MD2_Datasource(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self):
        pass

    def __len__(self):
        return 500000

    def __getitem__(self, index):
        return utils.md2TrainingExample()
