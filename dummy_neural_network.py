import torch
import torch.nn as nn
from torch.utils import data
from torch.optim import Adagrad
from tqdm import trange

import utils
from data import DummyBitDataset


# Dummy Neural Network
class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 2),
            nn.Linear(2, output_size),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


def test_dummy(n, bit_index=0, batch_size=128, num_batches=128, num_epochs=2):
    print("Initializing a model for bit {}...".format(bit_index))
    model = Model(n, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    data_source = DummyBitDataset(bit_index, n=n, batch_size=batch_size, num_batches=num_batches)
    data_loader = data.DataLoader(data_source, **{'batch_size': batch_size, 'num_workers': 50})

    max_epochs = trange(num_epochs)
    for epoch in max_epochs:
        max_epochs.set_description("Epoch {} - Generating data...".format(epoch + 1))
        iteration_count = 0
        for x, y in data_loader:
            optimizer.zero_grad()

            output = model(x)
            loss = criterion(output, y)
            loss.backward()

            optimizer.step()

            iteration_count += 1
            max_epochs.set_description(
                "Epoch {} - Loss {:.3f}; {:.0%}".format(epoch + 1, loss.item(),
                                                        iteration_count / data_source.num_batches))

    print("Benchmarking model against random guesser...")
    nn_guess_accuracy, random_guess_accuracy = utils.model_vs_random(model, data_loader)

    print("\n=========== Results ==============")
    print("     Net was correct: {:.2%}".format(nn_guess_accuracy))
    print("  Random was correct: {:.2%}".format(random_guess_accuracy))
    print("==================================")

    return nn_guess_accuracy, random_guess_accuracy, utils.count_parameters(model)


if __name__ == "__main__":
    nn_guess_accuracies, random_guess_accuracies = [], []
    batch_size, num_batches = 32, 256
    num_epochs = 5
    models_params = None

    # Message space definitions
    n = 4  # 4 bits e.g. m = 1000 / 0011 / 0101

    # Run test for each bit
    for index in range(n):
        net_acc, rand_acc, models_params = test_dummy(n=7, bit_index=index, batch_size=batch_size,
                                                    num_batches=num_batches,
                                                    num_epochs=num_epochs)
        nn_guess_accuracies = [net_acc] + nn_guess_accuracies
        random_guess_accuracies = [rand_acc] + random_guess_accuracies

    # Plot guessing accuracies
    utils.analysis.net_vs_random(nn_guess_accuracies, random_guess_accuracies, acc_size=batch_size * num_batches,
                           hash_method="Dummy: {}-bit message space".format(n),
                           nn_params="Model: {} params trained {} epochs with {} random samples each".format(
                               models_params, num_epochs, batch_size * num_batches))
