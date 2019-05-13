import torch
import torch.nn as nn
import utils
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from torch.utils import data
from torch.optim import Adagrad
from data.__init__ import Md2BitDataset


# MD2

class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.Dropout(1 / 128),
            nn.Linear(128, 64),
            nn.Linear(64, 8),
            nn.Linear(8, output_size),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


def run(bit_index=0, rounds=18, batch_size=16384, num_batches=32, num_epochs=2):
    print("=======================================")
    print("Trying on bit at {}".format(bit_index))
    learning_rate = 0.001
    model = Model(128, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    params = {'batch_size': batch_size, 'num_workers': 50}
    max_epochs = trange(num_epochs)

    train_source = Md2BitDataset(bit_index, rounds, batch_size=batch_size, num_batches=num_batches)
    train_loader = data.DataLoader(train_source, **params)

    all_losses = []
    for epoch in max_epochs:
        # Training
        max_epochs.set_description("Epoch {} - Generating data...".format(epoch + 1))
        iteration_count = 0
        for x, y in train_loader:
            iteration_count += 1
            # Transfer to GPU
            optimizer.zero_grad()

            output = model(x)
            loss = criterion(output, y)
            loss.backward()

            optimizer.step()
            all_losses.append(loss.item())

            max_epochs.set_description(
                "Epoch {} - Loss {:.3f}; {:.0%}".format(epoch + 1, loss.item(),
                                                        iteration_count / train_source.num_batches))

    plt.figure()
    plt.plot(all_losses)
    plt.show()

    # torch.save(model.state_dict(), "./models/bit_{}".format(bit_index))

    correct = 0
    correct_at_random = 0
    total = 0
    print("Estimating accuracy...")
    for x, y in train_loader:
        num_samples = y.size()[0]
        correct += torch.sum(y == torch.argmax(model(x), dim=1)).item()
        correct_at_random += np.sum(np.random.choice(2, num_samples))
        total += num_samples

    print("\nResults")
    print("Net was correct: {}".format(correct / total))
    print("Random was correct: {}".format(correct_at_random / total))
    return correct / total, correct_at_random / total, utils.count_parameters(model)


net_acc_all, rand_acc_all = [], []
batch_size, num_batches = 1024, 128
num_epochs = 2
half_value = []
for index in range(128):
    net_acc, rand_acc, models_params = run(bit_index=index, rounds=4, batch_size=batch_size, num_batches=num_batches, num_epochs=num_epochs)
    # net_acc, rand_acc = run(bit_index=index, test_size=16384, batch_size=128, num_batches=128, num_epochs=20)
    net_acc_all = [net_acc] + net_acc_all
    rand_acc_all = [rand_acc] + rand_acc_all

    utils.analysis.net_vs_random(net_acc_all, rand_acc_all, acc_size=batch_size * num_batches,
                           hash_method="MD2 - 4 rounds",
                           nn_params="Model: {} params trained {} epochs with {} random samples each".format(
                               models_params, num_epochs, batch_size * num_batches))



