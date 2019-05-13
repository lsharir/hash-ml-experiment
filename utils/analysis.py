from matplotlib import pyplot as plt
import numpy as np


def net_vs_random(net_acc_all, rand_acc_all, acc_size, hash_method, nn_params):
    bit_length = len(net_acc_all)
    bit_indices = range(bit_length)
    bit_labels = ["{}".format(bit_length - index - 1) for index in bit_indices]
    # bit_labels[0] = "MSB"
    bit_labels[-1] = "LSB"
    baseline = [.5] * bit_length
    lowest_acc = np.floor(20* min([min(net_acc_all), min(rand_acc_all)])) / 20
    plt.figure()
    plt.title("{}\nBit guessing accuracy by bit position ({} queries)\n{}".format(hash_method, acc_size, nn_params))
    plt.grid(True)
    plt.xticks(bit_indices, bit_labels, fontsize=16)
    plt.yticks(np.arange(lowest_acc, 1.025, 0.025))
    plt.ylim([lowest_acc, 1.05])
    plt.xlim([-0.5, bit_length - 0.5])
    net_plot, = plt.plot(net_acc_all, marker='o', linestyle='none', markersize=16, alpha=0.75)
    rand_plot, = plt.plot(rand_acc_all, marker='o', linestyle='none', markersize=16, alpha=0.25)
    plt.plot(net_acc_all, marker='_', color='black', linestyle='none', markersize=16, alpha=0.75)

    for i, j in zip(bit_indices, net_acc_all):
        plt.annotate("{:.2%}".format(j), xy=(i-0.05, j+0.025), fontsize=12)

    base_plot, = plt.plot(baseline, 'k:')
    plt.legend((base_plot, rand_plot, net_plot), ("Baseline", "Random Guesser", "Neural Network"))
    plt.tight_layout()
    plt.show()
