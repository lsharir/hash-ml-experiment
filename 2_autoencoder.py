import torch
import torch.nn as nn
import utils
import matplotlib.pyplot as plt


class autoencoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(True),
            nn.Linear(128, 128)
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, output_size), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


learning_rate = 0.0005
num_iters = 10000
model = autoencoder(7, 7)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)


def train(input_tensor, output_tensor):
    output = model(input_tensor)
    loss = criterion(output, output_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


all_losses = []

for data in range(num_iters):
    loss = train(*utils.randomTrainingExample())
    all_losses.append(loss)

plt.figure()
plt.plot(all_losses)
plt.show()

for test_index in range(10):
    x, y = utils.randomTrainingExample()
    print("=======")
    print("x={}; y={};".format(utils.toAscii(x), utils.toAscii(y)))
    y_pred = model(x)
    y_pred_ascii = utils.toAscii(utils.stabilize_prediction(y_pred))
    print("prediction={}".format(y_pred_ascii))
