import torch
import torch.nn as nn
import utils
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from torch.utils import data
from data import MD2_Datasource

# MD2

class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 4*128),
            nn.Dropout(0.1),
            nn.Linear(4*128, 2*128),
            nn.Linear(2*128, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


learning_rate = 0.0005
num_iters = 10000
model = Model(128, 128)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

params = {'batch_size': 2048, 'num_workers': 50}
max_epochs = range(100)

train_source = MD2_Datasource()
train_loader = data.DataLoader(train_source, **params)

all_losses = []

for epoch in tqdm(max_epochs):
    # Training
    for x, y in train_loader:
        # Transfer to GPU
        output = model(x)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all_losses.append(loss.item())

plt.figure()
plt.plot(all_losses)
plt.show()

for test_index in range(10):
    x, y = utils.md2TrainingExample()
    print("=======")
    print("x={}; y={};".format(utils.convert_bits_2_hex(x), utils.convert_bits_2_hex(y)))
    y_pred = model(x)
    y_pred_ascii = utils.convert_bits_2_hex(utils.stabilize_prediction(y_pred))
    print("prediction={}".format(y_pred_ascii))
