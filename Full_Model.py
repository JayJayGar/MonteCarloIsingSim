import torch
from sympy import false
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class FullModel(nn.Module):
    def __init__(self, n_inputs):
        super(FullModel, self).__init__()
        self.W1 = nn.Parameter(torch.randn(100, n_inputs) * 0.01)  # 100 hidden!
        self.b1 = nn.Parameter(torch.zeros(100))
        self.W2 = nn.Parameter(torch.randn(2, 100) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(2))

    def forward(self, x):
        hidden = torch.sigmoid(x @ self.W1.T + self.b1)
        output = hidden @ self.W2.T + self.b2
        return output

def trainmodel(train_configs, train_labels, epocs=200, lr=0.01):
    n_inputs = train_configs.shape[1]
    model = FullModel(n_inputs)

    X = torch.tensor(train_configs, dtype=torch.float32)
    y = torch.tensor(train_labels, dtype=torch.long)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epocs):
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if epoch % 10 == 0:
                print(f"epoch {epoch}, loss {epoch_loss}")
    return model

train_configs = np.load("TestSpins/train_configs.npy")
train_labels = np.load("TestSpins/train_labels.npy")
trained_model = trainmodel(train_configs, train_labels)
