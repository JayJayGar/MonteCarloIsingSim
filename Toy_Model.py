# To be added
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class ToyModel(nn.Module):
    def __init__(self, n_inputs):
        super(ToyModel, self).__init__()
        self.W1 = nn.Parameter(torch.randn(3, n_inputs) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(3))

        self.W2 = nn.Parameter(torch.randn(2, 3) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(2))

    def forward(self, x):
        hidden = torch.sigmoid(x @ self.W1.T + self.b1)
        output = hidden @ self.W2.T + self.b2
        return output

def trainmodel(train_configs, train_labels, epochs=200, lr=0.001, lambda_reg=0.001):
    n_inputs = train_configs.shape[1]
    model = ToyModel(n_inputs)

    X = torch.tensor(train_configs, dtype=torch.float32)
    y = torch.tensor(train_labels, dtype=torch.long)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            outputs = model(batch_X)

            ce_loss = criterion(outputs, batch_y)
            l1_loss = lambda_reg * torch.norm(model.W1, 1)  # Penalize complex weights
            loss = ce_loss + l1_loss

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

with torch.no_grad():
    X = torch.tensor(train_configs, dtype=torch.float32)
    y = torch.tensor(train_labels, dtype=torch.long)

    outputs = trained_model(X)
    _, predicted = torch.max(outputs, 1)

    print("\n=== Model Evaluation ===")
    print("Predicted class distribution:", np.bincount(predicted.numpy()))
    print("Actual class distribution:", np.bincount(train_labels))

    accuracy = (predicted == y).float().mean()
    print(f"Training Accuracy: {accuracy.item() * 100:.2f}%")

print(trained_model.W1[0].mean())
print(trained_model.W1[1].mean())
print(trained_model.W1[2].mean())

torch.save(trained_model.state_dict(), "toy_model.pth")
