# To be added
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from matplotlib import pyplot as plt

class ToyModel(nn.Module):
    def __init__(self, n_inputs):
        super(ToyModel, self).__init__()
        std = np.sqrt(2.0 / (n_inputs + 3))
        self.W1 = nn.Parameter(torch.randn(3, n_inputs) * std)
        self.b1 = nn.Parameter(torch.tensor([-1.0, 0.0, 1.0]))

        self.W2 = nn.Parameter(torch.randn(2, 3) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(2))

        self.n_inputs = n_inputs

    def forward(self, x):
        hidden = torch.sigmoid(x @ self.W1.T / np.sqrt(self.n_inputs) + self.b1)
        output = hidden @ self.W2.T + self.b2
        return output

def trainmodel(train_configs, train_labels, epochs=600, lr=0.01, lambda_reg=0.0):
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
            l1_loss = lambda_reg * (torch.norm(model.W1, 1) + torch.norm(model.W2, 1))
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

test_configs = np.load("TestSpins/test_configs.npy")
test_temps = np.load("TestSpins/test_temps.npy")

with torch.no_grad():
    X_test = torch.tensor(test_configs, dtype=torch.float32)
    outputs = trained_model(X_test)
    probabilities = torch.softmax(outputs, dim=1).numpy()

temps = test_temps

unique_temps = np.unique(test_temps)
avg_probs = np.zeros((len(unique_temps), 2))

for i, temp in enumerate(unique_temps):
    mask = test_temps == temp
    avg_probs[i] = probabilities[mask].mean(axis=0)

plt.figure(figsize=(5, 4))
plt.plot(unique_temps, avg_probs[:, 0], 'b^-', label='T < Tc')
plt.plot(unique_temps, avg_probs[:, 1], 'ro-', label='T > Tc')
plt.show()
