import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt

# ========================
# 1. Generate example data
# ========================
samples = 1000
timesteps = 10
features1 = 3
features2 = 2

X1 = np.random.rand(samples, timesteps, features1).astype(np.float32)
X2 = np.random.rand(samples, timesteps, features2).astype(np.float32)
y = np.random.rand(samples, 1).astype(np.float32)  # regression target

# Convert to tensors
X1 = torch.tensor(X1)
X2 = torch.tensor(X2)
y = torch.tensor(y)

# ========================
# 2. Create dataset and split
# ========================
dataset = TensorDataset(X1, X2, y)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ========================
# 3. Define 2-input LSTM model
# ========================
class TwoInputLSTM(nn.Module):
    def __init__(self, input_size1, input_size2, hidden_size1=64, hidden_size2=32):
        super(TwoInputLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size1, hidden_size1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size2, hidden_size2, batch_first=True)
        self.fc1 = nn.Linear(hidden_size1 + hidden_size2, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x1, x2):
        _, (h1, _) = self.lstm1(x1)
        _, (h2, _) = self.lstm2(x2)
        h1 = h1[-1]
        h2 = h2[-1]
        merged = torch.cat([h1, h2], dim=1)
        x = torch.relu(self.fc1(merged))
        out = self.fc2(x)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TwoInputLSTM(features1, features2).to(device)

# Loss function (we still use MSE internally) and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ========================
# 4. Train the model with RMSE tracking
# ========================
epochs = 50
train_rmse = []
val_rmse = []

for epoch in range(epochs):
    model.train()
    running_loss = 0
    for batch_X1, batch_X2, batch_y in train_loader:
        batch_X1, batch_X2, batch_y = batch_X1.to(device), batch_X2.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X1, batch_X2)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_X1.size(0)

    # Compute RMSE for training set
    train_loss = running_loss / train_size
    train_rmse.append(np.sqrt(train_loss))

    # Validation
    model.eval()
    val_running_loss = 0
    with torch.no_grad():
        for batch_X1, batch_X2, batch_y in val_loader:
            batch_X1, batch_X2, batch_y = batch_X1.to(device), batch_X2.to(device), batch_y.to(device)
            outputs = model(batch_X1, batch_X2)
            loss = criterion(outputs, batch_y)
            val_running_loss += loss.item() * batch_X1.size(0)

    val_loss = val_running_loss / val_size
    val_rmse.append(np.sqrt(val_loss))

    print(f"Epoch {epoch+1}/{epochs} - Train RMSE: {train_rmse[-1]:.4f} - Val RMSE: {val_rmse[-1]:.4f}")

# ========================
# 5. Plot training and validation RMSE
# ========================
plt.figure(figsize=(8,5))
plt.plot(train_rmse, label='Train RMSE')
plt.plot(val_rmse, label='Validation RMSE')
plt.title('Training and Validation RMSE')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()
plt.show()