import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Load dataset
def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=["Datetime"], index_col="Datetime")
    return df


# Handle missing data
def preprocess_data(df):
    df.interpolate(
        method="time", inplace=True
    )  # Interpolate missing values using time-based interpolation
    df.fillna(
        method="bfill", inplace=True
    )  # Fill remaining missing values with backward fill
    return df


# Feature selection based on correlation with SO2
def select_features(df, target="SO2", threshold=0.2):
    correlation = df.corr()[target].abs().sort_values(ascending=False)
    selected_features = correlation[correlation > threshold].index.tolist()
    if target not in selected_features:
        selected_features.append(target)
    return df[selected_features]


# Normalize data
def normalize_data(df):
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df), columns=df.columns, index=df.index
    )
    return df_scaled, scaler


# Create time series sequences
def create_sequences(data, target_column, past_hours=48, future_hours=24):
    sequences, targets = [], []
    for i in range(len(data) - past_hours - future_hours):
        seq = data.iloc[i : i + past_hours].values
        target = data.iloc[i + past_hours : i + past_hours + future_hours][
            target_column
        ].values
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)


# Define PyTorch Dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(
        self, input_size, hidden_size=128, num_layers=2, output_size=24, dropout=0.2
    ):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Take the last time step's output
        return out


# Training function
def train_model(model, train_loader, val_loader, epochs=50, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}"
        )

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.show()


# Load and preprocess data
df = load_data(
    r"E:\MayThesis2025\cleanned_datasets\(37t)ศาลหลักเมือง(ปิดสถานี).csv"
)  # Update with the actual filename
df = preprocess_data(df)
df = select_features(df)
df_scaled, scaler = normalize_data(df)

# Create sequences
sequences, targets = create_sequences(df_scaled, target_column="SO2")

# Split data into train and validation sets
train_seq, val_seq, train_target, val_target = train_test_split(
    sequences, targets, test_size=0.2, random_state=42
)

# Create PyTorch datasets and dataloaders
train_dataset = TimeSeriesDataset(train_seq, train_target)
val_dataset = TimeSeriesDataset(val_seq, val_target)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# Initialize model
input_size = train_seq.shape[2]
model = LSTMModel(input_size).to(device)

# Train model
train_model(model, train_loader, val_loader, epochs=50)

# Test prediction
model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        predictions.append(outputs.cpu().numpy())
        actuals.append(targets.cpu().numpy())

predictions = np.concatenate(predictions, axis=0)
actuals = np.concatenate(actuals, axis=0)

# Plot predictions vs actual
plt.figure(figsize=(10, 5))
plt.plot(actuals[:100, 0], label="Actual SO2")
plt.plot(predictions[:100, 0], label="Predicted SO2")
plt.xlabel("Time Steps")
plt.ylabel("SO2 Level")
plt.legend()
plt.title("Actual vs Predicted SO2 Levels")
plt.show()

# Save trained model
torch.save(model.state_dict(), "lstm_so2_model.pth")
print("Model training complete and saved!")
