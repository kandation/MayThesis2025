import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.notebook import tqdm  # For progress bars
from datetime import timedelta


# --- 1. Data Loading and Preprocessing ---


def load_and_preprocess(filepath, fill_method="linear"):
    """
    Loads, preprocesses, and imputes missing data in the weather data CSV.

    Args:
        filepath (str): Path to the CSV file.
        fill_method (str): Method for imputing missing values ('linear', 'ffill', 'bfill', or a number).

    Returns:
        pandas.DataFrame: The preprocessed DataFrame.
    """

    df = pd.read_csv(filepath, parse_dates=[0], index_col=0, na_values=[",,,,,,,,,"])

    # Convert all columns to numeric, coercing errors to NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Imputation
    if fill_method == "linear":
        df = df.interpolate(method="linear", limit_direction="both")
    elif fill_method == "ffill":
        df = df.ffill()
    elif fill_method == "bfill":
        df = df.bfill()
    elif isinstance(fill_method, (int, float)):
        df = df.fillna(fill_method)
    else:
        raise ValueError(
            "Invalid fill_method. Choose 'linear', 'ffill', 'bfill', or a numeric value."
        )

    if df.isnull().any().any():
        raise ValueError(
            "NaN values remain after imputation.  Check data and fill_method."
        )

    return df


# --- 2. Feature Engineering ---
def feature_engineering(df):
    """
    Creates new features, including cyclical time features and lagged features.
    """
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["dayofyear"] = df.index.dayofyear
    df["month"] = df.index.month
    df["year"] = df.index.year

    # Cyclical time features (sin/cos transformations)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["dayofyear_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["dayofyear_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Add lag features for SO2 (and potentially other key features).  Target variable lags!
    for i in range(1, 25):  # Lags up to 24 hours.
        df[f"SO2_lag_{i}"] = df["SO2"].shift(i)

    # One-Hot Encode cyclical time features (alternative to sin/cos)
    df = pd.get_dummies(
        df, columns=["hour", "dayofweek", "dayofyear", "month"], drop_first=True
    )

    # Drop rows with NaN values created by lagging
    df = df.dropna()
    return df


# --- 3. Feature Selection ---
def select_features(df, target_col="SO2", corr_threshold=0.05):
    """
    Selects features based on correlation with the target variable.

    Args:
        df (pd.DataFrame): The DataFrame with features.
        target_col (str): The name of the target variable column.
        corr_threshold (float): The minimum absolute correlation to include a feature.

    Returns:
        list: A list of selected feature names.
    """

    correlations = df.corr()[target_col].abs()
    selected_features = correlations[correlations >= corr_threshold].index.tolist()
    selected_features.remove(target_col)  # Don't include the target in the features

    # --- Feature Importance using a Tree-based Model (e.g., RandomForestRegressor)---
    from sklearn.ensemble import RandomForestRegressor

    X = df.drop(columns=[target_col])
    y = df[target_col]

    model = RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    )  # Use all cores
    model.fit(X, y)

    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    important_features = feature_importances.sort_values(ascending=False).index.tolist()

    # Combine Correlation and Feature Importance
    final_features = []
    for feature in important_features:
        if feature in selected_features:
            final_features.append(feature)

    print(f"Selected features based on correlation and importance: {final_features}")
    return final_features


# --- 4. Data Scaling ---


def scale_data(df, selected_features, target_col="SO2"):
    """
    Scales the selected features and the target variable using MinMaxScaler.

    Args:
        df: DataFrame
        selected_features: list
        target_col: str

    Returns:
        train_scaled: Scaled training data.
        test_scaled: Scaled test data.
        target_scaler: Scaler used for target variable, for inverse scaling.
        feature_scaler: Scaler object used for features
    """

    # Split data *before* scaling to prevent data leakage
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, shuffle=False
    )

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    # Scale features
    train_scaled = feature_scaler.fit_transform(train_df[selected_features])
    test_scaled = feature_scaler.transform(test_df[selected_features])

    # Scale target separately
    train_target_scaled = target_scaler.fit_transform(train_df[[target_col]])
    test_target_scaled = target_scaler.transform(test_df[[target_col]])

    # Combine scaled features and target
    train_scaled = np.concatenate([train_scaled, train_target_scaled], axis=1)
    test_scaled = np.concatenate([test_scaled, test_target_scaled], axis=1)

    return train_scaled, test_scaled, target_scaler, feature_scaler


# --- 5. Data Sequencing ---


class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length, y_size):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_length = seq_length
        self.y_size = y_size

    def __len__(self):
        return len(self.data) - self.seq_length - self.y_size + 1

    def __getitem__(self, idx):
        x = self.data[
            idx : idx + self.seq_length, :-1
        ]  # All features, up to sequence length
        y = self.data[
            idx + self.seq_length : idx + self.seq_length + self.y_size, -1
        ]  # Target for next 24 hrs
        return x, y


def create_data_loaders(train_scaled, test_scaled, seq_length, y_size, batch_size):
    """
    Creates training and testing data loaders.

    Args:
        train_scaled (np.ndarray): Scaled training data.
        test_scaled (np.ndarray): Scaled testing data.
        seq_length (int): The input sequence length.
        y_size (int): The output sequence length (prediction horizon).
        batch_size (int): The batch size.

    Returns:
        tuple: (train_loader, test_loader)
    """
    train_dataset = TimeSeriesDataset(train_scaled, seq_length, y_size)
    test_dataset = TimeSeriesDataset(test_scaled, seq_length, y_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
    )  # Shuffling is bad for timeseries. Drop last incomplete batch.
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, test_loader


# --- 6. LSTM Model Definition ---


class LSTMModel(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.2
    ):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate
        )
        self.linear = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)  # Add a dropout layer
        self.relu = nn.ReLU()  # Add ReLU activation

    def forward(self, x):
        # Initialize hidden and cell states with proper batch size and device.
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))  # Pass hidden and cell states
        out = self.dropout(out)  # Apply dropout
        out = self.linear(out[:, -1, :])  # Use only the last time step's output.
        out = self.relu(out)  # Apply ReLU
        return out


# --- 7. Training Loop ---


def train_model(
    model,
    train_loader,
    test_loader,
    optimizer,
    criterion,
    num_epochs,
    device,
    patience=10,
    target_scaler=None,
):
    """
    Trains the LSTM model.

    Args:
        model: The LSTM model.
        train_loader: Training data loader.
        test_loader: Testing data loader.
        optimizer: The optimizer.
        criterion: The loss function.
        num_epochs: Number of training epochs.
        device: 'cuda' or 'cpu'.
        patience: Patience for early stopping.

    Returns:
        model: The trained model.
        train_losses: List of training losses.
        test_losses: List of validation losses.
    """

    train_losses = []
    test_losses = []
    best_test_loss = float("inf")
    early_stopping_counter = 0

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model.train()
        running_train_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )  # Gradient clipping
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Validation loop
        model.eval()
        running_test_loss = 0.0
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Validation", leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_test_loss += loss.item() * inputs.size(0)

        epoch_test_loss = running_test_loss / len(test_loader.dataset)
        test_losses.append(epoch_test_loss)

        scheduler.step(epoch_test_loss)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Test Loss: {epoch_test_loss:.4f}"
        )

        # Early stopping check
        if epoch_test_loss < best_test_loss:
            best_test_loss = epoch_test_loss
            early_stopping_counter = 0
            # Save the best model so far.
            torch.save(model.state_dict(), "best_model.pth")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping triggered.")
                break

    # Load the best model
    model.load_state_dict(torch.load("best_model.pth"))
    return model, train_losses, test_losses


# --- 8. Evaluation ---


def evaluate_model(model, test_loader, criterion, device, target_scaler):
    """Evaluates the model and returns predictions and actual values."""
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
            actuals.append(labels.cpu().numpy())

    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)

    # Inverse transform the predictions and actual values
    predictions = target_scaler.inverse_transform(predictions)
    actuals = target_scaler.inverse_transform(actuals)

    return predictions, actuals


# --- 9. Plotting ---
def plot_results(train_losses, test_losses, predictions, actuals, y_size):
    """Plots training and validation losses, and a sample of predictions."""

    # Plot training and validation loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")

    # Plot predictions vs actuals
    plt.subplot(1, 2, 2)

    # Show a sample of the predictions (e.g., the first 100 time steps)
    sample_size = 100
    if len(actuals) >= sample_size:
        plot_actuals = actuals[:sample_size, :].flatten()  # Flatten to a 1D array.
        plot_predictions = predictions[:sample_size, :].flatten()

        plt.plot(plot_actuals, label="Actual Values")
        plt.plot(plot_predictions, label="Predictions")

        plt.xlabel("Time Steps (Hours)")  # Be specific about the time unit.
        plt.ylabel("SO2 Concentration")
        plt.legend()
        plt.title("Predictions vs Actual Values (Sample)")
    else:
        plt.text(
            0.5,
            0.5,
            "Not enough data to plot",
            horizontalalignment="center",
            verticalalignment="center",
        )

    plt.tight_layout()  # Prevents overlap of subplots
    plt.show()


# --- 10. Main Execution ---

if __name__ == "__main__":
    filepath = r"E:\MayThesis2025\cleanned_datasets\(37t)ศาลหลักเมือง(ปิดสถานี).csv"
    df = load_and_preprocess(
        filepath, fill_method="linear"
    )  # Or 'ffill', 'bfill', or a number.
    df = feature_engineering(df)
    selected_features = select_features(df, target_col="SO2")
    train_scaled, test_scaled, target_scaler, feature_scaler = scale_data(df, selected_features, target_col='SO2')

    # --- Hyperparameters ---
    seq_length = 72  # Input sequence length (lookback window)
    y_size = 24     # Output sequence length (prediction horizon)
    batch_size = 2048  # Use a large batch size.  Powers of 2 are generally preferred.
    hidden_size = 128 # Number of units in the LSTM layer.
    num_layers = 3    # Number of LSTM layers.
    learning_rate = 0.0005
    num_epochs = 100
    input_size = len(selected_features)  # Number of input features.
    output_size = y_size  # Predict SO2 for the next 24 hours
    dropout_rate = 0.2  # Dropout rate for regularization

    train_loader, test_loader = create_data_loaders(train_scaled, test_scaled, seq_length, y_size, batch_size)

    # --- Device Configuration ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Model, Optimizer, Loss ---
    model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5) # Add L2 regularization (weight decay)
    criterion = nn.MSELoss()  # Mean Squared Error Loss

    # --- Training ---
    model, train_losses, test_losses = train_model(model, train_loader, test_loader, optimizer, criterion, num_epochs, device, patience=15, target_scaler=target_scaler)

    # --- Evaluation ---
    predictions, actuals = evaluate_model(model, test_loader, criterion, device, target_scaler)

    # --- Plotting ---
    plot_results(train_losses, test_losses, predictions, actuals, y_size)

    # --- Save Scalers and Feature List---
    import joblib
    joblib.dump(target_scaler, 'target_scaler.pkl')
    joblib.dump(feature_scaler, 'feature_scaler.pkl')
    joblib.dump(selected_features, 'selected_features.pkl')

    # --- Make and save Predictions on the whole dataset (Optional but Useful) ---
    # Create a DataLoader for entire dataframe
    all_scaled = feature_scaler.transform(df[selected_features])
    all_target_scaled = target_scaler.transform(df[['SO2']])
    all_scaled = np.concatenate([all_scaled, all_target_scaled], axis = 1)
    all_dataset = TimeSeriesDataset(all_scaled, seq_length, y_size)
    all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False, drop_last=False)  # drop_last = False to get all predictions

    all_predictions, all_actuals = evaluate_model(model, all_loader, criterion, device, target_scaler)
    
    # Create a new DataFrame for predictions with proper DatetimeIndex
    # Ensure the index aligns correctly after accounting for seq_length and y_size.
    pred_index = df.index[seq_length + y_size -1:]
    if len(pred_index) > len(all_predictions):
         pred_index = pred_index[:len(all_predictions)]
    elif len(pred_index) < len(all_predictions):
         all_predictions = all_predictions[:len(pred_index)]


    predictions_df = pd.DataFrame(all_predictions, index=pred_index)
    predictions_df.columns = [f'pred_SO2_{i+1}' for i in range(y_size)]

    # Save the predictions
    predictions_df.to_csv('so2_predictions.csv')
    print("Predictions saved to so2_predictions.csv")
