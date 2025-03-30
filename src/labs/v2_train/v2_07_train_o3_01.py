# %%
# Import libraries and check for GPU availability
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os

# Check if GPU is available
print("Evaluate: Checking GPU availability...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPUs available:", gpus)
else:
    print("No GPU detected; using CPU.")

# %%
# Load the dataset
data_path = '/mnt/e/MayThesis2025/cleanned_datasets/(37t)สถานีอุตุนิยมวิทยาลำปาง.csv'
# Parse the "Datetime" column into datetime objects.
df = pd.read_csv(data_path, parse_dates=['Datetime'])
print("Evaluate: Data loaded successfully with shape:", df.shape)
print("Dataset head:")
print(df.head())

# %%
# Exploratory Data Analysis (EDA)
print("Evaluate: Starting EDA...")
print("Data info:")
df.info()

print("\nSummary statistics:")
print(df.describe())

print("\nMissing values per column:")
print(df.isna().sum())

# Visualize missing data (optional)
plt.figure(figsize=(10, 4))
sns.heatmap(df.isna(), cbar=False, cmap='viridis')
plt.title("Missing Data Heatmap")
plt.show()

# %%
# Data Cleaning: Trim head/tail missing rows and fill missing values by linear interpolation
print("Evaluate: Cleaning data...")

# Drop rows that are completely empty
df = df.dropna(how='all')

# For forecasting SO₂, we want to ensure that the target is available.
df = df[df['SO2'].notna()]

# Since many columns have missing data at the beginning or end, we “trim” those rows if needed.
# (For example, you may choose to drop the first/last few rows if they are mostly NaN.
# Here we assume the remaining interior missing values can be filled.)
# Apply linear interpolation to fill missing values
df.interpolate(method='linear', inplace=True)
# In case any NaNs remain at the boundaries, use forward/backward fill.
df.fillna(method='bfill', inplace=True)
df.fillna(method='ffill', inplace=True)

print("Evaluate: Data cleaned.")
print("Missing values after cleaning:")
print(df.isna().sum())

# %%
# Feature Selection: Identify and order features by correlation with SO2
print("Evaluate: Computing correlation with SO2...")
corr_matrix = df.corr()
print("Correlation of all features with SO2:")
print(corr_matrix['SO2'].sort_values(ascending=False))

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()

# Exclude the target 'SO2' itself from feature selection.
sorted_features = corr_matrix['SO2'].drop('SO2').abs().sort_values(ascending=False).index.tolist()
print("Selected features (sorted by correlation with SO2):", sorted_features)

# %%
# Prepare the dataset for time series forecasting
print("Evaluate: Preparing time series data...")

# Set parameters: using a lookback window and forecast horizon (in hours)
lookback = 48          # use the past 48 hours as input
forecast_horizon = 24   # predict SO2 for the next 24 hours

# Select features based on the sorted order from correlation analysis.
# Here we use all available features (except 'Datetime' and target 'SO2') in the determined order.
data_features = df[sorted_features]
target = df['SO2']

# Scale features and target using StandardScaler
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(data_features)
y_scaled = scaler_y.fit_transform(target.values.reshape(-1, 1))

# Combine features and target into one array for easier sliding window creation.
# The last column will be SO2.
data_combined = np.hstack([X_scaled, y_scaled])
num_features = data_features.shape[1]
total_features = num_features + 1  # including SO2

# Function to create sliding windows for supervised learning
def create_windows(data, lookback, forecast_horizon):
    Xs, ys = [], []
    for i in range(len(data) - lookback - forecast_horizon + 1):
        Xs.append(data[i:(i+lookback), :])
        ys.append(data[i+lookback:i+lookback+forecast_horizon, -1])  # only SO2 as target
    return np.array(Xs), np.array(ys)

X, y = create_windows(data_combined, lookback, forecast_horizon)
print("Evaluate: Created sliding windows with shapes:", X.shape, y.shape)

# Split the data into training and testing sets based on time (80% train, 20% test)
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]
print("Evaluate: Train/test split done. Training samples:", X_train.shape[0], "Testing samples:", X_test.shape[0])

# %%
# Build the LSTM model using TensorFlow (with GPU support and new techniques)
print("Evaluate: Building the LSTM model...")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional

# Construct a model using Bidirectional LSTM layers and dropout for regularization.
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(lookback, total_features)),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(forecast_horizon)  # Output layer: predict 24 hours of SO2
])
model.compile(optimizer='adam', loss='mse')

print("Evaluate: Model summary:")
model.summary()

# %%
# Train the model using a large batch size for faster processing (using GPU)
print("Evaluate: Training the model...")
batch_size = 1024  # large batch size for faster training with ample data
epochs = 10        # set epochs (adjust as needed for convergence)

history = model.fit(X_train, y_train,
                    validation_split=0.1,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1)

# Plot training history
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()

# %%
# Evaluate the model on the test set
print("Evaluate: Evaluating model on test data...")
test_loss = model.evaluate(X_test, y_test, batch_size=batch_size)
print("Evaluate: Test Loss (MSE):", test_loss)

# %%
# Make predictions on the test set and show a prediction example for the next 24 hours
print("Evaluate: Making predictions on test set...")
y_pred = model.predict(X_test, batch_size=batch_size)

# Inverse transform the scaled predictions and true values for interpretability
y_pred_inv = scaler_y.inverse_transform(y_pred)
y_test_inv = scaler_y.inverse_transform(y_test)

# Plot the first test sample's 24-hour prediction vs. actual values
plt.figure(figsize=(10, 6))
plt.plot(range(forecast_horizon), y_test_inv[0], marker='o', label="Actual SO₂")
plt.plot(range(forecast_horizon), y_pred_inv[0], marker='x', label="Predicted SO₂")
plt.xlabel("Hours Ahead")
plt.ylabel("SO₂")
plt.title("Prediction Example: Next 24 Hours")
plt.legend()
plt.show()

print("Evaluate: Prediction example for first test sample:")
print("Actual SO₂ values:", y_test_inv[0])
print("Predicted SO₂ values:", y_pred_inv[0])

# %%
# Final Report Summary
print("Final Report:")
print("Data shape after cleaning:", df.shape)
print("Selected features (in order):", sorted_features)
print("Number of training samples:", X_train.shape[0])
print("Number of testing samples:", X_test.shape[0])
print("Final Test Loss (MSE):", test_loss)
