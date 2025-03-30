# %% [markdown]
# # Predicting SO2 Levels Using LSTM with TensorFlow and CUDA
# This notebook builds a basic LSTM model to predict SO2 levels for the next 24 hours using a large batch size.

# %% [code]
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is set up and ready!")
    except RuntimeError as e:
        print(e)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("TensorFlow Version: ", tf.__version__)

# %% [code]
# Load and Explore Data (EDA)
print("Evaluate: Performing EDA...")
file_path = "/mnt/e/MayThesis2025/cleanned_datasets/(37t)สถานีอุตุนิยมวิทยาลำปาง.csv"
df = pd.read_csv(file_path, parse_dates=['Datetime'], index_col='Datetime')

print("Dataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nSummary Statistics:")
print(df.describe())

plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Data Heatmap")
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Matrix")
plt.show()

# %% [code]
# Data Preprocessing
print("Evaluate: Preparing Data...")
# Trim leading and trailing fully NaN rows
first_valid_idx = df.notna().any(axis=1).idxmax()
last_valid_idx = df.notna().any(axis=1)[::-1].idxmax()
df = df.loc[first_valid_idx:last_valid_idx]

# Linear interpolation for missing values
df = df.interpolate(method='linear', limit_direction='both')
print("Missing Values After Interpolation:")
print(df.isnull().sum())

# Add lagged SO2 features (past 24 hours)
for lag in range(1, 25):
    df[f'SO2_lag_{lag}'] = df['SO2'].shift(lag)

# Drop rows with NaN introduced by lagging
df = df.dropna()

# Feature selection (original features + lagged SO2)
features = df.drop(columns=['SO2']).columns
target = 'SO2'

# Normalize the data
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()
df[features] = scaler_features.fit_transform(df[features])
df[target] = scaler_target.fit_transform(df[[target]])

# Create sequences for LSTM
def create_sequences(data, seq_length, pred_length):
    X, y = [], []
    for i in range(len(data) - seq_length - pred_length + 1):
        X.append(data[features].iloc[i:i+seq_length].values)
        y.append(data[target].iloc[i+seq_length:i+seq_length+pred_length].values)
    return np.array(X), np.array(y)

seq_length = 24
pred_length = 24
X, y = create_sequences(df, seq_length, pred_length)

# Train-test split (80-20)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
test_dates = df.index[seq_length + train_size:seq_length + train_size + len(y_test) * pred_length]

print("Training Data Shape:", X_train.shape)
print("Test Data Shape:", X_test.shape)
print("Test Dates Length:", len(test_dates))

# %% [code]
# Build and Train LSTM Model
print("Evaluate: Building and Training Model...")
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(pred_length)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
model.summary()

batch_size = 1024  # Increased batch size for efficiency
history = model.fit(X_train, y_train, epochs=50, batch_size=batch_size, 
                    validation_split=0.2, verbose=1)

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Model Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# %% [code]
# Evaluate Model
print("Evaluate: Evaluating Model...")
y_pred = model.predict(X_test, batch_size=batch_size)

y_test_inv = scaler_target.inverse_transform(y_test.reshape(-1, pred_length))
y_pred_inv = scaler_target.inverse_transform(y_pred)

mse = mean_squared_error(y_test_inv, y_pred_inv)
mae = mean_absolute_error(y_test_inv, y_pred_inv)
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")

plt.figure(figsize=(15, 6))
plt.plot(y_test_inv.flatten(), label='Actual SO2', alpha=0.5)
plt.plot(y_pred_inv.flatten(), label='Predicted SO2', alpha=0.5)
plt.title("Actual vs Predicted SO2 (All Test Data)")
plt.xlabel("Time Step")
plt.ylabel("SO2 Level")
plt.legend()
plt.show()

month_hours = 30 * 24
month_steps = month_hours // pred_length
plt.figure(figsize=(15, 6))
plt.plot(y_test_inv[:month_steps].flatten(), label='Actual SO2', alpha=0.5)
plt.plot(y_pred_inv[:month_steps].flatten(), label='Predicted SO2', alpha=0.5)
plt.title("Actual vs Predicted SO2 (First Month of Test Data)")
plt.xlabel("Hour")
plt.ylabel("SO2 Level")
plt.legend()
plt.show()

# Find best and worst 1-month periods
monthly_mse = []
for i in range(0, len(y_test) - month_steps + 1, month_steps):
    start_idx = i
    end_idx = i + month_steps
    month_y_test = y_test_inv[start_idx:end_idx].flatten()
    month_y_pred = y_pred_inv[start_idx:end_idx].flatten()
    mse_month = mean_squared_error(month_y_test, month_y_pred)
    date_idx = start_idx * pred_length
    if date_idx < len(test_dates):
        monthly_mse.append((mse_month, test_dates[date_idx]))

if monthly_mse:
    best_month = min(monthly_mse, key=lambda x: x[0])
    worst_month = max(monthly_mse, key=lambda x: x[0])
    print(f"Best Month: {best_month[1]} (MSE: {best_month[0]:.4f})")
    print(f"Worst Month: {worst_month[1]} (MSE: {worst_month[0]:.4f})")

    best_start_idx = np.where(test_dates == best_month[1])[0][0] // pred_length
    best_y_test = y_test_inv[best_start_idx:best_start_idx + month_steps].flatten()
    best_y_pred = y_pred_inv[best_start_idx:best_start_idx + month_steps].flatten()
    plt.figure(figsize=(15, 6))
    plt.plot(best_y_test, label='Actual SO2', alpha=0.5)
    plt.plot(best_y_pred, label='Predicted SO2', alpha=0.5)
    plt.title(f"Best Performing Month ({best_month[1].strftime('%Y-%m')} - MSE: {best_month[0]:.4f})")
    plt.xlabel("Hour")
    plt.ylabel("SO2 Level")
    plt.legend()
    plt.show()

    worst_start_idx = np.where(test_dates == worst_month[1])[0][0] // pred_length
    worst_y_test = y_test_inv[worst_start_idx:worst_start_idx + month_steps].flatten()
    worst_y_pred = y_pred_inv[worst_start_idx:worst_start_idx + month_steps].flatten()
    plt.figure(figsize=(15, 6))
    plt.plot(worst_y_test, label='Actual SO2', alpha=0.5)
    plt.plot(worst_y_pred, label='Predicted SO2', alpha=0.5)
    plt.title(f"Worst Performing Month ({worst_month[1].strftime('%Y-%m')} - MSE: {worst_month[0]:.4f})")
    plt.xlabel("Hour")
    plt.ylabel("SO2 Level")
    plt.legend()
    plt.show()
else:
    print("Not enough data to evaluate monthly performance.")

# %% [code]
# Result Report and Prediction Example
print("Evaluate: Generating Results and Prediction Example...")
last_sequence = X[-1:]
pred_next_24 = model.predict(last_sequence)
pred_next_24_inv = scaler_target.inverse_transform(pred_next_24)

last_datetime = df.index[-1]
future_timestamps = pd.date_range(start=last_datetime, periods=pred_length+1, freq='H')[1:]

example_df = pd.DataFrame({
    'Datetime': future_timestamps,
    'Predicted_SO2': pred_next_24_inv.flatten()
})
print("\nPrediction Example for Next 24 Hours:")
print(example_df)

# %% [markdown]
# # Report Summary
# - **EDA**: Identified missing data patterns; handled via interpolation.
# - **Data Preparation**: Used original features plus 24-hour lagged SO2, trimmed NaN rows, interpolated, and normalized.
# - **Training**: LSTM trained on GPU with batch size 1024 over 50 epochs.
# - **Evaluation**: Visualizations for all test data, 1-month subset, and best/worst months (if applicable).
# - **Result**: Predicted SO2 for the next 24 hours.