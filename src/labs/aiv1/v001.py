import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load raw data
data = pd.read_csv("raw_data.csv", header=0)


# Fix Date Format
def fix_date(date):
    return datetime.strptime(f"20{int(date):06d}", "%Y%m%d")


data["Date"] = data["Date"].apply(fix_date)


# Fix Hour Format
def fix_hour(hour):
    hour = int(hour)
    if hour == 2400:
        return timedelta(days=1)  # Move to the next day
    return timedelta(hours=hour // 100, minutes=hour % 100)


data["Hour"] = data["Hour"].apply(fix_hour)

# Combine Date and Hour
data["Datetime"] = data["Date"] + data["Hour"]
data = data.drop(columns=["Date", "Hour"])

# Set Datetime as Index
data = data.set_index("Datetime")

# Handle Missing Data
data = data.interpolate(method="time")  # Interpolate based on time index

# Scale Data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)


# Create Time-Series Windows
def create_windows(data, window_size, predict_size):
    X, y = [], []
    for i in range(len(data) - window_size - predict_size + 1):
        X.append(data[i : i + window_size])
        y.append(
            data[i + window_size : i + window_size + predict_size, 0]
        )  # Assuming SO2 is column 0
    return np.array(X), np.array(y)


window_size = 48  # Use last 48 hours as input
predict_size = 24  # Predict next 24 hours
X, y = create_windows(scaled_data, window_size, predict_size)

# Train-Test Split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM Model
model = Sequential(
    [
        LSTM(
            64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])
        ),
        LSTM(32),
        Dense(24),  # Predict 24 values
    ]
)

model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Predictions
predictions = model.predict(X_test)
