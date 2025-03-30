# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

uri = r"e:\MayThesis2025\AiDataSample\(37t)ศาลหลักเมือง(ปิดสถานี).xlsx"
uri = r"E:\MayThesis2025\datasets\(37t)ศาลหลักเมือง(ปิดสถานี).xlsx"

# %%
# Load the data
df = pd.read_excel(uri, header=0, skiprows=0, engine="openpyxl")
df
# %%
# Data Preprocessing

# Rename columns for easier access (optional, but good practice)
df.rename(columns={"ปี/เดือน/วัน": "Date", "ชั่วโมง": "Hour"}, inplace=True)

#

# skip row 1
df = df.iloc[1:]
df

# make Date column as string no float digit
df["Date"] = df["Date"].astype(int).astype(str)
df["Hour"] = df["Hour"].astype(int).astype(str)


# %%
# -- Show the data info --
df.info()


# %%
# Convert Date and Hour to Datetime
def convert_datetime(row):
    date_str = str(row["Date"]).zfill(6)  # Pad date to 6 digits
    hour_str = str(row["Hour"]).zfill(4)  # Pad hour to 4 digits

    year_prefix = "20"  # Assuming years are in 2000s
    if date_str.startswith("9"):
        year_prefix = "20"

    year = year_prefix + date_str[:2]
    month = date_str[2:4]
    day = date_str[4:6]

    hour = hour_str[:2]
    minute = hour_str[2:]

    datetime_str = f"{year}-{month}-{day} {hour}:{minute}:00"

    try:
        return pd.to_datetime(datetime_str)
    except ValueError:
        return pd.NaT  # Handle potential parsing errors


df["Datetime"] = df.apply(convert_datetime, axis=1)
df.set_index("Datetime", inplace=True)


# %%
# -- Find max/min date --
start_date = df.index.min()
end_date = df.index.max()

# create a date range
date_range = pd.date_range(start=start_date, end=end_date, freq="h")

print("Duplicate timestamps:", df.index.duplicated().sum())

# Remove duplicates (choose one method)
df = df[~df.index.duplicated(keep="first")]  # Keep first occurrence
# df = df[~df.index.duplicated(keep='last')]  # Keep last occurrence
# df = df.groupby(df.index).mean()  # Average duplicate values


# Reindex the dataframe with the date range
df = df.reindex(date_range)
df.columns

# %%
# -- Clean columns name --
df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("/", "_")
df.columns
# %%
# Convert all data to numeric and replace non-numeric with NaN
df = df.apply(pd.to_numeric, errors="coerce")

# Handle missing values by filling with column means
df.fillna(df.mean(), inplace=True)

# %%
# Plot time series data
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["SO2"], label="SO2 Levels", color="b")
plt.xlabel("Date Time")
plt.ylabel("SO2 Concentration")
plt.title("Time Series of SO2 Levels")
plt.legend()
plt.grid()
plt.show()

# %%
# Replace empty strings with NaN and convert columns to numeric
for col in df.columns:
    if col not in []:  # No need to convert datetime index
        df[col] = df[col].replace(" ", np.nan)
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )  # Use coerce to turn non-numeric to NaN

# Handle missing values - Impute with mean for simplicity (can explore other strategies)
df.fillna(df.mean(), inplace=True)

# Select features and target
feature_columns = [
    "PM10",
    "CO",
    "NO",
    "NO2",
    "NOX",
    "O3",
    "Rain",
    "Net_rad",
    "Pressure",
    "Temp",
    "Rel_hum",
]
target_column = "SO2"

features = df[feature_columns].values
target = df[target_column].values


# Scale features and target
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

scaled_features = feature_scaler.fit_transform(features)
scaled_target = target_scaler.fit_transform(
    target.reshape(-1, 1)
)  # Reshape for scaling

# Prepare data for LSTM - create sequences (example: sequence length of 24 hours)
sequence_length = 24  # Look back 24 hours to predict next hour
X, y = [], []
for i in range(sequence_length, len(scaled_features)):
    X.append(scaled_features[i - sequence_length : i])
    y.append(scaled_target[i])

X = np.array(X)
y = np.array(y)

# Split data into training and testing sets (using 80% train, 20% test - time-based split)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM Model
model = Sequential()
model.add(
    LSTM(50, activation="relu", input_shape=(X_train.shape[1], X_train.shape[2]))
)  # 50 LSTM units
model.add(Dense(1))  # Output layer for single SO2 prediction
model.compile(optimizer="adam", loss="mse")  # Mean Squared Error loss

# Train the model
epochs = 50  # Number of training iterations
batch_size = 32  # Number of samples per gradient update
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

# Evaluate the model
loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Mean Squared Error on Test Set: {loss:.4f}")

# Make predictions
predictions = model.predict(X_test)
# Inverse transform predictions to original scale
predicted_so2 = target_scaler.inverse_transform(predictions)
actual_so2 = target_scaler.inverse_transform(y_test)

# Print some predictions vs actual values
print("\nSample Predictions (SO2):")
for i in range(10):  # Print first 10 predictions
    print(f"Predicted: {predicted_so2[i][0]:.2f}, Actual: {actual_so2[i][0]:.2f}")
