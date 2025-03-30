# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split


# %%
# --- Data Loading and Preprocessing ---
def preprocess_data(file_path):
    print("--- Starting Data Preprocessing ---")
    df = pd.read_excel(file_path, header=0, engine="openpyxl")
    print("Raw DataFrame head:")
    print(df.head())
    print("\nRaw DataFrame info:")
    print(df.info())

    # Rename columns for easier access (remove Thai and units)
    df.columns = [
        "Date_raw",
        "Hour_raw",
        "PM10",
        "CO",
        "NO",
        "NO2",
        "NOX",
        "SO2",
        "O3",
        "Rain",
        "Net_rad",
        "Pressure",
        "Temp",
        "Rel_hum",
    ]

    # Drop unit row
    df = df.iloc[1:]
    df = df.reset_index(drop=True)

    # Convert Date
    def convert_date(date_raw):
        date_str = str(int(date_raw))
        year_prefix = "200" if date_str.startswith("9") else "20"
        year = year_prefix + date_str[:2]
        month = date_str[2:4]
        day = date_str[4:]
        return f"{year}-{month}-{day}"

    df["Date"] = df["Date_raw"].apply(convert_date)

    # Convert Hour
    def convert_hour(hour_raw):
        hour_str = str(int(hour_raw))
        if hour_str == "2400":
            return "0000"  # Handled in datetime creation later
        return hour_str.zfill(4)

    df["Hour"] = df["Hour_raw"].apply(convert_hour)

    # Create Datetime and set as index
    df["Datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Hour"], format="%Y-%m-%d %H%M"
    )

    # Handle 2400 hour (shift to next day)
    for i in range(len(df)):
        if df["Hour_raw"].iloc[i] == 2400:
            df["Datetime"].iloc[i] += pd.Timedelta(days=1)

    df = df.set_index("Datetime")

    # Convert columns to numeric, errors='coerce' will turn non-numeric to NaN
    numeric_cols = [
        "PM10",
        "CO",
        "NO",
        "NO2",
        "NOX",
        "SO2",
        "O3",
        "Rain",
        "Net_rad",
        "Pressure",
        "Temp",
        "Rel_hum",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Forward fill missing values (can be adjusted)
    df = df.ffill()  # or df = df.fillna(df.mean()) for mean imputation

    df = df.drop(
        ["Date_raw", "Hour_raw", "Date", "Hour"], axis=1
    )  # Drop raw and intermediate columns

    print("\nProcessed DataFrame head:")
    print(df.head())
    print("\nProcessed DataFrame info:")
    print(df.info())
    print("--- Data Preprocessing Complete ---")
    return df


# %%
file_path = r"e:\MayThesis2025\AiDataSample\(37t)ศาลหลักเมือง(ปิดสถานี).xlsx"  # Replace with your file path
df_processed = preprocess_data(file_path)

# %%
# --- Feature Scaling ---
print("--- Starting Feature Scaling ---")
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_processed)
scaled_df = pd.DataFrame(
    scaled_data, columns=df_processed.columns, index=df_processed.index
)

print("\nScaled DataFrame head:")
print(scaled_df.head())
print("\nScaled DataFrame describe:")
print(scaled_df.describe())
print("--- Feature Scaling Complete ---")


# %%
# --- Data Sequencing for LSTM ---
def create_sequences(data, sequence_length, target_column="SO2"):
    print("--- Starting Sequence Creation ---")
    sequences = []
    targets = []
    data_np = data.values  # Use numpy array for efficiency
    target_index = list(data.columns).index(target_column)

    for i in range(len(data_np) - sequence_length):
        seq = data_np[i : i + sequence_length]
        target = data_np[
            i + sequence_length : i + sequence_length + 24, target_index
        ]  # Predict next 24 hours of SO2
        if len(target) == 24:  # Ensure we have 24 target values
            sequences.append(seq)
            targets.append(target)

    print("--- Sequence Creation Complete ---")
    return np.array(sequences), np.array(targets)


# %%
sequence_length = 24  # Use 24 hours of history to predict
X, y = create_sequences(scaled_df, sequence_length)

print("\nInput sequences (X) shape:", X.shape)
print("Target sequences (y) shape:", y.shape)
print("\nSample input sequence (X[0]):")
print(X[0])
print("\nSample target sequence (y[0]):")
print(y[0])

# %%
# --- Split Data into Training and Testing ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

print("\nX_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# %%
# --- Build LSTM Model ---
print("--- Building LSTM Model ---")
n_features = X_train.shape[2]  # number of features
n_output_steps = 24  # predict 24 hours

model = Sequential()
model.add(
    LSTM(50, activation="relu", input_shape=(sequence_length, n_features))
)  # Single LSTM layer
model.add(Dense(n_output_steps))  # Output layer for 24 hour prediction

model.compile(optimizer="adam", loss="mse")  # Mean Squared Error loss
print("LSTM Model Compiled")
print(model.summary())
print("--- LSTM Model Built ---")

# %%
# --- Train the Model ---
print("--- Starting Model Training ---")
epochs = 50  # You can adjust epochs
batch_size = 32  # You can adjust batch size
history = model.fit(
    X_train,
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.2,
    verbose=1,
)
print("--- Model Training Complete ---")

# %%
# --- Evaluate the Model (Optional) ---
print("--- Evaluating Model ---")
loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss}")
print("--- Model Evaluation Complete ---")

# %%
# --- Make Predictions ---
print("--- Making Predictions ---")
# Get the last sequence from the training data to start prediction for demonstration
last_sequence = X_test[-1:]  # Take the last sequence from test data for prediction
predicted_so2_scaled = model.predict(last_sequence)
print("\nScaled Predicted SO2 (predicted_so2_scaled):")
print(predicted_so2_scaled)

predicted_so2 = scaler.inverse_transform(
    np.concatenate(
        [
            np.zeros((1, scaled_data.shape[1] - 1)),
            predicted_so2_scaled.reshape(1, -1).T,
        ],
        axis=1,
    )
)[:, -1]

print("\nInverse Transformed Predicted SO2 (predicted_so2):")
print(predicted_so2)

print("\nPredicted SO2 values for the next 24 hours:")
print(predicted_so2)
print("--- Predictions Complete ---")
