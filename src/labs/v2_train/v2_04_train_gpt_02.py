#%% 🚀 1. Import Libraries & Check GPU
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Path to dataset
DATA_PATH = "/mnt/e/MayThesis2025/cleanned_datasets/(37t)สถานีอุตุนิยมวิทยาลำปาง.csv"

# Force TensorFlow to use GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0

# Check if GPU is available
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("List of available devices:", tf.config.list_physical_devices())

# Enable memory growth for GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is set up and ready!")
    except RuntimeError as e:
        print(e)

#%% 📊 2. Load Dataset & Exploratory Data Analysis (EDA)
df = pd.read_csv(
    DATA_PATH,
    parse_dates=['Datetime'],  
    infer_datetime_format=True
)

df.sort_values(by='Datetime', inplace=True)
df.reset_index(drop=True, inplace=True)

# Quick dataset overview
print("Initial data shape:", df.shape)
print(df.info())
print("\nMissing values per column:\n", df.isnull().sum())

# Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

#%% 🛠️ 3. Handle Missing Values (Interpolate & Drop Bad Data)
df.dropna(how='all', inplace=True)  # Drop fully empty rows
df.interpolate(method='linear', inplace=True)  # Fill missing values using linear interpolation
df.dropna(subset=['SO2'], inplace=True)  # Ensure target column has no missing values

print("After cleaning, shape:", df.shape)
print("Any remaining NA?\n", df.isnull().sum())

#%% 🔄 4. Prepare Data for LSTM (Feature Engineering)
WINDOW_SIZE = 24   # Use past 24 hours of data
TARGET_SHIFT = 24  # Predict SO2 24 hours ahead

# Feature selection (exclude 'Datetime')
feature_cols = [col for col in df.columns if col not in ['Datetime']]
target_col = 'SO2'
target_col_index = feature_cols.index(target_col)

def create_sequences(data, target_index, window_size=24, shift=24):
    """Convert time series into supervised learning format"""
    X, y = [], []
    for i in range(len(data) - window_size - shift + 1):
        X_seq = data[i : i + window_size, :]
        y_val = data[i + window_size + shift - 1, target_index]
        X.append(X_seq)
        y.append(y_val)
    return np.array(X), np.array(y)

# Convert DataFrame to NumPy array
data_values = df[feature_cols].values
X_all, y_all = create_sequences(data_values, target_col_index, WINDOW_SIZE, TARGET_SHIFT)

print("X_all shape:", X_all.shape, "| y_all shape:", y_all.shape)

#%% 📉 5. Train/Test Split
train_ratio = 0.8
train_size = int(train_ratio * len(X_all))

X_train = X_all[:train_size]
y_train = y_all[:train_size]
X_test  = X_all[train_size:]
y_test  = y_all[train_size:]

print("Train set X:", X_train.shape, "y:", y_train.shape)
print("Test  set X:", X_test.shape,  "y:", y_test.shape)

#%% 🏗️ 6. Normalize Data (MinMax Scaling)
num_train, seq_len, num_feats = X_train.shape

# Flatten before fitting the scaler
X_train_2d = X_train.reshape(num_train * seq_len, num_feats)
X_test_2d  = X_test.reshape(X_test.shape[0] * seq_len, num_feats)

# Scale features
feat_scaler = MinMaxScaler()
feat_scaler.fit(X_train_2d)

X_train_scaled = feat_scaler.transform(X_train_2d).reshape(num_train, seq_len, num_feats)
X_test_scaled  = feat_scaler.transform(X_test_2d).reshape(X_test.shape[0], seq_len, num_feats)

# Scale target
y_scaler = MinMaxScaler()
y_scaler.fit(y_train.reshape(-1, 1))

y_train_scaled = y_scaler.transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled  = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

#%% 🔥 7. Define LSTM Model (GPU-Optimized)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(WINDOW_SIZE, num_feats),
         recurrent_activation="sigmoid", recurrent_dropout=0),
    Dropout(0.2),

    LSTM(128, return_sequences=False, recurrent_activation="sigmoid", recurrent_dropout=0),
    Dropout(0.2),

    Dense(64, activation='relu'),
    Dense(1)  # Output: Predicted SO2
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='mse',
    metrics=['mae']
)

model.summary()

#%% 🚀 8. Train the LSTM Model on GPU
EPOCHS = 50
BATCH_SIZE = 1024  # Large batch size for faster training on GPU

history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_data=(X_test_scaled, y_test_scaled),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

#%% 📈 9. Plot Training Curves
plt.figure(figsize=(12,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("MSE Loss During Training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.figure(figsize=(12,5))
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title("MAE During Training")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.legend()
plt.show()

#%% 🏆 10. Evaluate Model
test_mse, test_mae = model.evaluate(X_test_scaled, y_test_scaled, verbose=0)
print("Test MSE:", test_mse)
print("Test MAE:", test_mae)

# Predict
y_pred_scaled = model.predict(X_test_scaled)
y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_true = y_test

#%% 🎯 11. Plot Predictions vs Actual
plt.figure(figsize=(12,6))
plt.plot(y_true[:300], label="True SO2", marker='o')
plt.plot(y_pred[:300], label="Predicted SO2", marker='x')
plt.title("SO2 Prediction vs Actual (First 300 Test Points)")
plt.xlabel("Test Data Index")
plt.ylabel("SO2")
plt.legend()
plt.show()
