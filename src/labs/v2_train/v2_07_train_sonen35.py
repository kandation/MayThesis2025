# %% Import required libraries
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta

print("TensorFlow version:", tf.__version__)

# Configure GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is set up and ready!")
    except RuntimeError as e:
        print(e)

# %% Load and inspect data
def load_data(filepath):
    print("Loading and inspecting data...")
    df = pd.read_csv(filepath, parse_dates=['Datetime'])
    print("\nDataset shape:", df.shape)
    print("\nMissing values:\n", df.isnull().sum())
    return df

# Load the data
df = load_data('/mnt/e/MayThesis2025/cleanned_datasets/(37t)สถานีอุตุนิยมวิทยาลำปาง.csv')

# %% EDA and Data Visualization
def perform_eda(df):
    print("Performing exploratory data analysis...")
    
    # Calculate missing values percentage
    missing_percentages = (df.isnull().sum() / len(df)) * 100
    print("\nMissing values percentage:\n", missing_percentages)
    
    # Plot missing values heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.show()
    
    # Time series plot for SO2
    plt.figure(figsize=(15, 5))
    plt.plot(df['Datetime'], df['SO2'])
    plt.title('SO2 Levels Over Time')
    plt.xlabel('Date')
    plt.ylabel('SO2')
    plt.show()
    
    # Correlation matrix
    correlation = df.select_dtypes(include=[np.number]).corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.show()
    
    return correlation

correlation = perform_eda(df)

# %% Data Preparation
def prepare_data(df):
    print("Preparing data...")
    
    # Trim head and tail where all values are NaN
    df = df.dropna(how='all')
    
    # Linear interpolation for missing values
    df_interpolated = df.interpolate(method='linear', limit_direction='both')
    
    # Select features based on correlation with SO2
    feature_cols = ['SO2', 'NO2', 'NO', 'NOX', 'O3', 'PM10', 'CO', 
                   'Net_rad', 'Pressure', 'Temp', 'Rel_hum']
    
    # Store original SO2 values for later use
    original_so2 = df_interpolated['SO2'].copy()
    
    # Scale the features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_interpolated[feature_cols])
    scaled_df = pd.DataFrame(scaled_data, columns=feature_cols)
    
    return scaled_df, scaler, original_so2, df_interpolated['Datetime']

scaled_df, scaler, original_so2, timestamps = prepare_data(df)

# %% Create sequences for LSTM
def create_sequences(data, seq_length, target_col='SO2'):
    print("Creating sequences for LSTM...")
    
    X, y = [], []
    for i in range(len(data) - seq_length - 24):  # 24 for next 24 hours prediction
        X.append(data.iloc[i:(i + seq_length)].values)
        y.append(data[target_col].iloc[i + seq_length:i + seq_length + 24].values)
    
    return np.array(X), np.array(y)

# Create sequences with 48-hour lookback
sequence_length = 48
X, y = create_sequences(scaled_df, sequence_length)

# Split data into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print("Training set shape:", X_train.shape, y_train.shape)
print("Test set shape:", X_test.shape, y_test.shape)

# %% Build and train LSTM model
def build_model(input_shape, output_shape):
    print("Building LSTM model...")
    
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(output_shape)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Build model
model = build_model((sequence_length, X_train.shape[2]), 24)
model.summary()

# Train model with GPU acceleration
print("Training model...")
batch_size = 1024  # Large batch size for GPU optimization
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=batch_size,
    validation_split=0.2,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]
)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.tight_layout()
plt.show()

# %% Evaluate model
def evaluate_model(model, X_test, y_test, scaler, timestamps):
    print("Evaluating model...")
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Inverse transform predictions and actual values for SO2
    predictions_unscaled = predictions / scaler.scale_[0]  # Inverse transform
    y_test_unscaled = y_test / scaler.scale_[0]  # Inverse transform
    
    # Calculate metrics
    mse = mean_squared_error(y_test_unscaled, predictions_unscaled)
    mae = mean_absolute_error(y_test_unscaled, predictions_unscaled)
    r2 = r2_score(y_test_unscaled.flatten(), predictions_unscaled.flatten())
    
    print("\nTest Set Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    # Plot actual vs predicted (full test set)
    plt.figure(figsize=(15, 6))
    plt.plot(y_test_unscaled.flatten(), label='Actual', alpha=0.5)
    plt.plot(predictions_unscaled.flatten(), label='Predicted', alpha=0.5)
    plt.title('Actual vs Predicted SO2 Levels (Full Test Set)')
    plt.xlabel('Time Steps')
    plt.ylabel('SO2 (Real Values)')
    plt.legend()
    plt.show()
    
    # Plot one month prediction
    month_steps = 24 * 30  # 30 days
    plt.figure(figsize=(15, 6))
    plt.plot(y_test_unscaled[:month_steps//24].flatten(), label='Actual', alpha=0.5)
    plt.plot(predictions_unscaled[:month_steps//24].flatten(), label='Predicted', alpha=0.5)
    plt.title('Actual vs Predicted SO2 Levels (One Month)')
    plt.xlabel('Time Steps')
    plt.ylabel('SO2 (Real Values)')
    plt.legend()
    plt.show()
    
    return predictions_unscaled

predictions = evaluate_model(model, X_test, y_test, scaler, timestamps)

# %% Example prediction with real values
def make_sample_prediction(model, recent_data, scaler, timestamps):
    print("Making sample prediction...")
    
    # Prepare input data
    input_sequence = recent_data[-sequence_length:].values.reshape(1, sequence_length, -1)
    
    # Make prediction
    prediction = model.predict(input_sequence)
    
    # Inverse transform prediction to real values
    prediction_unscaled = prediction / scaler.scale_[0]
    
    # Create timestamp index for next 24 hours
    last_timestamp = timestamps.iloc[-1]
    future_timestamps = [last_timestamp + timedelta(hours=x) for x in range(1, 25)]
    
    # Plot prediction
    plt.figure(figsize=(12, 5))
    plt.plot(future_timestamps, prediction_unscaled[0], marker='o')
    plt.title('24-Hour SO2 Forecast (Real Values)')
    plt.xlabel('Time')
    plt.ylabel('SO2 Level')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return prediction_unscaled[0]

# Make a sample prediction
sample_prediction = make_sample_prediction(model, scaled_df, scaler, timestamps)
print("\nPredicted SO2 levels for next 24 hours (Real Values):", sample_prediction)

# %% Save model
model.save('so2_prediction_model.h5')
print("Model saved as 'so2_prediction_model.h5'")