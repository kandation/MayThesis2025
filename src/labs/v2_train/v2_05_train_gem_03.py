import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
from tensorflow.keras.optimizers import Adam


# --- 1. Data Loading and Preprocessing ---

def load_and_preprocess(file_path):
    """Loads, preprocesses (trims leading/trailing all-NaN rows), and imputes."""
    df = pd.read_csv(file_path)

    # Convert 'Datetime' to datetime objects and handle errors
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df.dropna(subset=['Datetime'], inplace=True)
    df.set_index('Datetime', inplace=True)

    # --- Trim Leading and Trailing All-NaN Rows ---
    # Drop rows where ALL values are NaN, from the beginning
    first_valid_index = df.first_valid_index()
    if first_valid_index is not None:
        df = df.loc[first_valid_index:]

    # Drop rows where ALL values are NaN, from the end
    last_valid_index = df.last_valid_index()
    if last_valid_index is not None:
        df = df.loc[:last_valid_index]
    # --- End Trimming ---


    # Linear interpolation
    df.interpolate(method='linear', limit_direction='both', inplace=True)

    # Handle remaining NaNs (should be none, but check)
    if df.isnull().any().any():
        print("Warning: NaNs remain after interpolation.  Using ffill and bfill.")
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        if df.isnull().any().any():
            raise ValueError("NaNs persist after ffill and bfill.  Review data cleaning.")

    return df


# --- 2. Feature Selection and Engineering --- (No changes here)

def feature_selection_and_scaling(df):
    """Selects features, scales them, and creates sequences."""
    features = df.columns.tolist()
    features.remove('SO2')
    target = 'SO2'

    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()

    df[features] = scaler_features.fit_transform(df[features])
    df[[target]] = scaler_target.fit_transform(df[[target]])

    return df, features, target, scaler_features, scaler_target


# --- 3. Sequence Creation --- (No changes here)

def create_sequences(data, features, target, sequence_length, forecast_horizon):
    """Creates sequences for LSTM training and testing."""
    X, y = [], []
    for i in range(len(data) - sequence_length - forecast_horizon + 1):
        X.append(data[features].iloc[i:(i + sequence_length)].values)
        y.append(data[target].iloc[i + sequence_length : i + sequence_length + forecast_horizon].values)
    return np.array(X), np.array(y)


# --- 4. LSTM Model Building --- (No changes here)

def build_lstm_model(sequence_length, num_features, forecast_horizon, dropout_rate=0.2, lstm_units=64):
    """Builds and compiles the LSTM model."""
    model = Sequential()
    model.add(LSTM(lstm_units, activation='relu', return_sequences=True, input_shape=(sequence_length, num_features)))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(lstm_units, activation='relu', return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(forecast_horizon))

    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mse')
    return model


# --- 5. Training and Evaluation --- (No changes here)

def train_evaluate_model(model, X_train, y_train, X_test, y_test, batch_size, epochs, scaler_target):
    """Trains and evaluates the LSTM model."""
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_test, y_test), callbacks=[early_stopping, model_checkpoint],
                        verbose=1)

    best_model = tf.keras.models.load_model('best_model.h5')
    y_pred = best_model.predict(X_test)
    y_pred_inv = scaler_target.inverse_transform(y_pred)
    y_test_inv = scaler_target.inverse_transform(y_test)

    mse = mean_squared_error(y_test_inv.flatten(), y_pred_inv.flatten())
    mae = mean_absolute_error(y_test_inv.flatten(), y_pred_inv.flatten())
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_inv.flatten(), y_pred_inv.flatten())

    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"R2: {r2}")

    return history, y_pred_inv, y_test_inv, best_model


# --- 6. EDA, Visualization, and Reporting --- (No changes here)

def perform_eda(df):
    """Performs Exploratory Data Analysis (EDA)"""
    print("Data Summary:")
    print(df.describe())

    print("\nMissing Values:")
    print(df.isnull().sum())

    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.show()

    plt.figure(figsize=(14, 6))
    df['SO2'].plot()
    plt.title('Time Series of SO2')
    plt.xlabel('Datetime')
    plt.ylabel('SO2')
    plt.show()


def plot_predictions(y_test_inv, y_pred_inv, num_samples_to_plot=100):
    """Plots predicted vs actual SO2 values."""
    plt.figure(figsize=(14, 6))
    y_test_flat = y_test_inv.flatten()
    y_pred_flat = y_pred_inv.flatten()
    plt.plot(y_test_flat[:num_samples_to_plot], label='Actual SO2', color='blue')
    plt.plot(y_pred_flat[:num_samples_to_plot], label='Predicted SO2', color='red')
    plt.title('Actual vs Predicted SO2 (First 100 Time Steps)')
    plt.xlabel('Time Step')
    plt.ylabel('SO2')
    plt.legend()
    plt.show()


def prediction_example(model, X_test, scaler_target, y_test_inv, num_examples=5):
    """Generates and prints prediction examples."""
    print("\nPrediction Examples:")
    for i in range(num_examples):
        example_index = np.random.randint(0, len(X_test))
        predicted_values = model.predict(X_test[example_index:example_index+1], verbose=0)
        predicted_inv = scaler_target.inverse_transform(predicted_values)
        actual_inv = y_test_inv[example_index]

        print(f"\nExample {i+1}:")
        print("  Predicted SO2 (next 24 hours):", predicted_inv[0])
        print("  Actual SO2    (next 24 hours):", actual_inv)


# --- Main Script ---

if __name__ == "__main__":
    file_path = r'/mnt/e/MayThesis2025/cleanned_datasets/(37t)สถานีอุตุนิยมวิทยาลำปาง.csv'
    sequence_length = 72
    forecast_horizon = 24
    batch_size = 256
    epochs = 100
    test_size_proportion = 0.2

    df = load_and_preprocess(file_path)
    perform_eda(df)  # EDA after preprocessing
    df_scaled, features, target, scaler_features, scaler_target = feature_selection_and_scaling(df)
    X, y = create_sequences(df_scaled, features, target, sequence_length, forecast_horizon)
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_proportion, shuffle=False)
    model = build_lstm_model(sequence_length, len(features), forecast_horizon)
    print(model.summary())
    history, y_pred_inv, y_test_inv, best_model = train_evaluate_model(model, X_train, y_train, X_test, y_test, batch_size, epochs, scaler_target)

    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plot_predictions(y_test_inv, y_pred_inv)
    prediction_example(best_model, X_test, scaler_target, y_test_inv)