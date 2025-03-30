import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import shap
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import yaml  # Install PyYAML: pip install pyyaml
import keras_tuner as kt


# Load Configuration
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Load and Preprocess Data
def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df.set_index("Datetime", inplace=True)
        df.dropna(how="all", inplace=True)
        df.interpolate(method="linear", inplace=True)
        df.fillna(df.mean(), inplace=True)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        raise


# Feature Selection using SHAP
def select_features_shap(df: pd.DataFrame, target_col: str, top_k: int) -> list[str]:
    from sklearn.ensemble import RandomForestRegressor

    df = df.dropna(subset=[target_col])
    X, y = df.drop(columns=[target_col]), df[target_col]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    feature_importance = np.abs(shap_values.values).mean(axis=0)
    return (
        pd.DataFrame({"feature": X.columns, "importance": feature_importance})
        .sort_values(by="importance", ascending=False)
        .head(top_k)["feature"]
        .tolist()
    )


def build_model(hp, look_back, top_k):
    model = Sequential(
        [
            LSTM(
                hp.Int("units_1", min_value=32, max_value=128, step=32),
                activation="relu",
                return_sequences=True,
                input_shape=(look_back, top_k),
            ),
            Dropout(hp.Float("dropout_1", min_value=0.0, max_value=0.5, step=0.1)),
            LSTM(
                hp.Int("units_2", min_value=32, max_value=128, step=32),
                activation="relu",
            ),
            Dropout(hp.Float("dropout_2", min_value=0.0, max_value=0.5, step=0.1)),
            Dense(1),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        ),
        loss="mse",
    )
    return model


# Train LSTM Model & Predict
def train_and_predict(
    df: pd.DataFrame,
    target_col: str,
    top_k: int,
    look_back: int,
    epochs: int,
    batch_size: int,
    test_size: float,
) -> tuple[list[float], keras.Model]:
    df = df.dropna(subset=[target_col])
    top_features = select_features_shap(df, target_col, top_k)
    X, y = df[top_features], df[target_col]

    # Separate Scalers for X and y
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1))

    # Create Sequences
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - look_back):
        X_seq.append(X_scaled[i : i + look_back])
        y_seq.append(y_scaled[i + look_back])
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)

    # Time Series Split (Sequential)
    train_size = int(len(X_seq) * (1 - test_size))
    X_train, X_test = X_seq[:train_size], X_seq[train_size:]
    y_train, y_test = y_seq[:train_size], y_seq[train_size:]

    # --- Hyperparameter Tuning (Keras Tuner) ---
    log_dir = "logs/fit/"  # Directory for TensorBoard logs

    # Make sure the log directory exists.
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    tensorboard_callback = TensorBoard(
        log_dir=log_dir, histogram_freq=1, profile_batch=0
    )  # Add TensorBoard callback

    tuner = kt.Hyperband(
        lambda hp: build_model(hp, look_back, top_k),  # Pass look_back and top_k
        objective="val_loss",
        max_epochs=10,  # Reduced for demonstration
        factor=3,
        directory="my_dir",  # Change if needed
        project_name="so2_tuning",
    )

    early_stop = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )  # Increased patience

    tuner.search(
        X_train,
        y_train,
        epochs=epochs,  # Use configured epochs
        batch_size=batch_size,  # Use configured batch size
        validation_data=(X_test, y_test),
        callbacks=[early_stop, tensorboard_callback],
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)
    # --- End Hyperparameter Tuning ---

    # Refit model
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, tensorboard_callback],
        verbose=1,
    )

    # Predictions
    future_predictions = []
    last_values = X_scaled[-look_back:].reshape(1, look_back, top_k)
    for _ in range(24):
        pred = model.predict(last_values, verbose=0)[0][0]
        future_predictions.append(
            y_scaler.inverse_transform([[pred]])[0][0]
        )  # y_scaler
        new_input = np.roll(last_values, shift=-1, axis=1)
        new_input[0, -1, :] = pred  # Use scaled prediction
        last_values = new_input

    # Evaluation
    y_test_unscaled = y_scaler.inverse_transform(y_test)
    y_pred_test = model.predict(X_test, verbose=0)  # added , verbose=0
    y_pred_test_unscaled = y_scaler.inverse_transform(y_pred_test)

    mae = mean_absolute_error(y_test_unscaled, y_pred_test_unscaled)
    rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred_test_unscaled))
    r2 = r2_score(y_test_unscaled, y_pred_test_unscaled)

    print(f"Test MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R-squared: {r2:.4f}")

    # Plot Time-Series Prediction (Test Set)
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_unscaled[:, 0], label="Actual SO2 (Test Set)", color="blue")
    plt.plot(
        y_pred_test_unscaled[:, 0],
        label="Predicted SO2 (Test Set)",
        color="red",
        linestyle="--",
    )
    plt.xlabel("Time Steps (Test Set)")
    plt.ylabel("SO2 Level")
    plt.title("SO2 Prediction on Test Set")
    plt.legend()
    plt.show()

    # Plot Time-Series Prediction (Future)
    plt.figure(figsize=(10, 5))
    plt.plot(
        range(1, 25),
        future_predictions,
        marker="o",
        linestyle="dashed",
        label="Predicted SO2 (Future)",
        color="green",
    )
    plt.xlabel("Hours Ahead")
    plt.ylabel("SO2 Level")
    plt.title("SO2 Forecast for Next 24 Hours")
    plt.legend()
    plt.show()

    return future_predictions, model


# Main Execution
if __name__ == "__main__":
    config = load_config()
    file_path = os.path.join(config["data_dir"], config["filename"])

    df = load_data(file_path)
    predictions, model = train_and_predict(
        df,
        config["target_col"],
        config["top_k_features"],
        config["look_back"],
        config["epochs"],
        config["batch_size"],
        config["test_size"],
    )
    print(f"24-hour SO2 Predictions: {predictions}")
    model.summary()  # Print model summary
