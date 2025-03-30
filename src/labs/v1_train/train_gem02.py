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
import yaml
import keras_tuner as kt
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.model_selection import train_test_split


# --- GPU Configuration (Corrected and Improved) ---
def configure_gpu():
    """Configures TensorFlow to use GPUs with memory growth."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Set memory growth for each GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} Physical GPUs.")

            # Use a MirroredStrategy for multi-GPU training (if available)
            strategy = tf.distribute.MirroredStrategy()
            print(f"Using MirroredStrategy with {strategy.num_replicas_in_sync} GPUs.")
            return strategy  # Return the strategy

        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    else:
        print("No GPUs found, using CPU.")
    return None  # Return None if no GPUs or error


# Load Configuration
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Load and Preprocess Data (Optimized)
def load_data(file_path: str, target_col: str) -> pd.DataFrame:
    try:
        # Optimized data loading with specified dtypes and date parsing
        dateparse = lambda x: pd.to_datetime(
            x, format="%Y-%m-%d %H:%M:%S", errors="coerce"
        )  # Example, adjust to your date format
        df = pd.read_csv(
            file_path,
            encoding="utf-8",
            parse_dates=["Datetime"],
            date_parser=dateparse,
            index_col="Datetime",
        )  # Set index during read

        # Drop rows where the target column is NaN *before* any other operations.
        df.dropna(subset=[target_col], inplace=True)

        # Efficient interpolation and filling.
        df.interpolate(
            method="linear", inplace=True
        )  # Linear is often fast and good for time series.
        df.fillna(df.mean(numeric_only=True), inplace=True)  # Only numeric columns.
        return df

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        raise


# Feature Selection using SHAP (with Sampling)
def select_features_shap(
    df: pd.DataFrame, target_col: str, top_k: int, sample_size: int = 1000
) -> list[str]:
    from sklearn.ensemble import RandomForestRegressor

    X, y = df.drop(columns=[target_col]), df[target_col]

    # Stratified Sampling for SHAP analysis
    if len(X) > sample_size:
        try:
            _, X_sample, _, y_sample = train_test_split(
                X, y, test_size=sample_size, random_state=42, stratify=y
            )  # Stratify
        except ValueError:  # if y has only one class in training
            _, X_sample, _, y_sample = train_test_split(
                X, y, test_size=sample_size, random_state=42
            )
    else:
        X_sample, y_sample = X, y  # If dataset smaller than sample, use all.

    model = RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    )  # Use all cores.
    model.fit(X_sample, y_sample)

    explainer = shap.TreeExplainer(
        model
    )  # Use TreeExplainer, much faster for tree models
    shap_values = explainer(X_sample)
    feature_importance = np.abs(shap_values.values).mean(axis=0)
    return (
        pd.DataFrame({"feature": X_sample.columns, "importance": feature_importance})
        .sort_values(by="importance", ascending=False)
        .head(top_k)["feature"]
        .tolist()
    )


# Efficient Sequence Creation using sliding_window_view
def create_sequences(X, y, look_back):
    X_seq = sliding_window_view(X, (look_back, X.shape[1])).squeeze(axis=1)
    y_seq = sliding_window_view(y, (1,)).squeeze()[
        look_back - 1 :
    ]  # THIS WAS THE KEY FIX
    return X_seq, y_seq


def build_model(hp, look_back, top_k):
    # Fix: use 'tanh' or specify recurrent_activation
    model = Sequential(
        [
            LSTM(
                hp.Int("units_1", min_value=32, max_value=128, step=32),
                activation="tanh",  # Use tanh for cuDNN compatibility
                recurrent_activation="sigmoid",  # and specify recurrent_activation
                return_sequences=True,
                input_shape=(look_back, top_k),
            ),
            Dropout(hp.Float("dropout_1", min_value=0.0, max_value=0.5, step=0.1)),
            LSTM(
                hp.Int("units_2", min_value=32, max_value=128, step=32),
                activation="tanh",  # Use tanh for cuDNN compatibility
                recurrent_activation="sigmoid",  # and specify recurrent_activation
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


# Train LSTM Model & Predict (Optimized)
def train_and_predict(
    df: pd.DataFrame,
    target_col: str,
    top_k: int,
    look_back: int,
    epochs: int,
    batch_size: int,
    test_size: float,
    strategy=None,  # Add strategy as an argument
) -> tuple[list[float], keras.Model]:
    top_features = select_features_shap(
        df, target_col, top_k
    )  # Get features *before* scaling.
    X, y = df[top_features], df[target_col]

    # Separate Scalers for X and y
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1))

    # Create Sequences - Highly Optimized, and fixed y_seq
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, look_back)

    # Time Series Split
    train_size = int(len(X_seq) * (1 - test_size))
    X_train, X_test = X_seq[:train_size], X_seq[train_size:]
    y_train, y_test = y_seq[:train_size], y_seq[train_size:]

    # --- Hyperparameter Tuning with Strategy ---
    if strategy:
        with strategy.scope():  # Use the strategy's scope
            tuner = kt.Hyperband(
                lambda hp: build_model(hp, look_back, top_k),
                objective="val_loss",
                max_epochs=5,
                factor=3,
                directory="my_dir",
                project_name="so2_tuning",
            )

    else:  # If no strategy (no GPU), run normally.
        tuner = kt.Hyperband(
            lambda hp: build_model(hp, look_back, top_k),
            objective="val_loss",
            max_epochs=5,
            factor=3,
            directory="my_dir",
            project_name="so2_tuning",
        )

    log_dir = "logs/fit/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tensorboard_callback = TensorBoard(
        log_dir=log_dir, histogram_freq=1, profile_batch=0
    )
    early_stop = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    tuner.search(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, tensorboard_callback],
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Build the model within the strategy scope *again* for training
    if strategy:
        with strategy.scope():
            model = tuner.hypermodel.build(best_hps)
            # Refit model - Train with best hyperparameters for full epochs.
            history = model.fit(
                X_train,
                y_train,
                epochs=epochs,  # Train for the full number of epochs.
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                callbacks=[early_stop, tensorboard_callback],
                verbose=1,
            )

    else:
        model = tuner.hypermodel.build(best_hps)
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

    # --- Predictions (Optimized for future predictions) ---

    # Batch prediction for the test set
    y_pred_test = model.predict(X_test, batch_size=batch_size, verbose=0)
    y_pred_test_unscaled = y_scaler.inverse_transform(y_pred_test)
    y_test_unscaled = y_scaler.inverse_transform(y_test)

    # --- Future Predictions (Optimized) ---
    last_values = X_scaled[-look_back:]
    future_predictions = []

    # Reshape last_values for batch prediction
    # We need to predict one step at a time, but we can still use batch_size
    num_predictions = 24
    last_values_batch = np.tile(
        last_values, (batch_size, 1, 1)
    )  # Repeat for batch prediction

    for _ in range(num_predictions):
        # Predict in batches
        preds = model.predict(
            last_values_batch[:1], batch_size=batch_size, verbose=0
        )  # Predict one step
        pred = preds[0, 0]  # Get the first prediction
        future_predictions.append(y_scaler.inverse_transform([[pred]])[0, 0])
        # Update last_values with the *scaled* prediction
        last_values = np.roll(last_values, -1, axis=0)  # Efficient rolling
        last_values[-1, :] = pred  # Scaled prediction

        last_values_batch = np.tile(last_values, (batch_size, 1, 1))

    # --- Evaluation ---
    mae = mean_absolute_error(y_test_unscaled, y_pred_test_unscaled)
    rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred_test_unscaled))
    r2 = r2_score(y_test_unscaled, y_pred_test_unscaled)

    print(f"Test MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R-squared: {r2:.4f}")

    # Plotting (no changes here, just included for completeness)
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

    # Configure GPU *before* any other TensorFlow operations
    strategy = configure_gpu()

    df = load_data(file_path, config["target_col"])
    predictions, model = train_and_predict(
        df,
        config["target_col"],
        config["top_k_features"],
        config["look_back"],
        config["epochs"],
        config["batch_size"],
        config["test_size"],
        strategy,  # Pass the strategy
    )
    print(f"24-hour SO2 Predictions: {predictions}")
    model.summary()
