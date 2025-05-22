import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Check for GPU availability for TensorFlow
physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    # Configure TensorFlow to use the first available GPU and enable memory growth
    try:
        tf.config.experimental.set_visible_devices(physical_devices[0], "GPU")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU is available and memory growth is enabled.")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
        print("Proceeding with default GPU setup or CPU.")
else:
    print("GPU is not available, TensorFlow will use CPU.")

# --- 1. Data Generation (12 months of hourly data) ---
np.random.seed(42)  # for reproducibility
dates = pd.date_range(start="2024-01-01", end="2024-12-31 23:59:59", freq="h")
n_samples = len(dates)

data = pd.DataFrame(
    {
        "timestamp": dates,
        "no2": np.random.uniform(5, 50, n_samples)
        + 20 * np.sin(np.linspace(0, 12 * 2 * np.pi, n_samples)),  # Adding seasonality
        "co2": np.random.uniform(300, 500, n_samples)
        + 50 * np.sin(np.linspace(0, 2 * 2 * np.pi, n_samples)),  # Shorter seasonality
        "windspd": np.random.uniform(0.5, 15, n_samples),
        "temperature": np.random.uniform(10, 35, n_samples)
        + 5 * np.sin(np.linspace(0, 12 * 2 * np.pi, n_samples)),
        "humidity": np.random.uniform(30, 90, n_samples)
        - 10 * np.sin(np.linspace(0, 12 * 2 * np.pi, n_samples)),
        # SO2 will be somewhat dependent on other features + noise + trend/seasonality
        "so2": (
            0.1 * np.random.uniform(5, 50, n_samples)  # base no2 influence
            + 0.05 * np.random.uniform(300, 500, n_samples)  # base co2 influence
            + -0.5 * np.random.uniform(0.5, 15, n_samples)  # wind dispersion
            + np.random.normal(0, 5, n_samples)  # noise
            + 10
            + 5 * np.sin(np.linspace(0, 4 * 2 * np.pi, n_samples))
        ),  # seasonal trend for so2
    }
)

# Ensure so2 is not negative
data["so2"] = np.maximum(data["so2"], 0.5)
data.set_index("timestamp", inplace=True)

print("--- Sample Generated Data ---")
print(data.head())
print(f"\nGenerated {len(data)} hourly data points.")

# --- 2. Feature Engineering & Preprocessing ---
# Create the target variable: SO2 24 hours ahead
data["so2_target_24h"] = data["so2"].shift(-24)

# Drop rows with NaN values (created by shifting)
data.dropna(inplace=True)

# Select features and target
features = [
    "no2",
    "co2",
    "windspd",
    "temperature",
    "humidity",
    "so2",
]  # current so2 is a feature
target = "so2_target_24h"

X = data[features]
y = data[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)  # Time series data, so no shuffle

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(
    f"\nTraining set shape: X_train_scaled={X_train_scaled.shape}, y_train={y_train.shape}"
)
print(f"Testing set shape: X_test_scaled={X_test_scaled.shape}, y_test={y_test.shape}")


# --- 3. TensorFlow Keras Model (Simple Dense Neural Network) ---
print("\n--- Training TensorFlow Keras Model ---")

tf_model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(
            128, activation="relu", input_shape=(X_train_scaled.shape[1],)
        ),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1),  # Output layer (single value prediction)
    ]
)

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
tf_model.compile(optimizer=optimizer, loss="mse")  # Mean Squared Error for regression

tf_model.summary()

# Train the model
# Using a GPU will be automatic if TensorFlow detects one and it's configured.
history = tf_model.fit(
    X_train_scaled,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1,  # Use part of training data for validation during training
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
    ],  # Early stopping
    verbose=1,
)

tensorflow_pred = tf_model.predict(
    X_test_scaled
).flatten()  # Flatten to make it 1D array like y_test
tensorflow_mse = mean_squared_error(y_test, tensorflow_pred)
print(f"TensorFlow Model MSE: {tensorflow_mse:.4f}")


# --- 4. Visualization ---
print("\n--- Generating Visualizations ---")


# Function to create scatter plot of Actual vs. Predicted
def plot_actual_vs_predicted(actual, predicted, model_name="TensorFlow (Dense NN)"):
    plt.figure(figsize=(8, 8))
    plt.scatter(actual, predicted, alpha=0.5, label="Data points")
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        "k--",
        lw=2,
        label="Ideal Prediction Line",
    )  # Diagonal line
    plt.xlabel("Actual $SO_2$ (24h ahead)")
    plt.ylabel(f"Predicted $SO_2$ (24h ahead) - {model_name}")
    plt.title(f"Actual vs. Predicted $SO_2$ - {model_name}")
    plt.legend()
    plt.grid(True)
    plt.show()


# Function to create residual plot
def plot_residuals(actual, predicted, model_name="TensorFlow (Dense NN)"):
    residuals = actual - predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(predicted, residuals, alpha=0.5)
    plt.hlines(
        0,
        predicted.min(),
        predicted.max(),
        colors="r",
        linestyles="--",
        label="Zero Residual Line",
    )
    plt.xlabel(f"Predicted $SO_2$ - {model_name}")
    plt.ylabel("Residuals ($Actual - Predicted$)")
    plt.title(f"Residual Plot - {model_name}")
    plt.legend()
    plt.grid(True)
    plt.show()


# Function to create line graph of Actual vs. Predicted over time
def plot_time_series(actual_index, actual_values, predicted_values_tf, num_points=500):
    plt.figure(figsize=(15, 7))
    # Ensure num_points does not exceed available data
    num_points = min(num_points, len(actual_index))

    plt.plot(
        actual_index[:num_points],
        actual_values[:num_points],
        label="Actual $SO_2$ (24h ahead)",
        color="blue",
        marker=".",
        linestyle="-",
    )
    plt.plot(
        actual_index[:num_points],
        predicted_values_tf[:num_points],
        label="TensorFlow Predicted $SO_2$",
        color="red",
        linestyle="--",
        marker="x",
    )
    plt.xlabel("Time")
    plt.ylabel("$SO_2$ Value ($\mu g/m^3$)")  # Added units for clarity
    plt.title(
        f"Actual vs. TensorFlow Predicted $SO_2$ (First {num_points} Test Points)"
    )
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Scatter plot
plot_actual_vs_predicted(y_test, tensorflow_pred)

# Residual plot
plot_residuals(y_test, tensorflow_pred)

# Line graph
plot_time_series(y_test.index, y_test.values, tensorflow_pred)

print("\n--- Analysis Complete ---")
print(f"TensorFlow Model MSE: {tensorflow_mse:.4f}")

# Plot training history for TensorFlow model
plt.figure(figsize=(10, 6))
plt.plot(history.history["loss"], label="Training Loss (MSE)")
if "val_loss" in history.history:
    plt.plot(history.history["val_loss"], label="Validation Loss (MSE)")
plt.title("TensorFlow Model Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid(True)
plt.show()
