import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.metrics import mean_squared_error
import os

# --- Configuration ---
MODEL_FILENAME = "test_model_mse.h5"
N_SAMPLES = 100
N_FEATURES = 5
N_EPOCHS = 3  # Keep low for quick testing
BATCH_SIZE = 32

# --- 1. Create Dummy Data ---
print("1. Creating dummy data...")
X_train = np.random.rand(N_SAMPLES, N_FEATURES)
y_train = np.random.rand(N_SAMPLES, 1)  # Regression target

X_test = np.random.rand(N_SAMPLES // 2, N_FEATURES)
y_test = np.random.rand(N_SAMPLES // 2, 1)

# --- 2. Define and Compile the Original Model ---
print("\n2. Defining and compiling the original model...")
original_model = keras.Sequential(
    [
        layers.Dense(32, activation="relu", input_shape=(N_FEATURES,)),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),  # Output layer for regression
    ]
)

# Compile the model using the string 'mse' for loss
# This is what can sometimes cause issues on loading if Keras can't resolve it.
original_model.compile(
    optimizer="adam", loss="mse", metrics=["mae"]
)  # Added mae for illustration
original_model.summary()

# --- 3. Train the Original Model ---
print("\n3. Training the original model...")
history = original_model.fit(
    X_train,
    y_train,
    epochs=N_EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    verbose=1,  # Set to 0 or 1 for less/more output
)
print("Training complete.")

# --- 4. Save the Model ---
print(f"\n4. Saving the model to {MODEL_FILENAME}...")
original_model.save(MODEL_FILENAME)
print("Model saved.")

# --- Optional: Evaluate original model directly for comparison ---
print("\n--- Evaluating original model directly (for comparison) ---")
loss_orig, mae_orig = original_model.evaluate(X_test, y_test, verbose=0)
y_pred_orig = original_model.predict(X_test)
mse_orig_sklearn = mean_squared_error(y_test, y_pred_orig)
print(
    f"Original Model - Keras evaluate() loss (MSE): {loss_orig:.6f}, MAE: {mae_orig:.6f}"
)
print(f"Original Model - Scikit-learn MSE on predictions: {mse_orig_sklearn:.6f}")


# --- 5. Load the Model - Strategy 1: compile=False ---
print("\n\n--- Strategy 1: Loading model with compile=False ---")
print(
    "This is suitable if you only need to make predictions and don't need the compiled state (loss, optimizer)."
)
try:
    loaded_model_compile_false = keras.models.load_model(MODEL_FILENAME, compile=False)
    print("Model loaded successfully with compile=False.")

    # If you needed to train further or use model.evaluate(), you would recompile:
    # loaded_model_compile_false.compile(optimizer="adam", loss="mse")
    # print("Recompiled model loaded with compile=False.")

    # --- 6. Make Predictions (Strategy 1) ---
    print("\n6.1 Making predictions with model loaded via compile=False...")
    y_pred_compile_false = loaded_model_compile_false.predict(X_test)

    # --- 7. Calculate MSE (Strategy 1) ---
    mse_compile_false = mean_squared_error(y_test, y_pred_compile_false)
    print(f"MSE (compile=False loaded model): {mse_compile_false:.6f}")

except Exception as e:
    print(f"Error loading with compile=False: {e}")


# --- 5. Load the Model - Strategy 2: custom_objects ---
print("\n\n--- Strategy 2: Loading model with custom_objects ---")
print("This is suitable if you need the full compiled state restored.")
# The error message from your original post was:
# "Could not locate function 'mse'. Make sure custom classes are decorated with `@keras.saving.register_keras_serializable()`."
# "Full object config: {'module': 'keras.metrics', 'class_name': 'function', 'config': 'mse', 'registered_name': 'mse'}"
# This indicates Keras saved 'mse' as a reference to a *metric function*.
# Even if you only specify `loss='mse'`, Keras often implicitly tracks the loss as a metric too.
custom_objects_dict = {
    "mse": tf.keras.metrics.mean_squared_error,  # For the metric 'mse'
    # If the 'loss' itself was the issue (less likely given your error):
    # 'mse': tf.keras.losses.mean_squared_error # or tf.keras.losses.MeanSquaredError()
    # If you used custom activation functions or layers, they'd go here too.
}
# Sometimes, Keras also tracks the loss itself. If the above doesn't work or you see
# an error related to the loss, you might need to specify the loss function too,
# even if it's the same string:
# custom_objects_dict = {
#     'mse': tf.keras.losses.MeanSquaredError(), # For the loss
#     # 'mae': tf.keras.metrics.MeanAbsoluteError() # If you also had custom metrics by name
# }
# For your specific error, the metrics mapping is most crucial.

try:
    loaded_model_custom_objects = keras.models.load_model(
        MODEL_FILENAME, custom_objects=custom_objects_dict
    )
    print("Model loaded successfully with custom_objects.")
    loaded_model_custom_objects.summary()  # Should show the compiled loss and metrics

    # --- 6. Make Predictions (Strategy 2) ---
    print("\n6.2 Making predictions with model loaded via custom_objects...")
    y_pred_custom_objects = loaded_model_custom_objects.predict(X_test)

    # --- 7. Calculate MSE (Strategy 2) ---
    mse_custom_objects = mean_squared_error(y_test, y_pred_custom_objects)
    print(f"MSE (custom_objects loaded model, sklearn): {mse_custom_objects:.6f}")

    # You can also use model.evaluate() now
    loss_co, mae_co = loaded_model_custom_objects.evaluate(X_test, y_test, verbose=0)
    print(
        f"Keras evaluate() (custom_objects loaded model) - loss (MSE): {loss_co:.6f}, MAE: {mae_co:.6f}"
    )


except Exception as e:
    print(f"Error loading with custom_objects: {e}")
    print(
        "If this fails, try changing the custom_objects_dict to point to tf.keras.losses.MeanSquaredError() or tf.keras.losses.mean_squared_error for 'mse'."
    )


# --- 8. Clean up ---
if os.path.exists(MODEL_FILENAME):
    os.remove(MODEL_FILENAME)
    print(f"\nCleaned up {MODEL_FILENAME}")

print("\n--- Comparison of MSEs ---")
print(f"Original Model (sklearn MSE):      {mse_orig_sklearn:.6f}")
if "mse_compile_false" in locals():
    print(f"Loaded (compile=False, sklearn MSE): {mse_compile_false:.6f}")
if "mse_custom_objects" in locals():
    print(f"Loaded (custom_obj, sklearn MSE):  {mse_custom_objects:.6f}")
if "loss_co" in locals():
    print(f"Loaded (custom_obj, Keras eval MSE):{loss_co:.6f}")
