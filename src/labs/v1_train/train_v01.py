import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import shap
import matplotlib.pyplot as plt

# Load and Preprocess Data
def load_data(file_path):
    df = pd.read_csv(file_path, encoding="utf-8")
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    df.dropna(how='all', inplace=True)
    df.interpolate(method='linear', inplace=True)
    df.fillna(df.mean(), inplace=True)
    return df

# Feature Selection using SHAP
def select_features_shap(df, target_col='SO2', top_k=5):
    df = df.dropna(subset=[target_col])
    X, y = df.drop(columns=[target_col]), df[target_col]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    feature_importance = np.abs(shap_values.values).mean(axis=0)
    return pd.DataFrame({'feature': X.columns, 'importance': feature_importance})\
            .sort_values(by='importance', ascending=False)\
            .head(top_k)['feature'].tolist()

# Train LSTM Model & Predict
def train_and_predict(df, target_col='SO2', top_k=5, look_back=24):
    df = df.dropna(subset=[target_col])
    top_features = select_features_shap(df, target_col, top_k)
    X, y = df[top_features], df[target_col]
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))
    
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - look_back):
        X_seq.append(X_scaled[i:i + look_back])
        y_seq.append(y_scaled[i + look_back])
    
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
    
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(look_back, top_k)),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=1, validation_data=(X_test, y_test))
    
    future_predictions = []
    last_values = X_scaled[-look_back:].reshape(1, look_back, top_k)
    for _ in range(24):
        pred = model.predict(last_values)[0][0]
        future_predictions.append(scaler.inverse_transform([[pred]])[0][0])
        new_input = np.roll(last_values, shift=-1, axis=1)
        new_input[0, -1, :] = pred
        last_values = new_input
    
    # Plot Time-Series Prediction
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 25), future_predictions, marker='o', linestyle='dashed', label='Predicted SO2')
    plt.xlabel('Hours Ahead')
    plt.ylabel('SO2 Level')
    plt.title('SO2 Forecast for Next 24 Hours')
    plt.legend()
    plt.show()
    
    return future_predictions

# Main Execution
file_path = r"E:\MayThesis2025\cleanned_datasets\(37t)ศาลหลักเมือง(ปิดสถานี).csv"
df = load_data(file_path)
predictions = train_and_predict(df, top_k=5)
