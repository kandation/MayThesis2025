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


#%% 
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
        
    df.to_markdown()

    return df

#%%
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



# %%

if __name__ == "__main__":
    file_path = r'/mnt/e/MayThesis2025/cleanned_datasets/(37t)สถานีอุตุนิยมวิทยาลำปาง.csv'
    sequence_length = 72
    forecast_horizon = 24
    batch_size = 256
    epochs = 100
    test_size_proportion = 0.2

    df = load_and_preprocess(file_path)
    perform_eda(df)  # EDA after preprocessing