# %% Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy import stats

# Set style for seaborn
sns.set_style("whitegrid")

# %% Load Dataset
uri = "/mnt/e/MayThesis2025/src/labs4/v02_merge/output/banhuafai.csv"
df = pd.read_csv(uri, parse_dates=["Datetime"])  # Replace with your dataset
# set datetime as index
df.set_index("Datetime", inplace=True)

# Display shape and first few rows
print(f"Shape of dataset: {df.shape}")
print(df.head())

# Check data types
print(df.info())

# %% Missing Values Analysis
print(df.isnull().sum())

# Visualize missing data
msno.matrix(df)
plt.show()

# %% Check Duplicate Data
print(f"Duplicated rows: {df.duplicated().sum()}")

# %% Statistical Summary
# Numeric data summary
print(df.describe())

# Categorical data summary
print(df.describe(include="all"))

# %% Distribution Analysis (Histograms)
numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns
num_cols = len(numerical_cols)

# Calculate the number of rows needed (3 columns per row)
num_rows = int(np.ceil(num_cols / 3))

plt.figure(figsize=(15, 5 * num_rows))  # Adjust figure size based on rows
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(num_rows, 3, i)  # Adjust subplot layout dynamically
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f"Distribution of {col}")

plt.tight_layout()
plt.show()


# %% Outliers Detection (Boxplots)
num_cols = len(numerical_cols)

# Calculate required rows dynamically
num_rows = int(np.ceil(num_cols / 4))

plt.figure(figsize=(15, 5 * num_rows))  # Adjust figure size dynamically
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(num_rows, 4, i)  # Adjust subplot layout
    sns.boxplot(y=df[col])
    plt.title(f"Boxplot of {col}")

plt.tight_layout()
plt.show()

# %% Remove Outliers using Z-score
z_scores = np.abs(stats.zscore(df[numerical_cols]))
df_no_outliers = df[(z_scores < 3).all(axis=1)]
print(f"Shape after removing outliers: {df_no_outliers.shape}")

# %% Correlation Matrix
plt.figure(figsize=(12, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# %% Pairplot for Relationships
sns.pairplot(df, diag_kind="kde")
plt.show()

# %% Weather-Specific Analysis

# Ensure index is a datetime object
if not isinstance(df.index, pd.DatetimeIndex):
    df.index = pd.to_datetime(df.index, errors="coerce")

# Extract month from the index
df["month"] = df.index.month

# Debug: Print first few rows to confirm datetime is indexed
print(df.head())

# Monthly Temperature Trends
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="month", y="temperature_max", label="Max Temperature")
sns.lineplot(data=df, x="month", y="temperature_min", label="Min Temperature")
plt.xlabel("Month")
plt.ylabel("Temperature (°C)")
plt.title("Monthly Temperature Trends")
plt.legend()
plt.show()


# %% Monthly Rainfall Analysis
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x="month", y="precipitation", ci=None)
plt.xlabel("Month")
plt.ylabel("Average Precipitation (mm)")
plt.title("Average Monthly Rainfall")
plt.show()

# %% Humidity vs Temperature
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x="humidity", y="temperature_avg", alpha=0.5)
plt.xlabel("Humidity (%)")
plt.ylabel("Average Temperature (°C)")
plt.title("Humidity vs. Temperature")
plt.show()

# %% Wind Speed Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df["wind_speed"], bins=30, kde=True)
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Frequency")
plt.title("Wind Speed Distribution")
plt.show()

# %% Trend Analysis Over Time
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="date", y="temperature_avg", label="Avg Temperature")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.title("Temperature Trend Over Time")
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="date", y="precipitation", label="Precipitation")
plt.xlabel("Date")
plt.ylabel("Precipitation (mm)")
plt.title("Precipitation Trend Over Time")
plt.legend()
plt.show()
