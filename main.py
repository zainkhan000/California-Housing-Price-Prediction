# ==========================================
# Step 1: Imports & Setup
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# Step 2: Load Dataset
# ==========================================
print("Loading dataset...")
data = fetch_california_housing(as_frame=True)
df = pd.concat([data.data, data.target.rename('MedHouseVal')], axis=1)
display(df.head())

# ==========================================
# Step 3: Exploratory Data Analysis (EDA)
# ==========================================
print("\n--- EDA: Missing Values ---")
print(df.isnull().sum()) # Check for missing values

print("\n--- EDA: Basic Statistics ---")
display(df.describe())

# Plot distributions of the target variable
plt.figure(figsize=(8, 4))
sns.histplot(df['MedHouseVal'], bins=50, kde=True)
plt.title('Distribution of Median House Values')
plt.xlabel('Median House Value (in $100,000s)')
plt.show()

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()

# ==========================================
# Step 4: Train/Test Split
# ==========================================
X = df.drop(columns='MedHouseVal')
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# ==========================================
# Step 5: Train Model
# ==========================================
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# ==========================================
# Step 6: Evaluate Model
# ==========================================
mae = mean_absolute_error(y_test, y_pred)
# Note: squared=False is deprecated in newer scikit-learn versions, using root_mean_squared_error is preferred, but here is the manual workaround for compatibility:
rmse = np.sqrt(mean_squared_error(y_test, y_pred)) 
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R2:   {r2:.3f}")

# ==========================================
# Step 7: Plot Predicted vs Actual & Residuals
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Actual vs Predicted
axes[0].scatter(y_test, y_pred, alpha=0.4)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
axes[0].set_xlabel("Actual Values")
axes[0].set_ylabel("Predicted Values")
axes[0].set_title("Actual vs Predicted")

# Residuals Plot
residuals = y_test - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.4)
axes[1].axhline(y=0, color='red', lw=2)
axes[1].set_xlabel("Predicted Values")
axes[1].set_ylabel("Residuals")
axes[1].set_title("Residuals vs Predicted")

plt.tight_layout()
plt.show()

# ==========================================
# Step 8: Save Model (Optional Deliverable)
# ==========================================
with open('linear_regression_model.pkl', 'wb') as file:
    pickle.dump(model, file)
print("Model saved to 'linear_regression_model.pkl'")
