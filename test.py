import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. LOAD DATA & PREPROCESSING
# ==========================================
print("--- Loading Dataset ---")
# This looks for the file 'housing_data.csv' in the same folder
df = pd.read_csv('housing_data.csv')

# Handling Missing Values
df = df.dropna()

# We separate the data: 'Price' is what we want to predict (Target)
# Everything else is a feature (Square footage, bedrooms, etc.)
X = df.drop(columns=['Price']) 
y = df['Price']

# Select only numerical columns for the heatmap visualization
numerical_df = df.select_dtypes(include=[np.number])

# ==========================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
print("\n--- Generating Visualizations ---")
# 1. Feature Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png') 
plt.close()

# 2. Scatter Plot (Takes the first available feature vs Price)
main_feature = X.columns[0] 
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df[main_feature], y=y)
plt.title(f'House Price vs {main_feature}')
plt.xlabel(main_feature)
plt.ylabel('Price')
plt.tight_layout()
plt.savefig('price_scatter_plot.png')
plt.close()

print("Visualizations saved successfully as images!")

# ==========================================
# 3. TRAIN-TEST SPLIT & FEATURE SCALING
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 4. LINEAR REGRESSION MODELING
# ==========================================
print("\n--- Training Linear Regression Model ---")
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# ==========================================
# 5. PERFORMANCE EVALUATION METRICS
# ==========================================
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n================ MODEL METRICS ================")
print(f"Mean Squared Error (MSE) : {mse:.4f}")
print(f"R-squared Score (R2)     : {r2:.4f}")
print("===============================================")
