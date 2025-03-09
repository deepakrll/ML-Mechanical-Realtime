# Step 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Create a synthetic mechanical dataset
np.random.seed(42)

num_samples = 100
force = np.random.uniform(100, 1000, num_samples)  # Force in Newtons
velocity = np.random.uniform(0.5, 10, num_samples)  # Velocity in m/s
temperature = np.random.uniform(20, 150, num_samples)  # Temperature in Celsius
material_strength = 500 + 0.8 * force - 0.5 * velocity + 0.3 * temperature + np.random.normal(0, 10, num_samples)

# Creating DataFrame
data = pd.DataFrame({
    'Force (N)': force,
    'Velocity (m/s)': velocity,
    'Temperature (°C)': temperature,
    'Material Strength (MPa)': material_strength
})

# Step 3: Save dataset to Google Drive
file_path = "/content/drive/My Drive/mechanical_dataset.csv"
data.to_csv(file_path, index=False)
print(f"Dataset saved at {file_path}")

# Step 4: Load dataset and apply linear regression
df = pd.read_csv(file_path)

X = df[['Force (N)', 'Velocity (m/s)', 'Temperature (°C)']]
y = df['Material Strength (MPa)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)
print("Mean Squared Error:", mse)
print("R² Score:", r2)

# ------------------ Step 5: Visualization ------------------

# 1. Pairplot to show relationships between variables
sns.pairplot(df, diag_kind='kde', plot_kws={'alpha':0.7})
plt.suptitle("Pairplot of Mechanical Dataset", y=1.02)
plt.show()

# 2. Heatmap for correlation between variables
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# 3. 3D Scatter Plot
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['Force (N)'], df['Velocity (m/s)'], df['Material Strength (MPa)'], c=df['Temperature (°C)'], cmap='coolwarm')
ax.set_xlabel("Force (N)")
ax.set_ylabel("Velocity (m/s)")
ax.set_zlabel("Material Strength (MPa)")
plt.title("3D Scatter Plot of Mechanical Properties")
plt.show()

# 4. Regression Line Plot
plt.figure(figsize=(8,6))
sns.regplot(x=y_test, y=y_pred, scatter_kws={"color": "blue"}, line_kws={"color": "red"})
plt.xlabel("Actual Material Strength (MPa)")
plt.ylabel("Predicted Material Strength (MPa)")
plt.title("Actual vs Predicted Material Strength")
plt.show()
