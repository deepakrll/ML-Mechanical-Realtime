import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic data
np.random.seed(42)
operating_hours = np.random.randint(100, 5000, 100)
maintenance_cost = 5000 + 3.5 * operating_hours + np.random.normal(0, 5000, 100)

# Create DataFrame
df = pd.DataFrame({'Operating Hours': operating_hours, 'Maintenance Cost': maintenance_cost})

# Split data
X = df[['Operating Hours']]
y = df['Maintenance Cost']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Plot results
plt.figure(figsize=(8,5))
plt.scatter(X_test, y_test, color='blue', label="Actual Data")
plt.plot(X_test, y_pred, color='red', linewidth=2, label="Regression Line")
plt.xlabel("Operating Hours")
plt.ylabel("Maintenance Cost (in Rs.)")
plt.title(f"Tata Motors - Maintenance Cost Prediction\nR² Value: {r2:.2f}")
plt.legend()
plt.grid()
plt.show()

