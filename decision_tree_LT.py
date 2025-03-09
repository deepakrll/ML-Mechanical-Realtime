import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score

# Set random seed for reproducibility
np.random.seed(42)

# Simulated dataset for CNC Machine Maintenance
data_size = 200
vibration_level = np.random.randint(1, 100, data_size)  # Vibration severity (1-100 scale)
temperature = np.random.randint(30, 120, data_size)     # Machine Temperature (°C)
operational_hours = np.random.randint(50, 500, data_size)  # Hours since last maintenance

# Maintenance labels (0: No Maintenance, 1: Routine Maintenance, 2: Urgent Maintenance)
labels = np.select(
    [
        (vibration_level < 40) & (temperature < 70) & (operational_hours < 200),  # No Maintenance
        ((vibration_level >= 40) & (vibration_level < 70) & 
         (temperature >= 70) & (temperature < 100)) | 
        ((operational_hours >= 200) & (operational_hours < 350)),  # Routine Maintenance
        (vibration_level >= 70) | (temperature >= 100) | (operational_hours >= 350)  # Urgent Maintenance
    ],
    [0, 1, 2]
)

# Create DataFrame
df = pd.DataFrame({
    'Vibration Level': vibration_level,
    'Temperature (°C)': temperature,
    'Operational Hours': operational_hours,
    'Maintenance Status': labels
})

# Splitting data into training and testing sets
X = df[['Vibration Level', 'Temperature (°C)', 'Operational Hours']]
y = df['Maintenance Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Model
dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = dt_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Visualizing the Decision Tree
plt.figure(figsize=(14, 10))
plot_tree(dt_model, feature_names=X.columns, class_names=['No Maintenance', 'Routine Maintenance', 'Urgent Maintenance'],
          filled=True, rounded=True, fontsize=10)
plt.title(f"L&T - CNC Machine Maintenance Decision Tree (Accuracy: {accuracy:.2f})", fontsize=14)
plt.show()

# Display classification report
print("Classification Report:\n", classification_report(y_test, y_pred))
