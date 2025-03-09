import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Create synthetic dataset
np.random.seed(42)
data_size = 200
wear_level = np.random.randint(1, 100, data_size)
corrosion_level = np.random.randint(1, 100, data_size)
labels = np.where(wear_level + corrosion_level > 120, 1, 0)  # 1: Critical defect, 0: Minor defect

# Create DataFrame
df = pd.DataFrame({'Wear Level': wear_level, 'Corrosion Level': corrosion_level, 'Defect Type': labels})

# Split data
X = df[['Wear Level', 'Corrosion Level']]
y = df['Defect Type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Classification Accuracy: {accuracy}")
