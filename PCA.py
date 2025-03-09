import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Simulated dataset for Bajaj Auto's quality testing
np.random.seed(42)
data_size = 200

# Generating realistic data
chassis_vibration = np.random.normal(10, 2, data_size)  # mm/s²
engine_temp = np.random.normal(90, 15, data_size)  # °C
brake_efficiency = np.random.normal(80, 5, data_size)  # %
fuel_efficiency = np.random.normal(45, 5, data_size)  # km/l
production_time = np.random.normal(30, 5, data_size)  # minutes per unit

# Creating DataFrame
df = pd.DataFrame({
    'Chassis Vibration (mm/s²)': chassis_vibration,
    'Engine Temperature (°C)': engine_temp,
    'Brake Efficiency (%)': brake_efficiency,
    'Fuel Efficiency (km/l)': fuel_efficiency,
    'Production Time (min)': production_time
})

# Standardizing the data (PCA works better on scaled data)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Applying PCA
pca = PCA(n_components=2)  # Reduce to 2 principal components
principal_components = pca.fit_transform(scaled_data)

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_

# Creating a DataFrame for PCA results
pca_df = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])

# Visualizing PCA results
plt.figure(figsize=(10, 6))
sns.scatterplot(x=pca_df['PC1'], y=pca_df['PC2'], alpha=0.8, color='b')
plt.xlabel(f'Principal Component 1 ({explained_variance[0]*100:.2f}% variance)')
plt.ylabel(f'Principal Component 2 ({explained_variance[1]*100:.2f}% variance)')
plt.title(f'PCA on Bajaj Auto Motorcycle Quality Data\n(Top 2 Components Explain {sum(explained_variance)*100:.2f}% Variance)')
plt.axhline(0, color='grey', linestyle='--', alpha=0.5)
plt.axvline(0, color='grey', linestyle='--', alpha=0.5)
plt.show()

# Display explained variance
print(f"Explained Variance by PC1: {explained_variance[0]*100:.2f}%")
print(f"Explained Variance by PC2: {explained_variance[1]*100:.2f}%")
