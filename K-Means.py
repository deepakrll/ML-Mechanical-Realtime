import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D

# Simulated dataset for CNC Machine Performance at Tata Motors
np.random.seed(42)
data_size = 200
spindle_speed = np.random.randint(1000, 5000, data_size)  # RPM
tool_wear = np.random.randint(5, 100, data_size)  # Tool Wear Percentage
power_consumption = np.random.randint(10, 50, data_size)  # Power Consumption in kW

# Create DataFrame
df = pd.DataFrame({
    'Spindle Speed (RPM)': spindle_speed,
    'Tool Wear (%)': tool_wear,
    'Power Consumption (kW)': power_consumption
})

# Finding the optimal number of clusters using the Elbow Method
wcss = []
K_range = range(2, 10)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df)
    wcss.append(kmeans.inertia_)

# Plotting Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(K_range, wcss, marker='o', linestyle='--', color='b')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Within Cluster Sum of Squares)')
plt.title('Elbow Method to Determine Optimal K for Tata Motors CNC Machines')
plt.show()

# Apply K-Means with the best K (from Elbow Method, let's assume K=3)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(df)

# Calculate Silhouette Score
sil_score = silhouette_score(df[['Spindle Speed (RPM)', 'Tool Wear (%)', 'Power Consumption (kW)']], df['Cluster'])

# 3D Visualization of Clusters
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df['Spindle Speed (RPM)'], df['Tool Wear (%)'], df['Power Consumption (kW)'], 
                      c=df['Cluster'], cmap='viridis', s=50, alpha=0.8)
ax.set_xlabel('Spindle Speed (RPM)')
ax.set_ylabel('Tool Wear (%)')
ax.set_zlabel('Power Consumption (kW)')
ax.set_title(f'K-Means Clustering for Tata Motors CNC Machines\n(Silhouette Score: {sil_score:.2f})')
plt.legend(*scatter.legend_elements(), title='Clusters')
plt.show()

# Display cluster centers
print("Cluster Centers:\n", pd.DataFrame(kmeans.cluster_centers_, columns=df.columns[:-1]))
