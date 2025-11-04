# ================================================
# K-MEANS CLUSTERING ON ONLINE RETAIL DATASET
# ================================================

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------
# Step 1: Load dataset
# ------------------------------------------------
# Download from: https://archive.ics.uci.edu/ml/datasets/online+retail
df = pd.read_excel("Online Retail.xlsx")
print("Original shape:", df.shape)

# ------------------------------------------------
# Step 2: Data cleaning
# ------------------------------------------------
# Remove missing Customer IDs
df.dropna(subset=['CustomerID'], inplace=True)

# Remove canceled or invalid transactions
df = df[df['Quantity'] > 0]

# Compute total price
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# ------------------------------------------------
# Step 3: Feature engineering (RFM)
# ------------------------------------------------
# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])


# Reference date for recency calculation
ref_date = df['InvoiceDate'].max()

# Compute Recency, Frequency, and Monetary values
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (ref_date - x.max()).days,
    'InvoiceNo': 'count',
    'TotalPrice': 'sum'
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
print("RFM table created. Shape:", rfm.shape)
print(rfm.describe())


# ------------------------------------------------
# Step 4: Scaling
# ------------------------------------------------
X = rfm[['Recency', 'Frequency', 'Monetary']]

sns.pairplot(X)
plt.suptitle('RFM Relationships', y=1.02)
plt.savefig('pair-plot')

plt.figure(figsize=(5,4))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.savefig('heatmap',dpi=300.0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------
# Step 5: Elbow Method to find optimal k
# ------------------------------------------------
inertia = []
K = range(1, 11)

for k in K:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X_scaled)
    inertia.append(model.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K, inertia, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.savefig('model-inertia')

# ------------------------------------------------
# Step 6: Apply K-Means
# ------------------------------------------------
k = 4  # Choose optimal k based on elbow curve
kmeans = KMeans(n_clusters=k, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(X_scaled)

# ------------------------------------------------
# Step 7: Analyze clusters
# ------------------------------------------------
cluster_summary = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
print("\nCluster Summary:")
print(cluster_summary)

# ------------------------------------------------
# Step 8: PCA Visualization
# ------------------------------------------------
pca = PCA(n_components=2)
reduced = pca.fit_transform(X_scaled)

plt.figure(figsize=(7,5))
plt.scatter(reduced[:,0], reduced[:,1], c=rfm['Cluster'], cmap='viridis', alpha=0.7)
plt.title('Customer Segments (K-Means)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.savefig('clustering')

# ------------------------------------------------
# Step 9: Save clustered data
# ------------------------------------------------
rfm.to_csv("rfm_clusters.csv", index=False)
print("\nClustered data saved to 'rfm_clusters.csv'")
