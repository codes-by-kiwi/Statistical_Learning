
# from google.colab import drive
# drive.mount('/content/drive')

import pandas as pd

file_path_2 = '/content/drive/My Drive/SL assignment 2/data.US.csv'
df_US_data = pd.read_csv(file_path_2)

df_US_data.info()
df_US_data.describe()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pca = PCA()
pca.fit(df_US_data)

explained_variance = pca.explained_variance_ratio_

plt.figure(figsize=(8, 6))
plt.plot(np.cumsum(explained_variance), marker='o')
plt.title('Plot of Cumulative Proportion of Variance by PCA')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()



feature_names = df_US_data.columns
components_df = pd.DataFrame(pca.components_, columns=feature_names)
print("\nPrincipal Components (with feature names):")
print(components_df)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(df_US_data)

explained_variance = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance)
set_threshold = 0.90
n_components = np.argmax(cumulative_explained_variance >= set_threshold) + 1

plt.figure(figsize=(8, 6))
plt.plot(cumulative_explained_variance, marker='o', label='Cumulative Explained Variance')
plt.axvline(x=n_components, color='r', linestyle='--', label=f'Cut-off point')
plt.title('Plot of Cumulative Proportion of Variance by PCA')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(df_US_data)

explained_variance = pca.explained_variance_ratio_
cum_explained_variance = np.cumsum(explained_variance)
gradient = np.gradient(cum_explained_variance)

plt.figure(figsize=(8, 6))

plt.plot(cum_explained_variance, marker='o', label='Cumulative Explained Variance')

for i in range(1, len(cum_explained_variance)):
    slope = gradient[i]

    x_values = [i - 1, i]
    y_values = [cum_explained_variance[i - 1], cum_explained_variance[i] + slope * (i - (i - 1))]

    plt.plot(x_values, y_values, 'r-', alpha=0.7)

plt.title('Plot of Cumulative Proportion of Variance with Tangential Gradient Lines')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.legend()

plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pca = PCA()
pca.fit(df_US_data)

explained_variance = pca.explained_variance_ratio_
eigenvalues = pca.explained_variance_

kept_indices = np.where(eigenvalues >= 1)[0]

plt.figure(figsize=(8, 6))
plt.plot(np.cumsum(explained_variance), marker='o', label='Cumulative Explained Variance')

plt.scatter(kept_indices, np.cumsum(explained_variance)[kept_indices], color='red', label='Kept Components')
for idx in kept_indices:
    plt.text(idx, np.cumsum(explained_variance)[idx], f'PC{idx+1}', ha='center', fontsize=9, color='red')

plt.title('Cumulative Explained Variance with Kept Components(Kaiser)')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.xticks(np.arange(len(explained_variance)))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

eigenvalues = pca.explained_variance_
print(eigenvalues)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pca = PCA()
pca.fit(df_US_data)

eigenvalues = pca.explained_variance_
random_data = np.random.rand(*df_US_data.shape)

pca_random = PCA()
pca_random.fit(random_data)
eigenvalues_random = pca_random.explained_variance_

kept_indices = np.where(eigenvalues > eigenvalues_random)[0]

explained_variance = pca.explained_variance_ratio_

plt.figure(figsize=(8, 6))
plt.plot(np.cumsum(explained_variance), marker='o', label='Cumulative Explained Variance')
plt.scatter(kept_indices, np.cumsum(explained_variance)[kept_indices], color='red', label='Kept Components')
for idx in kept_indices:
    plt.text(idx, np.cumsum(explained_variance)[idx], f'PC{idx+1}', ha='center', fontsize=9, color='red')

plt.title('Cumulative Explained Variance with Kept Components (Eigenvalue > Random Data)')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.xticks(np.arange(len(explained_variance)))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

eigenvalues = pca.explained_variance_
print("Eigenvalues (explained variance for each principal component):")
print(eigenvalues)

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca_3comp = PCA(n_components=3)
principal_components = pca_3comp.fit_transform(df_US_data)

kmeans1 = KMeans(n_clusters=4)
kmeans1.fit(principal_components)
labels1 = kmeans1.labels_

fig, axs = plt.subplots(1, 3, figsize=(18, 6))
axs[0].scatter(principal_components[:, 0], principal_components[:, 1], c=labels1, cmap='viridis')
axs[0].set_title('PC1 vs PC2')
axs[0].set_xlabel('Principal Component 1')
axs[0].set_ylabel('Principal Component 2')
axs[0].grid(True)

axs[1].scatter(principal_components[:, 0], principal_components[:, 2], c=labels1, cmap='viridis')
axs[1].set_title('PC1 vs PC3')
axs[1].set_xlabel('Principal Component 1')
axs[1].set_ylabel('Principal Component 3')
axs[1].grid(True)

axs[2].scatter(principal_components[:, 1], principal_components[:, 2], c=labels1, cmap='viridis')
axs[2].set_title('PC2 vs PC3')
axs[2].set_xlabel('Principal Component 2')
axs[2].set_ylabel('Principal Component 3')
axs[2].grid(True)

plt.tight_layout()
plt.show()

feature_names = df_US_data.columns
components_df = pd.DataFrame(pca_3comp.components_, columns=feature_names)

print("\nPrincipal Component 1 details:")
print(components_df.iloc[0])

print("\nPrincipal Component 2 details:")
print(components_df.iloc[1])

print("\nPrincipal Component 3 details:")
print(components_df.iloc[2])

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

dist_to_centroid = []
for i in range(1, 20):  # 1 to 20 tried
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(principal_components)
    dist_to_centroid.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 20), dist_to_centroid, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

from sklearn.metrics import silhouette_score
import numpy as np

cluster_Range = range(2, 11)
silhouette_scores = []

for k in cluster_Range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df_US_data)
    score = silhouette_score(df_US_data, kmeans.labels_)
    silhouette_scores.append(score)

plt.plot(cluster_Range, silhouette_scores, marker='o')
plt.title('Silhouette Score for Optimal Number of Clusters')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.show()

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#over here i have just used 3 components
pca_3comp = PCA(n_components=3)
principal_components = pca_3comp.fit_transform(df_US_data)

kmeans2 = KMeans(n_clusters=2)
kmeans2.fit(principal_components)
labels2 = kmeans2.labels_

fig, axs = plt.subplots(1, 3, figsize=(18, 6))
axs[0].scatter(principal_components[:, 0], principal_components[:, 1], c=labels2, cmap='viridis')
axs[0].set_title('PC1 vs PC2')
axs[0].set_xlabel('Principal Component 1')
axs[0].set_ylabel('Principal Component 2')
axs[0].grid(True)

axs[1].scatter(principal_components[:, 0], principal_components[:, 2], c=labels2, cmap='viridis')
axs[1].set_title('PC1 vs PC3')
axs[1].set_xlabel('Principal Component 1')
axs[1].set_ylabel('Principal Component 3')
axs[1].grid(True)

axs[2].scatter(principal_components[:, 1], principal_components[:, 2], c=labels2, cmap='viridis')
axs[2].set_title('PC2 vs PC3')
axs[2].set_xlabel('Principal Component 2')
axs[2].set_ylabel('Principal Component 3')
axs[2].grid(True)

plt.tight_layout()
plt.show()

feature_names = df_US_data.columns
components_df = pd.DataFrame(pca_3comp.components_, columns=feature_names)

print("\nPrincipal Component 1 details:")
print(components_df.iloc[0])

print("\nPrincipal Component 2 details:")
print(components_df.iloc[1])

print("\nPrincipal Component 3 details:")
print(components_df.iloc[2])

cluster_sizes = np.bincount(labels2)
cluster_sizes_df = pd.Series(labels2).value_counts().sort_index()

print("Cluster sizes:")
print(cluster_sizes)

cluster_sizes = np.bincount(labels1)
cluster_sizes_df = pd.Series(labels1).value_counts().sort_index()

print("Cluster sizes:")
print(cluster_sizes)

"""#hierarchial clustering when clusters=2"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering

pca = PCA(n_components=3)
pca_data = pca.fit_transform(df_US_data)

linkage_matrix = linkage(pca_data, method='ward')

plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix, truncate_mode='lastp', p=30, leaf_rotation=90, leaf_font_size=10)
plt.axhline(y=6200, color='r', linestyle='--')
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data Point')
plt.ylabel('Distance')
plt.show()

n_clusters = 2
ag_Clus = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
labels3 = ag_Clus.fit_predict(pca_data)

plt.figure(figsize=(8, 6))
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=labels3, cmap='viridis', s=50)
plt.title(f'Hierarchical Clustering (n={n_clusters}) on PCA-reduced Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.colorbar(label='Cluster Label')
plt.show()

cluster_sizes_3 = np.bincount(labels3)
cluster_sizes_3 = pd.Series(labels3).value_counts().sort_index()

print("Cluster sizes:")
print(cluster_sizes_3)

from sklearn.preprocessing import StandardScaler
personality_columns = df_US_data.columns[1:31]

df_US_data['cluster'] = labels3
df_US_data['cluster'] = labels3 - 1
cluster_groups = df_US_data.groupby('cluster')[personality_columns].mean()

scaled_cluster_groups = pd.DataFrame(
    StandardScaler().fit_transform(cluster_groups.T),
    index=cluster_groups.columns,
    columns=cluster_groups.index
).T

cluster_groups = df_US_data.groupby('cluster').mean()
plt.figure(figsize=(15, 6))
sns.heatmap(scaled_cluster_groups.T, cmap='coolwarm', annot=True)
plt.title("Average Personality Characteristics by Cluster")
plt.xlabel("Cluster/Group")
plt.ylabel("Personality Characteristic")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()