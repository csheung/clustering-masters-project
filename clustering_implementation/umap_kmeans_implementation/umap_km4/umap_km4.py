import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap

from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import time
start = time.time()


# Read csv to import preprocessed DataFrame
df = pd.read_csv("../../../remove_all_region_has_disposition_clustering_df.csv")
df_with_keys = df.copy()

df.drop(columns='PcrKey', inplace=True)
df.info()
print(df.shape)
print()

# Rename VentilatorCare/Adjustment column for convenient file storage
df.rename(columns = {'VentilatorCare/Adjustment':'VentilatorCareOrAdjustment'}, inplace=True)
print("Renamed columns to VentilatorCareOrAdjustment\n")


'''
    UMAP Dimensionality Reduction
'''
# Fit UMAP
reducer = umap.UMAP(n_neighbors=50, min_dist=0.0, n_components=2, random_state=42)
embedding = reducer.fit_transform(df)  # Reduce the dimensions to 2D


'''
    K-Means Clustering on Reduced Data
'''
# Perform K-Means clustering on reduced data
kmeans = KMeans(n_init=10, n_clusters=3, random_state=42)
kmeans.fit(embedding)


'''
    Evaluation Metrics
'''
# Clustering Evaluation - Silhouette score
sl_score = silhouette_score(df, kmeans.labels_)
print("Silhouette_score: ", sl_score)
print()


'''
    Initial UMAP Result Visualization
'''
# Define a color palette for clusters
cluster_colors = {0: 'blue', 1: 'red', 2: 'green'} 

# Apply the color mapping
kmeans_colors = [cluster_colors[label] for label in kmeans.labels_]


# Plot and save the results of UMAP + KMeans
plt.figure(figsize=(12, 10))
plt.scatter(embedding[:, 0], embedding[:, 1], c=kmeans_colors, alpha=0.3, edgecolors='none')
plt.title('UMAP + KMeans Clustering Visualization', fontsize=25)
plt.xlabel('UMAP1', fontsize=15)
plt.ylabel('UMAP2', fontsize=15)
plt.savefig('umap_kmeans_result.png')


'''
    Result Visualization
'''
# PCA - 2D
try:
    # Clustering Visualization - PCA
    pca = PCA(n_components=2, random_state=42)
    df_pca = pca.fit_transform(df)

    # Plot and save the results
    plt.figure(figsize=(12, 10))
    plt.scatter(df_pca[:, 0], df_pca[:, 1], c=kmeans_colors, alpha=0.3, edgecolors='none')
    plt.title('PCA 2D Clustering Visualization (UMAP + K-Means)', fontsize=25)
    plt.savefig('kmeans_pca_result.png')
    
except Exception as e:
    print("Error in K-Means 2D PCA: ", e)

print()


# t-SNE - 2D
try:
    # Clustering Visualization - t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
    df_tsne = tsne.fit_transform(df)

    # Plot and save the results
    plt.figure(figsize=(12, 10))
    plt.scatter(df_tsne[:, 0], df_tsne[:, 1], c=kmeans_colors, alpha=0.3, edgecolors='none')
    plt.title('t-SNE 2D Clustering Visualization (UMAP + K-Means)', fontsize=25)
    plt.savefig('kmeans_tsne_result.png')
    
except Exception as e:
    print("Error for t-SNE on K-Means result: ", e)

print()


'''
    3D PCA Visualization for K-Means Clustering
'''
# PCA - 3D
try:
    # Perform PCA with 3 components
    pca_3d = PCA(n_components=3, random_state=42)
    df_pca_3d = pca_3d.fit_transform(df)

    # Define a set of angles (azimuth, elevation)
    angles = [(45, 30), (45, 60), (135, 30), (135, 60), (225, 30), (225, 60), (315, 30), (315, 60)]

    # Generate scatter plots for each angle
    for azim, elev in angles:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(azim=azim, elev=elev)

        # Scatter plot for each cluster
        for cluster in range(kmeans.n_clusters):
            cluster_indices = np.where(kmeans.labels_ == cluster)
            ax.scatter(df_pca_3d[cluster_indices, 0], df_pca_3d[cluster_indices, 1], df_pca_3d[cluster_indices, 2], 
                       alpha=0.5, edgecolors='none', s=30, label=f'Cluster {cluster}', color=cluster_colors[cluster])

        ax.set_title(f'3D PCA Clustering Visualization (Azimuth: {azim}, Elevation: {elev})', fontsize=25)
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.set_zlabel('PCA Component 3')
        ax.legend()
        plt.savefig(f'kmeans_3d_pca_view_{azim}_{elev}.png')

except Exception as e:
    print("Error in 3D PCA visualization: ", e)

print()


'''
    Cluster Interpretation
'''
# Add the cluster number to the original scaled data
df_clustered = df_with_keys.copy()
df_clustered["cluster"] = kmeans.labels_

# Save the df with clustered labels
try:
    df_only_kmeans_cluster = df_clustered[["PcrKey", "cluster"]]
    df_only_kmeans_cluster.to_csv("kmeans_only_cluster_col.csv", index=False)
    df_clustered.to_csv("kmeans_clustered_df.csv", index=False)
except Exception as e:
    print("Error occurred while saving the labeled DataFrame: ", e)

print()


# Statistical Summaries
try:
    for cluster in set(df_clustered['cluster']):
        cluster_stats = df_clustered[df_clustered['cluster'] == cluster].drop('cluster', axis=1).describe()
        cluster_stats.to_csv(f'stats_kmeans_cluster_{cluster}.csv')

except Exception as e:
    print("Error occurred while extracting Statistical Summaries of K-Means Clusters: ", e)

print()



end = time.time()
print(f"*** Total minutes used: {(end - start) / 60} ***")