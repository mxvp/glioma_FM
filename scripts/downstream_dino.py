import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# === Configs ===
EMBEDDING_DIR = "tumor_embeddings_output"
OUT_DIR = "embedding_analysis_output"
os.makedirs(OUT_DIR, exist_ok=True)

# === Load Your Features ===
features = np.load(os.path.join(EMBEDDING_DIR, "tumor_slice_embeddings.npy"))  # shape (N, 256)
with open(os.path.join(EMBEDDING_DIR, "filenames.txt")) as f:
    filenames = [line.strip() for line in f]

# === Visualization ===
def visualize_embeddings(features: np.ndarray, filenames: list[str], method='umap'):
    if method == 'umap':
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric="cosine", random_state=42)
        embeddings_2d = reducer.fit_transform(features)
        title = "UMAP Projection"
        colnames = ["UMAP1", "UMAP2"]
    elif method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
        embeddings_2d = reducer.fit_transform(features)
        title = "t-SNE Projection"
        colnames = ["tSNE1", "tSNE2"]
    else:
        raise ValueError("method must be 'umap' or 'tsne'")

    df = pd.DataFrame(embeddings_2d, columns=colnames)
    df["Slice"] = filenames

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=colnames[0], y=colnames[1], data=df, s=10)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{method}_scatter.png"), dpi=300)
    plt.close()

    return df

# === Clustering ===
def cluster_embeddings(features: np.ndarray, min_k=2, max_k=10):
    scores = {}
    for k in range(min_k, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(features)
        score = silhouette_score(features, cluster_labels)
        scores[k] = score
        print(f"K={k}, Silhouette Score={score:.4f}")

    best_k = max(scores, key=scores.get)
    print(f"\nBest K based on Silhouette: {best_k}")

    final_kmeans = KMeans(n_clusters=best_k, random_state=42)
    final_labels = final_kmeans.fit_predict(features)

    return final_labels, final_kmeans

# === Run UMAP and t-SNE ===
df_umap = visualize_embeddings(features, filenames, method='umap')
df_tsne = visualize_embeddings(features, filenames, method='tsne')

# === Clustering ===
cluster_labels, kmeans_model = cluster_embeddings(features)
df_umap["Cluster"] = cluster_labels

# === Plot Clusters on UMAP ===
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_umap, x="UMAP1", y="UMAP2", hue="Cluster", palette="tab10", s=15)
plt.title("UMAP with KMeans Clusters")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "umap_clusters.png"), dpi=300)
plt.close()

# === Save DataFrames ===
df_umap.to_csv(os.path.join(OUT_DIR, "embedding_umap_with_clusters.csv"), index=False)
df_tsne.to_csv(os.path.join(OUT_DIR, "embedding_tsne.csv"), index=False)
