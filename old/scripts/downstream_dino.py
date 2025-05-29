import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# === Configs ===
EMBEDDING_DIR = "/oak/stanford/groups/ogevaert/maxvpuyv/projects/brain/tumor_embeddings_output"
OUT_DIR = "embedding_analysis_output"
MUTATION_FILE = "/oak/stanford/groups/ogevaert/maxvpuyv/projects/brain/data/metadata/xena/mc3_gene_level_GBM_mc3_gene_level.txt"  # genes x TCGA sample IDs (12-char)
os.makedirs(OUT_DIR, exist_ok=True)

# === Load slice-level embeddings and filenames ===
features = np.load(os.path.join(EMBEDDING_DIR, "tumor_slice_embeddings.npy"))
with open(os.path.join(EMBEDDING_DIR, "filenames.txt")) as f:
    filenames = [line.strip() for line in f]

# === Extract TCGA Patient IDs ===
def extract_tcga_id(filename):
    match = re.search(r'TCGA(\d{6})', filename)
    if match:
        raw_id = match.group(1)
        return f"TCGA-06-{raw_id[-4:]}"
    return None

patient_ids = [extract_tcga_id(f) for f in filenames]

# === Aggregate to patient-level ===
df = pd.DataFrame(features)
df["PatientID"] = patient_ids
patient_embeddings = df.groupby("PatientID").mean()

# === Load mutation status for IDH1/IDH2 ===
mut = pd.read_csv(MUTATION_FILE, sep="\t", index_col=0)
idh1 = mut.loc["IDH1"]
idh2 = mut.loc["IDH2"]
idh_status = ((idh1 + idh2) > 0).astype(int)
idh_status.index = idh_status.index.str[:12]
idh_status.name = "IDH_mut"

# === Visualization ===
def visualize_embeddings(features: np.ndarray, patient_ids: list[str], idh_status: pd.Series, method='umap'):
    if method == 'umap':
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, metric="cosine", random_state=42)
        emb2d = reducer.fit_transform(features)
        title = "UMAP Projection"
        colnames = ["UMAP1", "UMAP2"]
    elif method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=20, n_iter=1000, random_state=42)
        emb2d = reducer.fit_transform(features)
        title = "t-SNE Projection"
        colnames = ["tSNE1", "tSNE2"]
    else:
        raise ValueError("Invalid method")

    df_plot = pd.DataFrame(emb2d, columns=colnames)
    df_plot["PatientID"] = patient_ids
    df_plot = df_plot.merge(idh_status.reset_index().rename(columns={"index": "PatientID"}), on="PatientID", how="left")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_plot, x=colnames[0], y=colnames[1], hue="IDH_mut", palette="Set1", s=40)
    plt.title(f"{title} Colored by IDH Mutation")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{method}_idh_colored.png"), dpi=300)
    plt.close()

    return df_plot

# === Clustering ===
def cluster_embeddings(features: np.ndarray, min_k=2, max_k=10):
    scores = {}
    for k in range(min_k, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(features)
        score = silhouette_score(features, labels)
        scores[k] = score
        print(f"K={k}, Silhouette Score={score:.4f}")

    best_k = max(scores, key=scores.get)
    final_kmeans = KMeans(n_clusters=best_k, random_state=42)
    final_labels = final_kmeans.fit_predict(features)
    print(f"\n✅ Best K = {best_k}")
    return final_labels

# === Run Projections ===
X = patient_embeddings.values
patient_list = patient_embeddings.index.tolist()

df_umap = visualize_embeddings(X, patient_list, idh_status, method="umap")
df_tsne = visualize_embeddings(X, patient_list, idh_status, method="tsne")

# === Cluster Patients ===
clusters = cluster_embeddings(X)
df_umap["Cluster"] = clusters

# === Plot Clustered UMAP ===
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_umap, x="UMAP1", y="UMAP2", hue="Cluster", palette="tab10", s=40)
plt.title("UMAP with KMeans Clusters (Patient-level)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "umap_clusters_patient_level.png"), dpi=300)
plt.close()

# === Save Results ===
df_umap.to_csv(os.path.join(OUT_DIR, "umap_patient_level.csv"), index=False)
df_tsne.to_csv(os.path.join(OUT_DIR, "tsne_patient_level.csv"), index=False)
np.save(os.path.join(OUT_DIR, "patient_embeddings.npy"), X)
