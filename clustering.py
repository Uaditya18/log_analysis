import os
import json
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from collections import defaultdict, Counter
from sklearn.metrics import silhouette_score
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_logs_from_file(log_path):
    """
    Load log entries from a file with robust encoding handling.

    Args:
        log_path: Path to the log file.

    Returns:
        list: List of log entries (strings).
    """
    logs = []
    encodings = ['utf-8', 'latin-1', 'iso-8859-1']  # Try these encodings
    for encoding in encodings:
        try:
            with open(log_path, 'r', encoding=encoding, errors='ignore') as file:
                logs = [line.strip() for line in file if line.strip()]
            logger.info(f"Successfully read {log_path} with {encoding} encoding")
            return logs
        except UnicodeDecodeError as e:
            logger.warning(f"Failed to read {log_path} with {encoding} encoding: {e}")
            continue
        except Exception as e:
            logger.error(f"Error reading {log_path}: {e}")
            return []
    logger.error(f"Could not read {log_path} with any encoding. Returning empty list.")
    return []

def vectorize_logs(log_lines, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(log_lines)
    return embeddings

def cluster_logs(embeddings, eps=0.3, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    cluster_labels = dbscan.fit_predict(embeddings)
    return cluster_labels

def group_logs_by_cluster(logs, labels):
    clustered_logs = defaultdict(list)
    for log, label in zip(logs, labels):
        clustered_logs[label].append(log)
    return clustered_logs

def export_clusters_to_json(clustered_logs, output_file="clustered_logs.json"):
    export_data = []
    for cluster_id, logs in sorted(clustered_logs.items()):
        export_data.append({
            "cluster_id": int(cluster_id),
            "log_count": int(len(logs)),
            "logs": logs
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2)
    print(f"\n📝 Clustered logs exported to: {output_file}")

# === 🏁 Pipeline Entrypoint ===
if __name__ == "__main__":
    log_file_path = "data\\logs\\weblogs\\Apache.log"  # Use double backslashes on Windows
    assert os.path.exists(log_file_path), f"{log_file_path} not found!"

    print("🔄 Loading logs...")
    logs = load_logs_from_file(log_file_path)

    print("🧠 Generating embeddings...")
    embeddings = vectorize_logs(logs)

    print("📊 Clustering logs...")
    labels = cluster_logs(embeddings)

    print("📦 Organizing clustered logs...")
    clustered = group_logs_by_cluster(logs, labels)

    print("🖨️ Printing clustered logs...")
    for cid, logs in clustered.items():
        print(f"\n\n=== 🧩 Cluster {cid} ({len(logs)} log(s)) ===")
        for log in logs:
            print(log)

    # === 🔢 Cluster Metrics ===
    cluster_sizes = Counter(labels)
    n_clusters = len([cid for cid in cluster_sizes if cid != -1])
    n_noise = cluster_sizes.get(-1, 0)

    print(f"\n📊 Clustering Summary:")
    print(f"🔹 Total logs: {len(logs)}")
    print(f"🔸 Number of clusters (excluding noise): {n_clusters}")
    print(f"🔸 Noise points (Cluster -1): {n_noise}")
    print(f"📦 Cluster sizes:")
    for cid, size in sorted(cluster_sizes.items()):
        print(f"   ▪ Cluster {cid}: {size} logs")

    if n_clusters >= 2:
        sil_score = silhouette_score(embeddings, labels, metric='cosine')
        print(f"📈 Silhouette Score: {sil_score:.4f}")

    # === 💾 Export to JSON ===
    export_clusters_to_json(clustered, "clustered_logs.json")