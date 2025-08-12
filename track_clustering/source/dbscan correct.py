import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from pathlib import Path

HOME_PATH = r'D:\georgiatech_ms_qcf\courses\machine_learning_cs_7641\project\midterm_upload'

def load_circuit_properties(geojson_path = Path(os.path.join(HOME_PATH, 'data', 'f1-circuits_2022-2025.geojson'))) -> pd.DataFrame:
    """
    Reads the f1-circuits.geojson file and extracts circuit properties.
    """
    geojson_path = Path(geojson_path)
    with geojson_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    records = []
    for feature in data.get('features', []):
        props = feature.get('properties', {})
        records.append({
            'id': props.get('id'),
            'Name': props.get('Name'),
            'length': props.get('length'),
            'altitude': props.get('altitude')
        })
    df = pd.DataFrame(records)
    return df.dropna(subset=['length','altitude'])

def cluster_and_treat_noise_as_clusters(features: np.ndarray, eps: float, min_samples: int):
    """
    Performs DBSCAN clustering and then assigns a unique cluster ID to every noise point.

    This ensures that every point belongs to a cluster and that isolated points
    are treated as their own unique, single-point clusters.

    Returns:
        np.ndarray: An array of labels where no point is labeled -1.
    """
    
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(features)

   
    max_cluster_id = np.max(labels)
    
    
    next_cluster_id = max_cluster_id + 1

    
    noise_indices = np.where(labels == -1)[0]
    for index in noise_indices:
        labels[index] = next_cluster_id
        next_cluster_id += 1
        
    return labels

def plot_clusters_properties(df: pd.DataFrame, labels: np.ndarray, eps: float, min_samples: int):
    
    plt.figure(figsize=(12, 8))

    
    scatter = plt.scatter(
        df['length'],
        df['altitude'],
        c=labels,
        cmap='tab20b_r', # 'hsv' is good for many distinct colors
        s=60
    )

    plt.xlabel('Length (m)')
    plt.ylabel('Altitude (m)')

    plt.title(f'DBSCAN with Noise as Individual Clusters (Initial eps={eps}, min_samples={min_samples})')

    unique_labels = np.unique(labels)
    if len(unique_labels) < 25:
        cbar = plt.colorbar(scatter, label='Cluster ID')
        cbar.set_ticks(unique_labels)

    plt.tight_layout()
    plt.show()


def main():
    df = load_circuit_properties(geojson_path = Path(os.path.join(HOME_PATH, 'data', 'f1-circuits_2022-2025.geojson')))
    print(f"Loaded {len(df)} circuits from Directory")

    features = df[['length', 'altitude']].to_numpy()

    
    eps_val = 200
    min_samples_val = 2 

    
    labels = cluster_and_treat_noise_as_clusters(
        features,
        eps=eps_val,
        min_samples=min_samples_val
    )

    unique_labels = set(labels)
    n_clusters = len(unique_labels)
    n_noise = np.sum(np.array(labels) == -1)
    
    print(f"\nClustering with initial eps={eps_val} and min_samples={min_samples_val}")
    print(f"Found {n_clusters} final clusters (including single-point clusters).")
    print(f"Final noise points: {n_noise}")

    
    df['cluster'] = labels
    print("\nSample of clustered data:")
    print(df.sort_values('cluster').tail(10))

    
    plot_clusters_properties(df, labels, eps=eps_val, min_samples=min_samples_val)

if __name__ == '__main__':
    main()