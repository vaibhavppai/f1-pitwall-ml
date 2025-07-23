import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path

HOME_PATH = r'D:\georgiatech_ms_qcf\courses\machine_learning_cs_7641\project\midterm_upload'

def load_circuit_properties(geojson_path = Path(os.path.join(HOME_PATH, 'data', 'f1-circuits_2022-2025.geojson'))) -> pd.DataFrame:
    """
    Reads the f1-circuits.geojson file and extracts all track coordinates as a 2D array of [lon, lat].
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


def cluster_features(features: np.ndarray, n_components: int, random_state=42):
    """
    Performs Gaussian Mixture Model clustering on 2D features.
    Returns fitted GMM, labels, and scaler.
    """
    scaler = StandardScaler()
    feats_scaled = scaler.fit_transform(features)
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    labels = gmm.fit_predict(feats_scaled)
    return gmm, labels, scaler


def plot_clusters_properties(df: pd.DataFrame, labels: np.ndarray):
    """
    Scatter plot of circuit length vs altitude colored by cluster.
    """
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(df['length'], df['altitude'], c=labels, cmap='viridis', s=50)
    plt.xlabel('Length (m)')
    plt.ylabel('Altitude (m)')
    plt.title('GMM Clusters on Circuit Length and Altitude')
    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()
    plt.show()


def main():
    # Load properties
    df = load_circuit_properties(geojson_path = Path(os.path.join(HOME_PATH, 'data', 'f1-circuits_2022-2025.geojson')))
    print(f"Loaded {len(df)} circuits from {'f1-circuits.geojson'}")
    
    # Prepare feature array
    features = df[['length','altitude']].values

    # Cluster
    gmm, labels, scaler = cluster_features(features, n_components=9)
    print(f"GMM converged: {gmm.converged_}, components: {9}")

    # Attach labels
    df['cluster'] = labels
    print(df[['id','Name','length','altitude','cluster']].head())

    # Plot
    plot_clusters_properties(df, labels)

if __name__ == '__main__':
    main()