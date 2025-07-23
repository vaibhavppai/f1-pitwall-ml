import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings

# Suppress warnings from matplotlib about finding fonts
warnings.filterwarnings("ignore", category=UserWarning)

HOME_PATH = r'D:\georgiatech_ms_qcf\courses\machine_learning_cs_7641\project\midterm_upload'

def process_track_dataframe(df: pd.DataFrame, track_name: str) -> dict:
    """
    Takes a DataFrame with x_m, y_m columns, calculates physical properties,
    and returns a dictionary of key performance metrics.
    """
    try:

        delta_x = df['x_m'].diff()
        delta_y = df['y_m'].diff()
        segment_length = np.sqrt(delta_x**2 + delta_y**2)
        segment_length.iloc[0] = 0
        df['distance'] = segment_length.cumsum()

        dx_ds = np.gradient(df['x_m'])
        dy_ds = np.gradient(df['y_m'])
        d2x_ds2 = np.gradient(dx_ds)
        d2y_ds2 = np.gradient(dy_ds)
        
        numerator = (dx_ds * d2y_ds2 - dy_ds * d2x_ds2)
        denominator = (dx_ds**2 + dy_ds**2)**(1.5)
        
        curvature = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
        df['curvature'] = curvature
        df.fillna(0, inplace=True)


        straight_threshold = 0.002
        percent_straights = (df['curvature'].abs() < straight_threshold).sum() / len(df) * 100
        avg_abs_curvature = df['curvature'].abs().mean()
        corner_threshold = 0.015
        in_corner = df['curvature'].abs() > corner_threshold
        num_corners = (in_corner & ~in_corner.shift(fill_value=False)).sum()
        std_curvature = df['curvature'].std()

        return {
            'track_name': track_name,
            'percent_straights': percent_straights,
            'avg_abs_curvature': avg_abs_curvature,
            'num_corners': num_corners,
            'std_curvature': std_curvature
        }
    except Exception as e:
        print(f"Could not process track '{track_name}'. Error: {e}")
        return None

def main():
    """
    Main function to load track data from a GeoJSON file and run the
    classification pipeline with 4 clusters.
    """

    # geojson_path = Path('f1-circuits_2022-2025.geojson' )
    geojson_path = Path(os.path.join(HOME_PATH, 'data', 'f1-circuits_2022-2025.geojson'))

    if not geojson_path.exists():
        print(f"Error: GeoJSON file not found at '{geojson_path}'")
        return

    print(f"Loading track data from {geojson_path}...")
    with geojson_path.open('r', encoding='utf-8') as f:
        geojson_data = json.load(f)

    all_track_data = []
    for feature in geojson_data['features']:
        props = feature.get('properties', {})
        geom = feature.get('geometry', {})
        if geom.get('type') != 'LineString': continue
        track_name = props.get('Name', f"Unnamed Track {props.get('id', '')}")
        coordinates = geom.get('coordinates', [])
        if len(coordinates) < 3: continue

        lon, lat = zip(*coordinates)
        df = pd.DataFrame({'lon': lon, 'lat': lat})

        R = 6371e3
        lat_avg_rad = np.deg2rad(df['lat'].mean())
        df['x_m'] = R * np.deg2rad(df['lon']) * np.cos(lat_avg_rad)
        df['y_m'] = R * np.deg2rad(df['lat'])
        df['x_m'] -= df['x_m'].iloc[0]
        df['y_m'] -= df['y_m'].iloc[0]
        
        track_metrics = process_track_dataframe(df, track_name)
        if track_metrics:
            all_track_data.append(track_metrics)

    if not all_track_data:
        print("No valid track data was processed. Exiting.")
        return

    print(f"Successfully processed {len(all_track_data)} tracks.")
    summary_df = pd.DataFrame(all_track_data).set_index('track_name')
    features = summary_df[['percent_straights', 'avg_abs_curvature', 'num_corners', 'std_curvature']]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    gmm = GaussianMixture(n_components=4, random_state=42, n_init=10)
    summary_df['cluster'] = gmm.fit_predict(scaled_features)

    cluster_centers = summary_df.groupby('cluster').mean()
    print("\n--- Cluster Centers (Mean Values) ---")
    print(cluster_centers.round(3))

    # Define archetypes based on median values of key metrics
    median_straights = summary_df['percent_straights'].median()
    median_std_curv = summary_df['std_curvature'].median()

    def classify_cluster(row):
        speed = "High Speed" if row['percent_straights'] > median_straights else "Low Speed"
        style = "Stop-and-Go" if row['std_curvature'] > median_std_curv else "Flowing"
        return f"{speed} - {style}"

    cluster_map = cluster_centers.apply(classify_cluster, axis=1).to_dict()
    summary_df['classification'] = summary_df['cluster'].map(cluster_map)

    # --- 4. Display Results ---
    print("\n--- 4-Way Track Classification Summary ---")
    sorted_summary = summary_df.sort_values(['classification', 'percent_straights'], ascending=[True, False])
    print(sorted_summary[['classification', 'percent_straights', 'avg_abs_curvature', 'num_corners', 'std_curvature']].round(3))

    plt.style.use('seaborn-v0_8-whitegrid')
    colors = ['royalblue', 'tomato', 'forestgreen', 'darkorange']
    
    # 2D Visualization
    fig2d, ax2d = plt.subplots(figsize=(16, 12))
    for i, (name, group) in enumerate(summary_df.groupby('classification')):
        ax2d.scatter(group['percent_straights'], group['avg_abs_curvature'], c=colors[i], label=name, s=60, alpha=0.8)
    
    for track_name, row in summary_df.iterrows():
        ax2d.text(row['percent_straights'] + 0.2, row['avg_abs_curvature'], track_name, fontsize=9)

    ax2d.set_xlabel('Percentage of Track as Straights (%)', fontsize=12)
    ax2d.set_ylabel('Average Absolute Curvature (rad/m)', fontsize=12)
    ax2d.set_title('2D Track Classification (4 Clusters)', fontsize=16)
    ax2d.legend()
    plt.tight_layout()

    # 3D Visualization
    fig3d = plt.figure(figsize=(16, 12))
    ax3d = fig3d.add_subplot(111, projection='3d')
    
    for i, (name, group) in enumerate(summary_df.groupby('classification')):
        ax3d.scatter(
            xs=group['percent_straights'], 
            ys=group['avg_abs_curvature'], 
            zs=group['std_curvature'], 
            c=colors[i], label=name, s=50
        )

    ax3d.set_xlabel('Percentage of Straights (%)')
    ax3d.set_ylabel('Average Absolute Curvature')
    ax3d.set_zlabel('Standard Deviation of Curvature')
    ax3d.set_title('3D Track Classification (4 Clusters)', fontsize=16)
    ax3d.legend()

    plt.show()

if __name__ == '__main__':
    main()
