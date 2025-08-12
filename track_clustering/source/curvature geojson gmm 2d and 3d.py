import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
import warnings

# Suppress warnings from matplotlib about finding fonts
warnings.filterwarnings("ignore", category=UserWarning)

HOME_PATH = r'D:\georgiatech_ms_qcf\courses\machine_learning_cs_7641\project\midterm_upload'

def process_track_dataframe(df: pd.DataFrame, track_name: str) -> dict:
    """
    Takes a DataFrame with x_m, y_m columns, calculates physical properties,
    and returns a dictionary of key performance metrics. This is the core
    analysis logic, refactored to be reusable.

    Args:
        df (pd.DataFrame): DataFrame containing 'x_m' and 'y_m' columns.
        track_name (str): The name of the track for labeling.

    Returns:
        dict: A dictionary of calculated features for the track.
    """
    try:

        delta_x = df['x_m'].diff()
        delta_y = df['y_m'].diff()
        segment_length = np.sqrt(delta_x**2 + delta_y**2)
        segment_length.iloc[0] = 0
        df['distance'] = segment_length.cumsum()

        # 2. Calculate Curvature
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
    classification pipeline.
    """
    
    # geojson_path = Path('f1-circuits_2022-2025.geojson')
    geojson_path = Path(os.path.join(HOME_PATH, 'data', 'f1-circuits_2022-2025.geojson'))

    if not geojson_path.exists():
        print(f"Error: GeoJSON file not found at '{geojson_path}'")
        print("Please ensure the folder 'f1-circuits-master' exists relative to where you are running the script.")
        return

    print(f"Loading track data from {geojson_path}...")
    with geojson_path.open('r', encoding='utf-8') as f:
        geojson_data = json.load(f)

    all_track_data = []
    for feature in geojson_data['features']:
        props = feature.get('properties', {})
        geom = feature.get('geometry', {})

        if geom.get('type') != 'LineString':
            continue

        track_name = props.get('Name', f"Unnamed Track {props.get('id', '')}")
        coordinates = geom.get('coordinates', [])
        
        if len(coordinates) < 3:
            continue

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
        print("No valid track data was processed from the GeoJSON file. Exiting.")
        return

    print(f"Successfully processed {len(all_track_data)} tracks.")
    summary_df = pd.DataFrame(all_track_data).set_index('track_name')
    features = summary_df[['percent_straights', 'avg_abs_curvature', 'num_corners', 'std_curvature']]

    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    gmm = GaussianMixture(n_components=2, random_state=42, n_init=10)
    summary_df['cluster'] = gmm.fit_predict(scaled_features)
    probabilities = gmm.predict_proba(scaled_features)

    
    cluster_means = summary_df.groupby('cluster')['percent_straights'].mean()
    high_speed_cluster_id = cluster_means.idxmax()
    print(f"\nCluster {high_speed_cluster_id} identified as 'High Speed' (has higher average % of straights).")

    summary_df['classification'] = summary_df['cluster'].apply(
        lambda x: 'High Speed' if x == high_speed_cluster_id else 'Low Speed'
    )
    summary_df['high_speed_prob'] = probabilities[:, high_speed_cluster_id]

    
    print("\n--- Track Classification Summary (Full Data) ---")
    sorted_summary = summary_df.sort_values('high_speed_prob', ascending=False)
    print(sorted_summary[['classification', 'high_speed_prob', 'percent_straights', 'avg_abs_curvature', 'num_corners', 'std_curvature']].round(3))

    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    
    fig2d, ax2d = plt.subplots(figsize=(14, 10))
    high_speed_tracks = summary_df[summary_df['classification'] == 'High Speed']
    low_speed_tracks = summary_df[summary_df['classification'] == 'Low Speed']

    ax2d.scatter(high_speed_tracks['percent_straights'], high_speed_tracks['avg_abs_curvature'], c='royalblue', label='High Speed', s=60, alpha=0.8)
    ax2d.scatter(low_speed_tracks['percent_straights'], low_speed_tracks['avg_abs_curvature'], c='tomato', label='Low Speed', s=60, alpha=0.8)

    for track_name, row in summary_df.iterrows():
        ax2d.text(row['percent_straights'] + 0.2, row['avg_abs_curvature'], track_name, fontsize=9)

    ax2d.set_xlabel('Percentage of Track as Straights (%)', fontsize=12)
    ax2d.set_ylabel('Average Absolute Curvature (rad/m)', fontsize=12)
    ax2d.set_title('2D Track Classification from GeoJSON using GMM', fontsize=16)
    ax2d.legend()
    plt.tight_layout()

    
    fig3d = plt.figure(figsize=(15, 12))
    ax3d = fig3d.add_subplot(111, projection='3d')

    ax3d.scatter(
        xs=high_speed_tracks['percent_straights'], 
        ys=high_speed_tracks['avg_abs_curvature'], 
        zs=high_speed_tracks['std_curvature'], 
        c='royalblue', label='High Speed', s=50
    )
    ax3d.scatter(
        xs=low_speed_tracks['percent_straights'], 
        ys=low_speed_tracks['avg_abs_curvature'], 
        zs=low_speed_tracks['std_curvature'], 
        c='tomato', label='Low Speed', s=50
    )

    ax3d.set_xlabel('Percentage of Straights (%)')
    ax3d.set_ylabel('Average Absolute Curvature')
    ax3d.set_zlabel('Standard Deviation of Curvature')
    ax3d.set_title('3D Track Classification using GMM', fontsize=16)
    ax3d.legend()

    plt.show()

if __name__ == '__main__':
    main()
