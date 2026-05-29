"""Download real meteorological data from NASA POWER API and generate
24 typical-day scenarios (6 wind x 4 solar) via k-means clustering.

Target location: Inner Mongolia (41.0N, 111.5E) — representative of
Northwest China wind/solar development zones.
"""
import requests
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from config import DATA_DIR

# === Configuration ===
LAT = 41.0    # Inner Mongolia (Hohhot area)
LON = 111.5
YEAR = 2025   # Most recent complete year

# NASA POWER API endpoint
BASE_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"

# Wind speed at 50m → power conversion using simplified power curve
WIND_CUT_IN = 3.0     # m/s
WIND_RATED = 12.0     # m/s
WIND_CUT_OUT = 25.0   # m/s

# Solar GHI → PV power conversion
# GHI in W/m² → capacity factor (simplified: linear with 1000 W/m² = 1.0 pu)


def download_nasa_power(lat, lon, year):
    """Download hourly meteorological data from NASA POWER."""
    params = {
        'parameters': 'WS50M,ALLSKY_SFC_SW_DWN,T2M',
        'community': 'RE',
        'longitude': lon,
        'latitude': lat,
        'start': f'{year}0101',
        'end': f'{year}1231',
        'format': 'JSON',
    }
    print(f"  Downloading NASA POWER data for ({lat}, {lon}), {year}...")
    resp = requests.get(BASE_URL, params=params, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    print(f"  Download complete. Parsing...")
    return data


def wind_speed_to_power(ws):
    """Convert wind speed (m/s) to capacity factor using simplified power curve."""
    if ws < WIND_CUT_IN:
        return 0.0
    elif ws > WIND_CUT_OUT:
        return 0.0
    elif ws < WIND_RATED:
        # Cubic power curve between cut-in and rated
        return ((ws - WIND_CUT_IN) / (WIND_RATED - WIND_CUT_IN)) ** 3
    else:
        return 1.0


def ghi_to_pv_power(ghi):
    """Convert GHI (W/m²) to PV capacity factor.
    Simplified: linear up to 1000 W/m², capped at 1.0.
    """
    return min(ghi / 1000.0, 1.0)


def process_nasa_data(raw_data, lat, lon, year):
    """Process raw NASA POWER JSON into daily wind/solar profiles."""
    props = raw_data['properties']['parameter']
    ws50m = props['WS50M']   # Wind speed at 50m
    ghi = props['ALLSKY_SFC_SW_DWN']  # All-sky surface shortwave down

    # Build DataFrame
    records = []
    for datetime_str in sorted(ws50m.keys()):
        # Format: YYYYMMDDHH
        dt = pd.to_datetime(datetime_str, format='%Y%m%d%H')
        ws = ws50m[datetime_str]
        sw = ghi[datetime_str]
        if ws < -900 or sw < -900:  # missing data flag
            continue
        wind_cf = wind_speed_to_power(ws)
        pv_cf = ghi_to_pv_power(sw)
        records.append({
            'datetime': dt,
            'date': dt.date(),
            'hour': dt.hour,
            'wind_speed_ms': ws,
            'ghi_wm2': sw,
            'wind_cf': round(wind_cf, 6),
            'pv_cf': round(pv_cf, 6),
        })

    df = pd.DataFrame(records)
    print(f"  Processed {len(df)} hourly records ({len(df['date'].unique())} days)")

    # Scale to our park capacity
    df['wind_MW'] = df['wind_cf'] * 40   # 40 MW wind farm
    df['pv_MW'] = df['pv_cf'] * 64       # 64 MW PV farm

    return df


def cluster_daily_profiles(df, column, n_clusters, name):
    """Cluster daily profiles (24h vectors) into n_clusters typical days."""
    # Pivot to daily × hourly matrix
    pivot = df.pivot_table(values=column, index='date', columns='hour', aggfunc='mean')
    pivot = pivot.dropna()  # drop incomplete days

    X = pivot.values  # shape: (n_days, 24)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # Get cluster centroids as typical day profiles
    centroids = kmeans.cluster_centers_

    # Sort centroids by total daily energy (high to low)
    total_energy = centroids.sum(axis=1)
    sorted_idx = np.argsort(-total_energy)  # descending
    centroids_sorted = centroids[sorted_idx]

    # Map old labels to new sorted indices for day counts
    new_labels = np.zeros_like(labels)
    for new_i, old_i in enumerate(sorted_idx):
        new_labels[labels == old_i] = new_i

    print(f"  {name} clustering: {len(pivot)} days → {n_clusters} clusters")
    for i in range(n_clusters):
        count = np.sum(new_labels == i)
        energy = centroids_sorted[i].sum()
        print(f"    Cluster {i+1}: {count} days, daily energy={energy:.1f}")

    # Build output: each cluster = one scenario with 24 hourly values (pu)
    scenarios = {}
    for i in range(n_clusters):
        scenarios[f'{name}_scenario_{i+1}'] = [
            round(v, 4) for v in centroids_sorted[i]
        ]

    return scenarios, centroids_sorted, new_labels


def generate_24_scenarios(wind_scenarios, solar_scenarios, df_with_dates):
    """Cross-combine wind and solar scenarios into 24 combined scenarios."""
    all_scenarios = []
    for wi in range(6):
        for si in range(4):
            sid = wi * 4 + si + 1
            for h in range(24):
                all_scenarios.append({
                    'scenario_id': sid,
                    'wind_scenario': wi + 1,
                    'solar_scenario': si + 1,
                    'hour': h,
                    'hour_label': f'{h:02d}:00',
                    'wind_MW': round(wind_scenarios[f'wind_scenario_{wi+1}'][h], 4),
                    'solar_MW': round(solar_scenarios[f'solar_scenario_{si+1}'][h], 4),
                    'load_MW': round(df_with_dates['pv_MW'].iloc[h], 4),  # placeholder
                })

    df_out = pd.DataFrame(all_scenarios)
    return df_out


def main():
    print("=" * 60)
    print("NASA POWER REAL DATA DOWNLOAD & SCENARIO CLUSTERING")
    print("=" * 60)

    # Step 1: Download
    raw = download_nasa_power(LAT, LON, YEAR)

    # Step 2: Process
    df = process_nasa_data(raw, LAT, LON, YEAR)

    # Save raw processed data
    raw_path = os.path.join(DATA_DIR, 'nasa_power_2025_inner_mongolia.csv')
    df.to_csv(raw_path, index=False)
    print(f"  Raw data saved to {raw_path}")

    # Step 3: Cluster wind profiles
    print("\nClustering daily wind profiles (CF scaled to 40MW)...")
    wind_scenarios, wind_centroids, _ = cluster_daily_profiles(
        df, 'wind_MW', 6, 'wind')

    # Step 4: Cluster solar profiles
    print("\nClustering daily solar profiles (CF scaled to 64MW)...")
    solar_scenarios, solar_centroids, _ = cluster_daily_profiles(
        df, 'pv_MW', 4, 'solar')

    # Step 5: Load profile (use existing typical load, or constant)
    # For now, retain the existing load profile averaged across all days
    df_load = pd.read_csv(os.path.join(DATA_DIR, 'load_actual.csv'))
    load_mw = df_load['load_MW'].values

    # Step 6: Generate 24 combined scenarios
    print("\nGenerating 24 combined wind×solar scenarios...")
    all_scenarios = []
    for wi in range(6):
        for si in range(4):
            sid = wi * 4 + si + 1
            for h in range(24):
                all_scenarios.append({
                    'scenario_id': sid,
                    'wind_scenario': wi + 1,
                    'solar_scenario': si + 1,
                    'hour': h,
                    'hour_label': f'{h:02d}:00',
                    'wind_MW': round(wind_scenarios[f'wind_scenario_{wi+1}'][h], 4),
                    'solar_MW': round(solar_scenarios[f'solar_scenario_{si+1}'][h], 4),
                    'load_MW': round(load_mw[h], 4),
                })

    df_24 = pd.DataFrame(all_scenarios)

    # Save as backup with different name (don't overwrite original)
    out_path = os.path.join(DATA_DIR, 'all_24_scenarios_real.csv')
    df_24.to_csv(out_path, index=False)
    print(f"\n  24 scenarios saved to {out_path}")

    # Step 7: Summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    for sid in range(1, 25):
        df_s = df_24[df_24['scenario_id'] == sid]
        wind_total = df_s['wind_MW'].sum()
        solar_total = df_s['solar_MW'].sum()
        load_total = df_s['load_MW'].sum()
        print(f"  S{sid:2d}: wind={wind_total:.1f} MWh, solar={solar_total:.1f} MWh, "
              f"load={load_total:.1f} MWh, total={wind_total+solar_total:.1f} MWh")

    # Save scenario metadata
    meta = {
        'data_source': 'NASA POWER',
        'location': f'({LAT}, {LON})',
        'year': YEAR,
        'parameters': 'WS50M, ALLSKY_SFC_SW_DWN',
        'wind_clusters': 6,
        'solar_clusters': 4,
        'wind_capacity_MW': 40,
        'solar_capacity_MW': 64,
        'wind_power_curve': f'cubic, cutin={WIND_CUT_IN}m/s, rated={WIND_RATED}m/s',
        'processing_date': time.strftime('%Y-%m-%d'),
    }
    meta_path = os.path.join(DATA_DIR, 'data_source_metadata.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Metadata saved to {meta_path}")


if __name__ == '__main__':
    main()
