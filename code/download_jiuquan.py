"""Download and cluster NASA POWER data for second location: Jiuquan, Gansu.

Then run dual electrolyzer and UF fine grid for comparison with Inner Mongolia.
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
from config import DATA_DIR, OUT_DIR, WIND_CAP_MW, PV_CAP_MW

# === Jiuquan, Gansu ===
LAT = 40.0
LON = 96.5
YEAR = 2025
LOCATION_NAME = "jiuquan_gansu"

BASE_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"

WIND_CUT_IN = 3.0
WIND_RATED = 12.0
WIND_CUT_OUT = 25.0


def wind_speed_to_power(ws):
    if ws < WIND_CUT_IN:
        return 0.0
    elif ws > WIND_CUT_OUT:
        return 0.0
    elif ws < WIND_RATED:
        return ((ws - WIND_CUT_IN) / (WIND_RATED - WIND_CUT_IN)) ** 3
    else:
        return 1.0


def ghi_to_pv_power(ghi):
    return min(ghi / 1000.0, 1.0)


def download_nasa_power(lat, lon, year):
    params = {
        'parameters': 'WS50M,ALLSKY_SFC_SW_DWN,T2M',
        'community': 'RE',
        'longitude': lon,
        'latitude': lat,
        'start': f'{year}0101',
        'end': f'{year}1231',
        'format': 'JSON',
    }
    print(f"  Downloading NASA POWER for ({lat}, {lon}), {year}...")
    resp = requests.get(BASE_URL, params=params, timeout=120)
    resp.raise_for_status()
    return resp.json()


def process_nasa_data(raw_data):
    props = raw_data['properties']['parameter']
    ws50m = props['WS50M']
    ghi = props['ALLSKY_SFC_SW_DWN']

    records = []
    for datetime_str in sorted(ws50m.keys()):
        dt = pd.to_datetime(datetime_str, format='%Y%m%d%H')
        ws = ws50m[datetime_str]
        sw = ghi[datetime_str]
        if ws < -900 or sw < -900:
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
    df['wind_MW'] = df['wind_cf'] * WIND_CAP_MW
    df['pv_MW'] = df['pv_cf'] * PV_CAP_MW
    return df


def cluster_daily_profiles(df, column, n_clusters, name):
    pivot = df.pivot_table(values=column, index='date', columns='hour', aggfunc='mean')
    pivot = pivot.dropna()
    X = pivot.values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_
    total_energy = centroids.sum(axis=1)
    sorted_idx = np.argsort(-total_energy)
    centroids_sorted = centroids[sorted_idx]
    new_labels = np.zeros_like(labels)
    for new_i, old_i in enumerate(sorted_idx):
        new_labels[labels == old_i] = new_i

    print(f"  {name} clustering: {len(pivot)} days -> {n_clusters} clusters")
    for i in range(n_clusters):
        count = np.sum(new_labels == i)
        energy = centroids_sorted[i].sum()
        print(f"    Cluster {i+1}: {count} days, daily energy={energy:.1f}")

    scenarios = {}
    for i in range(n_clusters):
        scenarios[f'{name}_scenario_{i+1}'] = [round(v, 4) for v in centroids_sorted[i]]
    return scenarios, centroids_sorted, new_labels


def generate_24_scenarios(wind_scenarios, solar_scenarios, load_mw):
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
    return pd.DataFrame(all_scenarios)


if __name__ == '__main__':
    print("=" * 60)
    print(f"NASA POWER DATA FOR {LOCATION_NAME.upper().replace('_', ' ')}")
    print("=" * 60)

    # Step 1: Download
    raw = download_nasa_power(LAT, LON, YEAR)
    df = process_nasa_data(raw)

    # Save raw
    raw_path = os.path.join(DATA_DIR, f'nasa_power_2023_{LOCATION_NAME}.csv')
    df.to_csv(raw_path, index=False)
    print(f"  Raw data saved to {raw_path}")

    # Step 2: Cluster
    print("\nClustering daily wind profiles...")
    wind_scenarios, _, _ = cluster_daily_profiles(df, 'wind_MW', 6, 'wind')

    print("\nClustering daily solar profiles...")
    solar_scenarios, _, _ = cluster_daily_profiles(df, 'pv_MW', 4, 'solar')

    # Step 3: Load profile (same as Inner Mongolia)
    df_load = pd.read_csv(os.path.join(DATA_DIR, 'load_actual.csv'))
    load_mw = df_load['load_MW'].values

    # Step 4: Generate 24 scenarios
    df_24 = generate_24_scenarios(wind_scenarios, solar_scenarios, load_mw)
    out_path = os.path.join(DATA_DIR, f'all_24_scenarios_{LOCATION_NAME}.csv')
    df_24.to_csv(out_path, index=False)
    print(f"\n  24 scenarios saved to {out_path}")

    # Summary
    print("\n=== SUMMARY ===")
    for sid in range(1, 25):
        df_s = df_24[df_24['scenario_id'] == sid]
        wind_total = df_s['wind_MW'].sum()
        solar_total = df_s['solar_MW'].sum()
        print(f"  S{sid:2d}: wind={wind_total:.1f} MWh, solar={solar_total:.1f} MWh, "
              f"total={wind_total+solar_total:.1f} MWh")
