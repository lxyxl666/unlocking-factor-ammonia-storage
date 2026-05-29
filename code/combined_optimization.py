"""Combined A+B Joint Optimization + Single-Electrolyzer Baseline.

Three analyses:
1. Dual electrolyzer off-grid — all 24 scenarios × 7 storage levels
2. Single ALKEL-only baseline (40MW, 15% min load, 20%/h ramp) — all 24 scenarios
3. PEMEL flexibility vs storage substitution effect
"""
import pandas as pd
import numpy as np
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from config import *
from dual_electrolyzer_milp import solve_dual_electrolyzer, build_dual_electrolyzer_model
from pulp import PULP_CBC_CMD, value


def run_dual_offgrid_all_scenarios(df, storage_caps):
    """Run dual electrolyzer off-grid for all 24 scenarios × all storage levels."""
    results = []
    scenario_ids = sorted(df['scenario_id'].unique())
    total = len(scenario_ids) * len(storage_caps)
    n = 0
    t0 = time.time()

    for sid in scenario_ids:
        df_s = df[df['scenario_id'] == sid].sort_values('hour')
        Pw = df_s['wind_MW'].values
        Ps = df_s['solar_MW'].values
        Pl = df_s['load_MW'].values

        for cap in storage_caps:
            n += 1
            res = solve_dual_electrolyzer(
                Pw, Ps, Pl, daily_target=72,
                grid_connected=False, storage_cap_MWh=cap,
                maximize_production=True, time_limit_sec=60)
            if res:
                res['scenario_id'] = int(sid)
                res['storage_cap_MWh'] = cap
                results.append(res)

            if n % 20 == 0:
                elapsed = time.time() - t0
                print(f"  Dual off-grid: {n}/{total} ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"  Dual off-grid done: {len(results)} results in {elapsed:.1f}s")
    return results


def run_single_alkel_baseline(df, storage_caps):
    """Run SINGLE aggregated ALKEL electrolyzer as baseline.

    Model: 40MW ALKEL (same total capacity as dual 20+20MW).
    Same SOS2 efficiency, min load 15%, ramp 20%/h, startup 500 yuan.
    No PEMEL — all H2 production goes through ALKEL constraints.

    This simulates the conventional approach of treating electrolyzers
    as one aggregate unit.
    """
    results = []
    scenario_ids = sorted(df['scenario_id'].unique())
    total = len(scenario_ids) * len(storage_caps)
    n = 0
    t0 = time.time()

    for sid in scenario_ids:
        df_s = df[df['scenario_id'] == sid].sort_values('hour')
        Pw = df_s['wind_MW'].values
        Ps = df_s['solar_MW'].values
        Pl = df_s['load_MW'].values

        for cap in storage_caps:
            n += 1
            # Use dual model but force PEMEL to zero to emulate single ALKEL
            # We pass min_load_params to make PEMEL min = 1.0 (impossible → always off)
            ml = {'alkel_min': 0.15, 'pemel_min': 1.00}  # PEMEL can never turn on
            res = solve_dual_electrolyzer(
                Pw, Ps, Pl, daily_target=72,
                grid_connected=False, storage_cap_MWh=cap,
                maximize_production=True, time_limit_sec=60,
                min_load_params=ml)
            if res:
                res['scenario_id'] = int(sid)
                res['storage_cap_MWh'] = cap
                res['model_type'] = 'single_alkel'
                results.append(res)

            if n % 20 == 0:
                elapsed = time.time() - t0
                print(f"  Single ALKEL: {n}/{total} ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"  Single ALKEL done: {len(results)} results in {elapsed:.1f}s")
    return results


def run_single_pemel_baseline(df, storage_caps):
    """Run SINGLE aggregated PEMEL as optimistic upper bound.

    Model: 40MW PEMEL (same total capacity).
    Min load 5%, no ramp constraint, 50 yuan startup.
    Represents the best-case single electrolyzer possible.
    """
    results = []
    scenario_ids = sorted(df['scenario_id'].unique())
    total = len(scenario_ids) * len(storage_caps)
    n = 0
    t0 = time.time()

    for sid in scenario_ids:
        df_s = df[df['scenario_id'] == sid].sort_values('hour')
        Pw = df_s['wind_MW'].values
        Ps = df_s['solar_MW'].values
        Pl = df_s['load_MW'].values

        for cap in storage_caps:
            n += 1
            # PEMEL-like single electrolyzer: min 5%, no ramp
            # Set ALKEL min to 1.0 so it never runs; all through PEMEL
            ml = {'alkel_min': 1.00, 'pemel_min': 0.05}
            res = solve_dual_electrolyzer(
                Pw, Ps, Pl, daily_target=72,
                grid_connected=False, storage_cap_MWh=cap,
                maximize_production=True, time_limit_sec=60,
                min_load_params=ml)
            if res:
                res['scenario_id'] = int(sid)
                res['storage_cap_MWh'] = cap
                res['model_type'] = 'single_pemel'
                results.append(res)

            if n % 20 == 0:
                elapsed = time.time() - t0
                print(f"  Single PEMEL: {n}/{total} ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"  Single PEMEL done: {len(results)} results in {elapsed:.1f}s")
    return results


def compute_comparison_table(df):
    """Build a comparison table: Single ALKEL vs Single PEMEL vs Dual at 0 storage."""
    print("\n=== COMPARISON: Single vs Dual Electrolyzer (No Storage) ===")
    print(f"{'SID':>4s}  {'1xALKEL':>10s}  {'1xPEMEL':>10s}  {'Dual':>10s}  "
          f"{'Dual vs ALKEL':>14s}  {'Dual vs PEMEL':>14s}")
    print(f"{'':>4s}  {'NH3(t)':>10s}  {'NH3(t)':>10s}  {'NH3(t)':>10s}  "
          f"{'gain':>14s}  {'gap':>14s}")
    print("-" * 80)

    rows = []
    for sid in sorted(df['scenario_id'].unique()):
        ds = df[df['scenario_id'] == sid]
        alk = ds[ds['model_type'] == 'single_alkel']
        pem = ds[ds['model_type'] == 'single_pemel']
        dual = ds[ds['model_type'] == 'dual']

        if len(alk) == 0 or len(pem) == 0 or len(dual) == 0:
            continue

        alk_nh3 = alk[alk['storage_cap_MWh'] == 0]['daily_nh3'].values
        pem_nh3 = pem[pem['storage_cap_MWh'] == 0]['daily_nh3'].values
        dual_nh3 = dual[dual['storage_cap_MWh'] == 0]['daily_nh3'].values

        if len(alk_nh3) == 0 or len(pem_nh3) == 0 or len(dual_nh3) == 0:
            continue

        a, p, d = alk_nh3[0], pem_nh3[0], dual_nh3[0]
        gain_vs_alk = (d - a) / a * 100 if a > 0 else 0
        gap_vs_pem = (p - d) / p * 100 if p > 0 else 0

        print(f"{sid:4d}  {a:10.2f}  {p:10.2f}  {d:10.2f}  "
              f"{gain_vs_alk:+13.1f}%  {gap_vs_pem:+13.1f}%")

        rows.append({
            'scenario_id': sid,
            'single_alkel_nh3': round(a, 2),
            'single_pemel_nh3': round(p, 2),
            'dual_nh3': round(d, 2),
            'gain_vs_alkel_pct': round(gain_vs_alk, 1),
            'gap_vs_pemel_pct': round(gap_vs_pem, 1),
        })

    return rows


if __name__ == '__main__':
    print("=" * 60)
    print("JOINT A+B OPTIMIZATION + SINGLE-ELECTROLYZER BASELINE")
    print("=" * 60)

    df = pd.read_csv(os.path.join(DATA_DIR, SCENARIOS_FILE))

    # Storage range
    storage_caps = [0, 5, 10, 20, 30, 50, 100]

    total_solves = len(df['scenario_id'].unique()) * len(storage_caps) * 3
    print(f"\nTotal: 24 scenarios × {len(storage_caps)} storage levels × 3 models "
          f"= {total_solves} solves")
    print(f"Estimated time: ~{total_solves * 8 // 60} min\n")

    # 1. Dual electrolyzer off-grid
    print("[1/3] Dual electrolyzer (ALKEL+PEMEL) off-grid...")
    dual_results = run_dual_offgrid_all_scenarios(df, storage_caps)
    for r in dual_results:
        r['model_type'] = 'dual'

    # 2. Single ALKEL baseline
    print("\n[2/3] Single ALKEL baseline (40MW, 15% min, 20%/h ramp)...")
    alk_results = run_single_alkel_baseline(df, storage_caps)

    # 3. Single PEMEL baseline
    print("\n[3/3] Single PEMEL baseline (40MW, 5% min, unconstrained ramp)...")
    pem_results = run_single_pemel_baseline(df, storage_caps)

    # Merge all
    all_results = dual_results + alk_results + pem_results
    df_all = pd.DataFrame(all_results)

    # Comparison table
    comparison = compute_comparison_table(df_all)

    # Save
    out = {
        'n_results': len(all_results),
        'storage_capacities': storage_caps,
        'comparison_table': comparison,
    }
    with open(os.path.join(OUT_DIR, 'combined_optimization_results.json'), 'w',
              encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2,
                  default=lambda x: float(x) if hasattr(x, 'item') else x)

    df_all.to_csv(os.path.join(OUT_DIR, 'combined_optimization_all.csv'), index=False)
    print(f"\nResults saved to {OUT_DIR}")
