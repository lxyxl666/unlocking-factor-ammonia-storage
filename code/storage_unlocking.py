"""Storage Unlocking Factor Analysis (Plan B).

Quantifies the "storage as production enabler" mechanism:
- Without storage: brief renewable deficits force electrolyzer shutdown,
  wasting surrounding hours of renewable surplus
- With small storage: deficit is bridged, electrolyzer maintains minimum
  load, amplifies renewable utilization by ~3-10x the storage capacity

The Unlocking Factor (UF) is defined as:
  UF = (E_ren_used_with_storage - E_ren_used_without_storage) / S_cap

Runs on all 24 scenarios at 1 MWh resolution (0-100 MWh).
"""
import pandas as pd
import numpy as np
import json
import os
from pulp import (LpProblem, LpVariable, LpMinimize, lpSum, value,
                  PULP_CBC_CMD, LpBinary, LpContinuous)

from config import *
from dual_electrolyzer_milp import solve_dual_electrolyzer


def compute_unlocking_factor(scenario_data, storage_cap_MWh,
                             grid_connected=False):
    """Compute UF for a single scenario-storage combination.

    Args:
        scenario_data: dict with 'P_wind', 'P_pv', 'P_load' (24h numpy arrays)
        storage_cap_MWh: storage capacity

    Returns:
        dict with: daily_nh3, ren_used_MWh, ton_cost, uf (may be None for 0 storage)
    """
    Pw = scenario_data['P_wind']
    Ps = scenario_data['P_pv']
    Pl = scenario_data['P_load']

    res = solve_dual_electrolyzer(
        Pw, Ps, Pl, daily_target=72,  # max possible, solver pushes up
        grid_connected=False,
        storage_cap_MWh=storage_cap_MWh,
        maximize_production=True,  # maximize NH3, not meet fixed target
        time_limit_sec=60)

    if res is None:
        return None

    return {
        'storage_MWh': storage_cap_MWh,
        'daily_nh3': res['daily_nh3'],
        'ton_cost': res['ton_cost'],
        'ren_used_MWh': res['total_load_MWh'],
        'total_ren_MWh': res['total_ren_MWh'],
        'equipment_utilization': res['equipment_utilization'],
    }


def analyze_all_scenarios(df_scenarios, storage_range=None):
    """Compute UF across all 24 scenarios for the given storage range.

    Args:
        df_scenarios: DataFrame with all 24 scenarios
        storage_range: list of storage capacities in MWh (default: 0-100 step 1)

    Returns:
        full_results: list of dicts per scenario-storage combination
        summary: aggregated statistics
    """
    if storage_range is None:
        storage_range = list(range(0, 101, 1))  # 0-100 MWh at 1 MWh step

    scenario_ids = sorted(df_scenarios['scenario_id'].unique())
    full_results = []

    for sid in scenario_ids:
        df_s = df_scenarios[df_scenarios['scenario_id'] == sid].sort_values('hour')
        data = {
            'P_wind': df_s['wind_MW'].values,
            'P_pv': df_s['solar_MW'].values,
            'P_load': df_s['load_MW'].values,
        }

        # Baseline: no storage
        base = compute_unlocking_factor(data, 0)
        if base is None:
            # Off-grid infeasible even at zero production: some hours
            # have renewable < base load, making off-grid physically impossible.
            # These scenarios REQUIRE grid connection or load shedding.
            print(f"  S{sid}: baseline infeasible (ren < load in some hours) — "
                  f"off-grid NOT VIABLE")
            continue

        base_ren_used = base['ren_used_MWh']
        base_nh3 = base['daily_nh3']

        for cap in storage_range:
            res = compute_unlocking_factor(data, cap)
            if res is None:
                continue

            delta_ren = res['ren_used_MWh'] - base_ren_used
            uf = delta_ren / cap if cap > 0 else 0
            delta_nh3 = res['daily_nh3'] - base_nh3

            res['scenario_id'] = int(sid)
            res['delta_ren_used_MWh'] = round(delta_ren, 4)
            res['unlocking_factor'] = round(uf, 4)
            res['delta_nh3_tons'] = round(delta_nh3, 4)
            full_results.append(res)

        # Progress
        best_nh3 = max(
            [r for r in full_results if r['scenario_id'] == int(sid)],
            key=lambda x: x['daily_nh3'])
        best_cost = min(
            [r for r in full_results if r['scenario_id'] == int(sid)],
            key=lambda x: x['ton_cost'])

        print(f"  S{sid:2d}: base NH3={base_nh3:.2f}t, "
              f"best NH3={best_nh3['daily_nh3']:.2f}t "
              f"(@{best_nh3['storage_MWh']}MWh), "
              f"best cost={best_cost['ton_cost']:.2f} "
              f"(@{best_cost['storage_MWh']}MWh)")

    # Summary statistics
    if not full_results:
        print("  WARNING: No feasible results across any scenario!")
        return [], {'error': 'no_feasible_results', 'n_scenarios': len(scenario_ids)}

    df = pd.DataFrame(full_results)
    df_nonzero = df[df['storage_MWh'] > 0]

    summary = {
        'n_scenarios': len(scenario_ids),
        'storage_range_MWh': [min(storage_range), max(storage_range)],
        'uf_mean': round(df_nonzero['unlocking_factor'].mean(), 4),
        'uf_median': round(df_nonzero['unlocking_factor'].median(), 4),
        'uf_std': round(df_nonzero['unlocking_factor'].std(), 4),
        'uf_max': round(df_nonzero['unlocking_factor'].max(), 4),
        'uf_min': round(df_nonzero['unlocking_factor'].min(), 4),
        # UF at optimal (cheapest) storage for each scenario
        'uf_at_optimal': {},
    }

    # Per-scenario optimal UF
    for sid in scenario_ids:
        sid_results = df[df['scenario_id'] == int(sid)]
        if len(sid_results) == 0:
            continue
        best = sid_results.loc[sid_results['ton_cost'].idxmin()]
        summary['uf_at_optimal'][str(sid)] = {
            'optimal_storage_MWh': int(best['storage_MWh']),
            'uf': round(best['unlocking_factor'], 4),
            'daily_nh3': round(best['daily_nh3'], 4),
            'ton_cost': round(best['ton_cost'], 2),
        }

    # Classify scenarios by storage sensitivity
    for sid in scenario_ids:
        sid_results = df[df['scenario_id'] == int(sid)]
        if len(sid_results) == 0:
            continue
        uf_10 = sid_results[sid_results['storage_MWh'] == 10]
        if len(uf_10) > 0:
            uf_val = uf_10.iloc[0]['unlocking_factor']
        else:
            uf_val = 0
        summary[f'scenario_{sid}_uf_at_10MWh'] = round(uf_val, 4)

    return full_results, summary


if __name__ == '__main__':
    print("=" * 60)
    print("STORAGE UNLOCKING FACTOR ANALYSIS (PLAN B)")
    print("=" * 60)

    df_scenarios = pd.read_csv(os.path.join(DATA_DIR, SCENARIOS_FILE))

    # Quick test: 0-100 MWh at 10 MWh step
    print("\n--- Coarse grid (10 MWh step) ---")
    storage_range = list(range(0, 101, 10))
    results, summary = analyze_all_scenarios(df_scenarios, storage_range)

    print(f"\n=== UF Summary ===")
    print(f"Mean UF: {summary['uf_mean']:.4f}")
    print(f"Median UF: {summary['uf_median']:.4f}")
    print(f"Max UF: {summary['uf_max']:.4f}")
    print(f"UF at optimal storage per scenario:")
    for sid, info in sorted(summary['uf_at_optimal'].items(),
                            key=lambda x: int(x[0])):
        print(f"  S{int(sid):2d}: opt_storage={info['optimal_storage_MWh']:3d}MWh, "
              f"UF={info['uf']:.3f}, NH3={info['daily_nh3']:.2f}t, "
              f"cost={info['ton_cost']}")

    # Save
    with open(os.path.join(OUT_DIR, 'unlocking_factor_results.json'), 'w',
              encoding='utf-8') as f:
        json.dump({
            'summary': summary,
            'n_results': len(results),
        }, f, ensure_ascii=False, indent=2)

    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(OUT_DIR, 'unlocking_factor_all.csv'),
                      index=False)

    print(f"\nResults saved to {OUT_DIR}")
