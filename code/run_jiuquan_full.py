"""Full pipeline for Jiuquan: dual electrolyzer + UF + comparison with Inner Mongolia."""
import pandas as pd
import numpy as np
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from config import *
from dual_electrolyzer_milp import solve_dual_electrolyzer
from storage_unlocking import compute_unlocking_factor

LOCATION = "jiuquan_gansu"
SCENARIOS_PATH = os.path.join(DATA_DIR, f'all_24_scenarios_{LOCATION}.csv')

if __name__ == '__main__':
    print("=" * 60)
    print(f"FULL PIPELINE: {LOCATION.upper().replace('_', ' ')}")
    print("=" * 60)

    df = pd.read_csv(SCENARIOS_PATH)

    # ================================================================
    # Part 1: Dual electrolyzer grid-connected
    # ================================================================
    print("\n[1/3] Dual electrolyzer (grid-connected)...")
    scenario_ids = sorted(df['scenario_id'].unique())
    all_dual = []
    scenario_best = {}

    for sid in scenario_ids:
        df_s = df[df['scenario_id'] == sid].sort_values('hour')
        Pw = df_s['wind_MW'].values
        Ps = df_s['solar_MW'].values
        Pl = df_s['load_MW'].values

        best = None
        for D in PRODUCTION_LEVELS:
            res = solve_dual_electrolyzer(Pw, Ps, Pl, D, grid_connected=True, time_limit_sec=60)
            if res:
                res['scenario_id'] = int(sid)
                res['production'] = D
                all_dual.append(res)
                if best is None or res['ton_cost'] < best['ton_cost']:
                    best = res

        if best:
            scenario_best[int(sid)] = best
            print(f"  S{sid:2d}: best={best['ton_cost']:.0f} yuan/t @ {best['production']}t/d, "
                  f"pass={best['pass_count']}/3")

    annual_nh3 = sum(v['daily_target'] * DAYS_PER_SCENARIO for v in scenario_best.values())
    annual_cost = sum(v['total_cost'] * DAYS_PER_SCENARIO for v in scenario_best.values())
    avg_cost = annual_cost / annual_nh3 if annual_nh3 > 0 else 0
    compliant = sum(1 for v in scenario_best.values() if v['all_pass'])

    print(f"  Annual: {annual_nh3:.0f}t, {annual_cost:.0f} yuan, "
          f"{avg_cost:.2f} yuan/t, {compliant}/24 full compliance")

    # ================================================================
    # Part 2: Storage UF (coarse + fine for key scenarios)
    # ================================================================
    print("\n[2/3] Storage UF analysis...")

    # Coarse grid: all 24 scenarios, 10 MWh step
    uf_results = []
    for sid in scenario_ids:
        df_s = df[df['scenario_id'] == sid].sort_values('hour')
        data = {
            'P_wind': df_s['wind_MW'].values,
            'P_pv': df_s['solar_MW'].values,
            'P_load': df_s['load_MW'].values,
        }

        base = compute_unlocking_factor(data, 0)
        if base is None:
            print(f"  S{sid}: baseline infeasible, skipping")
            continue

        base_ren = base['ren_used_MWh']
        base_nh3 = base['daily_nh3']

        for cap in range(0, 101, 10):
            res = compute_unlocking_factor(data, cap)
            if res:
                delta_ren = res['ren_used_MWh'] - base_ren
                uf = delta_ren / cap if cap > 0 else 0
                delta_nh3 = res['daily_nh3'] - base_nh3
                res['scenario_id'] = int(sid)
                res['delta_ren_used_MWh'] = round(delta_ren, 4)
                res['unlocking_factor'] = round(uf, 4)
                res['delta_nh3_tons'] = round(delta_nh3, 4)
                uf_results.append(res)

        best_nh3 = max((r for r in uf_results if r['scenario_id'] == sid),
                       key=lambda x: x['daily_nh3'])
        print(f"  S{sid:2d}: base={base_nh3:.1f}t, max={best_nh3['daily_nh3']:.1f}t "
              f"(@{best_nh3['storage_MWh']}MWh)")

    # Find top UF scenarios for fine grid
    df_uf = pd.DataFrame(uf_results)
    df_nz = df_uf[df_uf['storage_MWh'] > 0]
    uf_by_sid = df_nz.groupby('scenario_id')['unlocking_factor'].max().sort_values(ascending=False)
    top_sids = uf_by_sid.head(5).index.tolist()
    print(f"\n  Top UF scenarios for fine grid: {top_sids}")
    print(f"  Max coarse UF: {uf_by_sid.iloc[0]:.2f} (S{int(top_sids[0])})")

    # Fine grid for top 5
    fine_results = {}
    for sid in top_sids:
        df_s = df[df['scenario_id'] == sid].sort_values('hour')
        data = {
            'P_wind': df_s['wind_MW'].values,
            'P_pv': df_s['solar_MW'].values,
            'P_load': df_s['load_MW'].values,
        }
        base = compute_unlocking_factor(data, 0)
        base_ren = base['ren_used_MWh']

        results = []
        for cap in range(0, 101, 1):
            res = compute_unlocking_factor(data, cap)
            if res:
                delta_ren = res['ren_used_MWh'] - base_ren
                uf = delta_ren / cap if cap > 0 else 0
                res['unlocking_factor'] = round(uf, 4)
                res['delta_ren_used_MWh'] = round(delta_ren, 4)
                results.append(res)

        best_uf = max((r for r in results if r.get('unlocking_factor', 0) > 0),
                      key=lambda r: r.get('unlocking_factor', 0), default=None)
        if best_uf:
            print(f"  S{int(sid)} fine: UF={best_uf['unlocking_factor']:.2f} "
                  f"@{best_uf['storage_MWh']}MWh")
        fine_results[f"S{int(sid)}"] = results

    # ================================================================
    # Part 3: Comparison Inner Mongolia vs Jiuquan
    # ================================================================
    print("\n[3/3] Cross-location comparison...")

    # Load Inner Mongolia results for comparison
    im_dual_path = os.path.join(OUT_DIR, 'dual_electrolyzer_results.json')
    im_uf_path = os.path.join(OUT_DIR, 'unlocking_factor_all.csv')

    comparison = {
        'jiuquan': {
            'avg_ton_cost': round(avg_cost, 2),
            'annual_nh3': round(annual_nh3, 0),
            'full_compliance': compliant,
            'coarse_uf_max': round(float(uf_by_sid.iloc[0]), 2) if len(uf_by_sid) > 0 else 0,
        }
    }

    if os.path.exists(im_dual_path):
        with open(im_dual_path) as f:
            im = json.load(f)
        comparison['inner_mongolia'] = {
            'avg_ton_cost': im['annual_summary']['avg_ton_cost'],
            'annual_nh3': im['annual_summary']['annual_nh3_tons'],
            'full_compliance': im['annual_summary']['compliance']['full'],
        }

    if os.path.exists(im_uf_path):
        df_im_uf = pd.read_csv(im_uf_path)
        df_im_nz = df_im_uf[df_im_uf['storage_MWh'] > 0]
        im_max_uf = df_im_nz.groupby('scenario_id')['unlocking_factor'].max()
        comparison['inner_mongolia']['coarse_uf_max'] = round(float(im_max_uf.max()), 2)

    print(f"\n  {'':>20s} {'Inner Mongolia':>16s} {'Jiuquan':>16s}")
    print(f"  {'-'*55}")
    for key in ['avg_ton_cost', 'annual_nh3', 'full_compliance', 'coarse_uf_max']:
        im_val = comparison.get('inner_mongolia', {}).get(key, 'N/A')
        jq_val = comparison.get('jiuquan', {}).get(key, 'N/A')
        print(f"  {key:>20s}: {str(im_val):>16s} {str(jq_val):>16s}")

    # Save all
    out = {
        'location': LOCATION,
        'dual_electrolyzer': {
            'all_results': all_dual,
            'scenario_best': {str(k): v for k, v in scenario_best.items()},
        },
        'uf_coarse': uf_results,
        'uf_fine': {k: v for k, v in fine_results.items()},
        'comparison': comparison,
    }
    out_path = os.path.join(OUT_DIR, f'jiuquan_full_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2,
                  default=lambda x: float(x) if hasattr(x, 'item') else x)

    df_uf.to_csv(os.path.join(OUT_DIR, f'uf_coarse_{LOCATION}.csv'), index=False)
    print(f"\nResults saved to {out_path}")
