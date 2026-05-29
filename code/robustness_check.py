"""Robustness check: ±15% wind/solar perturbation for key UF scenarios.

Tests whether UF peak is stable under renewable resource uncertainty.
Scenarios: S1 (high wind), S17 (mid wind/high solar), S18 (mid wind/mid solar).
"""
import pandas as pd
import numpy as np
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from config import *
from storage_unlocking import compute_unlocking_factor


def run_perturbed_uf(scenario_data, perturbation_wind, perturbation_solar, label):
    """Run fine grid UF with perturbed wind/solar data."""
    Pw = scenario_data['P_wind'] * (1 + perturbation_wind)
    Ps = scenario_data['P_pv'] * (1 + perturbation_solar)
    Pl = scenario_data['P_load']

    data = {'P_wind': Pw, 'P_pv': Ps, 'P_load': Pl}

    base = compute_unlocking_factor(data, 0)
    if base is None:
        return None, f"Infeasible at baseline"

    base_ren_used = base['ren_used_MWh']
    base_nh3 = base['daily_nh3']

    results = []
    for cap in range(0, 101, 1):
        res = compute_unlocking_factor(data, cap)
        if res:
            delta_ren = res['ren_used_MWh'] - base_ren_used
            uf = delta_ren / cap if cap > 0 else 0.0
            delta_nh3 = res['daily_nh3'] - base_nh3
            res['delta_ren_used_MWh'] = round(delta_ren, 4)
            res['unlocking_factor'] = round(uf, 4)
            res['delta_nh3_tons'] = round(delta_nh3, 4)
            results.append(res)

    if not results or len(results) < 2:
        return None, "No feasible results"

    best_uf = max((r for r in results if r.get('unlocking_factor', 0) > 0),
                  key=lambda r: r.get('unlocking_factor', 0), default=None)
    best_cost = min(results, key=lambda r: r['ton_cost'])

    return {
        'label': label,
        'perturbation_wind': perturbation_wind,
        'perturbation_solar': perturbation_solar,
        'base_nh3': round(base_nh3, 2),
        'base_ren_used': round(base_ren_used, 1),
        'peak_uf': round(best_uf['unlocking_factor'], 2) if best_uf else 0,
        'peak_uf_at_MWh': int(best_uf['storage_MWh']) if best_uf else 0,
        'peak_uf_nh3': round(best_uf['daily_nh3'], 2) if best_uf else 0,
        'best_cost': round(best_cost['ton_cost'], 2),
        'best_cost_at_MWh': int(best_cost['storage_MWh']),
        'n_results': len(results),
    }, None


if __name__ == '__main__':
    print("=" * 60)
    print("ROBUSTNESS CHECK: +/-15% Wind/Solar Perturbation")
    print("=" * 60)

    df = pd.read_csv(os.path.join(DATA_DIR, SCENARIOS_FILE))

    # Perturbation levels
    perturbations = [
        (-0.15, 0.00, "Wind -15%"),
        (+0.15, 0.00, "Wind +15%"),
        (0.00, -0.15, "Solar -15%"),
        (0.00, +0.15, "Solar +15%"),
        (-0.15, -0.15, "Both -15%"),
        (+0.15, +0.15, "Both +15%"),
    ]

    all_results = {}
    for sid in [1, 17, 18]:
        print(f"\n{'='*40}")
        df_s = df[df['scenario_id'] == sid].sort_values('hour')
        scenario_data = {
            'P_wind': df_s['wind_MW'].values,
            'P_pv': df_s['solar_MW'].values,
            'P_load': df_s['load_MW'].values,
        }

        # Baseline (unperturbed)
        base_result, _ = run_perturbed_uf(scenario_data, 0.0, 0.0, "Base")
        print(f"S{sid} Base: UF={base_result['peak_uf']} @ "
              f"{base_result['peak_uf_at_MWh']}MWh, "
              f"NH3={base_result['base_nh3']}t -> "
              f"{base_result['peak_uf_nh3']}t")

        sid_results = [base_result]

        for pw, ps, label in perturbations:
            t0 = time.time()
            result, error = run_perturbed_uf(scenario_data, pw, ps, label)
            elapsed = time.time() - t0
            if result:
                sid_results.append(result)
                print(f"  {label}: UF={result['peak_uf']} @ "
                      f"{result['peak_uf_at_MWh']}MWh, "
                      f"NH3={result['base_nh3']}t -> {result['peak_uf_nh3']}t "
                      f"({elapsed:.0f}s)")
            else:
                print(f"  {label}: FAILED - {error}")

        all_results[f"S{sid}"] = sid_results

    # Summary table
    print(f"\n\n{'='*80}")
    print("ROBUSTNESS SUMMARY")
    print(f"{'='*80}")
    for sid_key in ['S1', 'S17', 'S18']:
        results = all_results[sid_key]
        print(f"\n{sid_key}:")
        print(f"  {'Perturbation':<20s} {'Base NH3':>8s} {'Peak UF':>8s} "
              f"{'@MWh':>6s} {'Peak NH3':>8s} {'dUF'}")
        print(f"  {'-'*60}")
        base_uf = results[0]['peak_uf']
        for r in results:
            duf = r['peak_uf'] - base_uf
            print(f"  {r['label']:<20s} {r['base_nh3']:>8.2f} {r['peak_uf']:>8.2f} "
                  f"{r['peak_uf_at_MWh']:>6d} {r['peak_uf_nh3']:>8.2f} "
                  f"{duf:>+5.2f}")

        # UF range across perturbations
        ufs = [r['peak_uf'] for r in results]
        print(f"  UF range: [{min(ufs):.2f}, {max(ufs):.2f}], "
              f"base={base_uf:.2f}, CV={np.std(ufs)/np.mean(ufs):.3f}")

    # Save
    out_path = os.path.join(OUT_DIR, 'robustness_check.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\nResults saved to {out_path}")
