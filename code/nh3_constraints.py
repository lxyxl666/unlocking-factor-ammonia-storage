"""NH3 synthesis constraint sensitivity analysis.

Compares baseline NH3 modeling (5% min load, no startup cost)
against realistic Haber-Bosch constraints (40% min load, 2000 yuan/start)
for key scenarios S1, S17, S18 in both grid-connected and off-grid modes.
"""
import pandas as pd
import numpy as np
import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from config import *
from dual_electrolyzer_milp import solve_dual_electrolyzer

DATA_DIR_ABS = DATA_DIR
OUT_DIR_ABS = OUT_DIR

# Load scenario data
df = pd.read_csv(os.path.join(DATA_DIR_ABS, SCENARIOS_FILE))

# Two NH3 parameter configurations
BASELINE = {'nh3_min': 0.05, 'nh3_startup_cost': 0.0}
CONSTRAINED = {'nh3_min': NH3_MIN_RATIO, 'nh3_startup_cost': NH3_STARTUP_COST}

# Scenarios to test
SCENARIOS = [1, 17, 18]

# Modes: (target_tpd, grid_connected, max_production)
MODES = [
    ('Grid 36 t/d', 36, True, False),
    ('Off-grid max', 72, False, True),
]

print("=" * 70)
print("NH3 Synthesis Constraint Sensitivity Analysis")
print("=" * 70)

results = []

for scenario_id in SCENARIOS:
    df_s = df[df['scenario_id'] == scenario_id].sort_values('hour')
    P_wind = df_s['wind_MW'].values
    P_pv = df_s['solar_MW'].values
    P_load = df_s['load_MW'].values

    for mode_name, target, grid_on, max_prod in MODES:
        for label, nh3_cfg in [('Baseline', BASELINE), ('Constrained', CONSTRAINED)]:
            r = solve_dual_electrolyzer(
                P_wind, P_pv, P_load, target,
                grid_connected=grid_on,
                storage_cap_MWh=0,
                maximize_production=max_prod,
                nh3_params=nh3_cfg,
                time_limit_sec=60,
            )
            if r is None:
                print(f"  S{scenario_id} {mode_name} {label}: INFEASIBLE")
                results.append({
                    'scenario': scenario_id, 'mode': mode_name,
                    'nh3_config': label, 'feasible': False,
                })
            else:
                print(f"  S{scenario_id} {mode_name} {label}: "
                      f"NH3={r['daily_nh3']:.1f} t/d, "
                      f"ton_cost={r['ton_cost']:.1f} yuan/t, "
                      f"NH3 starts={r['nh3_startups']}, "
                      f"util={r['equipment_utilization']:.3f}")
                results.append({
                    'scenario': scenario_id, 'mode': mode_name,
                    'nh3_config': label, 'feasible': True,
                    'daily_nh3': r['daily_nh3'],
                    'ton_cost': r['ton_cost'],
                    'nh3_startups': r['nh3_startups'],
                    'equipment_utilization': r['equipment_utilization'],
                    'alkel_energy_share': r['alkel_energy_share'],
                })

# Summary
print("\n" + "=" * 70)
print("Summary for manuscript:")
print("=" * 70)

for mode_name, _, _, _ in MODES:
    print(f"\n{mode_name}:")
    for scenario_id in SCENARIOS:
        base = [r for r in results if r['scenario'] == scenario_id
                and r['mode'] == mode_name and r['nh3_config'] == 'Baseline'
                and r['feasible']]
        const = [r for r in results if r['scenario'] == scenario_id
                 and r['mode'] == mode_name and r['nh3_config'] == 'Constrained'
                 and r['feasible']]
        if base and const:
            b = base[0]; c = const[0]
            delta_nh3 = c['daily_nh3'] - b['daily_nh3']
            delta_cost = c['ton_cost'] - b['ton_cost']
            print(f"  S{scenario_id}: NH3 {b['daily_nh3']:.1f} -> {c['daily_nh3']:.1f} t/d "
                  f"({delta_nh3:+.1f}), ton_cost {b['ton_cost']:.0f} -> {c['ton_cost']:.0f} "
                  f"({delta_cost:+.0f}), NH3 starts {b['nh3_startups']} -> {c['nh3_startups']}")

# Save results
out_path = os.path.join(OUT_DIR_ABS, 'nh3_constraints_results.json')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to: {out_path}")
print("Done.")
