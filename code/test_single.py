"""Test dual-electrolyzer MILP on single scenario."""
import pandas as pd, os, sys, time

sys.path.insert(0, os.path.dirname(__file__))
from config import *
from dual_electrolyzer_milp import solve_dual_electrolyzer

df = pd.read_csv(os.path.join(DATA_DIR, SCENARIOS_FILE))
df_s = df[df['scenario_id'] == 1].sort_values('hour')
Pw = df_s['wind_MW'].values
Ps = df_s['solar_MW'].values
Pl = df_s['load_MW'].values

print(f"Testing S1 (high wind, high solar) at 36 t/d...")
t0 = time.time()
res = solve_dual_electrolyzer(Pw, Ps, Pl, 36, grid_connected=True, time_limit_sec=60)
t1 = time.time()

if res:
    print(f"  Solve time: {t1 - t0:.2f}s")
    print(f"  Ton cost: {res['ton_cost']} yuan/t")
    print(f"  Daily NH3: {res['daily_nh3']} t")
    print(f"  Self use: {res['self_use_ratio']:.4f}")
    print(f"  Green ratio: {res['green_ratio']:.4f}")
    print(f"  Feedin ratio: {res['feedin_ratio']:.4f}")
    print(f"  All pass: {res['all_pass']}")
    print(f"  ALKEL startups: {res['alkel_startups']}")
    print(f"  PEMEL startups: {res['pemel_startups']}")
    print(f"  ALKEL energy share: {res['alkel_energy_share']:.4f}")
    print(f"  Avg ALKEL load (when on): {res['avg_alkel_load']:.4f}")
    print(f"  Avg PEMEL load (when on): {res['avg_pemel_load']:.4f}")
    print(f"  Equipment utilization: {res['equipment_utilization']:.4f}")
    print(f"  Cost breakdown: {res['cost_breakdown']}")
    print(f"\n  ALKEL hourly: {res['alkel_ratio']}")
    print(f"  PEMEL hourly: {res['pemel_ratio']}")
    print(f"  NH3 hourly:  {res['nh3_ratio']}")
else:
    print("  INFEASIBLE")
