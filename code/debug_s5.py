"""Debug S5 infeasibility in off-grid mode."""
import pandas as pd, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from config import *
from dual_electrolyzer_milp import build_dual_electrolyzer_model
from pulp import PULP_CBC_CMD

df = pd.read_csv(os.path.join(DATA_DIR, SCENARIOS_FILE))

for sid in [5, 6, 7, 8]:
    df_s = df[df['scenario_id'] == sid].sort_values('hour')
    Pw = df_s['wind_MW'].values
    Ps = df_s['solar_MW'].values
    Pl = df_s['load_MW'].values

    total_ren = sum(Pw + Ps)
    total_load = sum(Pl)
    min_wind = min(Pw)
    min_solar = min(Ps)

    # Check: can we even cover load + min ALKEL (15% of 20MW = 3MW) + min PEMEL (5% of 20MW = 1MW)?
    min_equip = ALKEL_P_RATED * ALKEL_MIN_RATIO + PEMEL_P_RATED * PEMEL_MIN_RATIO + NH3_P_RATED * PEMEL_MIN_RATIO
    max_hourly_need = max(Pl) + ALKEL_P_RATED + PEMEL_P_RATED + NH3_P_RATED  # at full load
    min_hourly_need = max(Pl) + min_equip  # at min load

    print(f"\nS{sid}: wind={df_s['wind_scenario'].iloc[0]}, solar={df_s['solar_scenario'].iloc[0]}")
    print(f"  Total ren={total_ren:.1f} MWh, Total load+min_equip*24={total_load+min_equip*24:.1f} MWh")
    print(f"  Min wind={min_wind:.2f} MW, Max hourly need={max_hourly_need:.1f} MW")
    print(f"  Hours where ren < load+min_equip: {sum(1 for t in range(24) if Pw[t]+Ps[t] < Pl[t]+min_equip)}")

    # Try with objective=min(-NH3) but with production_max
    for daily_target in [72, 36, 10, 5]:
        prob = build_dual_electrolyzer_model(Pw, Ps, Pl, daily_target,
                                             grid_connected=False,
                                             storage_cap_MWh=0,
                                             maximize_production=True)
        prob.solve(PULP_CBC_CMD(msg=False, timeLimit=10))
        status_str = {1: 'Optimal', 0: 'NotSolved', -1: 'Infeasible', -2: 'Unbounded', -3: 'Undefined'}
        print(f"  target<={daily_target}: {status_str.get(prob.status, prob.status)}")
