"""Fine grid (1 MWh step) for top UF scenarios with real NASA data."""
import pandas as pd, os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
from config import *
from storage_unlocking import compute_unlocking_factor

df = pd.read_csv(os.path.join(DATA_DIR, SCENARIOS_FILE))

for sid in [1, 17, 18, 9, 20]:
    df_s = df[df['scenario_id'] == sid].sort_values('hour')
    data = {
        'P_wind': df_s['wind_MW'].values,
        'P_pv': df_s['solar_MW'].values,
        'P_load': df_s['load_MW'].values,
    }

    base = compute_unlocking_factor(data, 0)
    if base is None:
        print(f"S{sid}: baseline infeasible, skipping")
        continue
    base_ren_used = base['ren_used_MWh']
    base_nh3 = base['daily_nh3']
    print(f"S{sid} baseline: NH3={base_nh3:.2f}t, "
          f"ren_used={base_ren_used:.1f}MWh")

    results = []
    t_start = time.time()
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
        if cap % 20 == 0:
            elapsed = time.time() - t_start
            print(f"  {cap}MWh done ({len(results)} results, {elapsed:.1f}s)")

    # Save
    out = os.path.join(OUT_DIR, f'S{sid}_fine_grid.json')
    json.dump(results, open(out, 'w', encoding='utf-8'), indent=2,
              default=lambda x: float(x) if hasattr(x, 'item') else x)

    # Best UF and best cost
    best_uf = max(results, key=lambda r: r.get('unlocking_factor', 0))
    best_cost = min(results, key=lambda r: r['ton_cost'])
    print(f"  Best UF: {best_uf['unlocking_factor']:.4f} at "
          f"{best_uf['storage_MWh']}MWh, NH3={best_uf['daily_nh3']:.2f}t")
    print(f"  Best cost: {best_cost['ton_cost']:.2f} at "
          f"{best_cost['storage_MWh']}MWh\n")

print("Done!")
