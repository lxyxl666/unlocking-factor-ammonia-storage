"""Battery cycle-depth-dependent degradation analysis.

Post-processes SOC profiles from off-grid MILP dispatch using rainflow
cycle counting and a DoD-dependent cycle-life curve to compare:
  (a) Linear 15-year amortization (current model)
  (b) Cycle-based effective lifetime + annualized cost

Quantifies whether the simple linear model meaningfully distorts the UF
conclusion for small, deeply-cycled storage systems.
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

# ================================================================
# 1. Rainflow cycle counting (ASTM E1049-85 4-point algorithm)
# ================================================================
def rainflow_cycles(soc_profile):
    """Extract full and half cycles from an SOC time series.

    Args:
        soc_profile: list of SOC values, length T+1, with soc[0] == soc[-1]
                     (periodic boundary condition satisfied).

    Returns:
        list of (depth_of_discharge_pu, is_full_cycle) tuples.
        DoD in range (0, 1] where 1.0 = full discharge from max SOC.
    """
    if len(soc_profile) < 3:
        return []

    # Extract turning points (local extrema)
    tp = [soc_profile[0]]
    for i in range(1, len(soc_profile) - 1):
        prev, curr, next_v = soc_profile[i - 1], soc_profile[i], soc_profile[i + 1]
        if (curr - prev) * (next_v - curr) < 0:  # sign change = turning point
            tp.append(curr)
    tp.append(soc_profile[-1])

    if len(tp) < 3:
        return []

    # Normalize to max SOC for DoD calculation
    max_soc = max(tp)
    if max_soc <= 0:
        return []
    tp_norm = [s / max_soc for s in tp]

    cycles = []
    stack = list(tp_norm)

    # 4-point rainflow: process ranges A-B-C-D
    i = 0
    while len(stack) >= 4 and i < len(stack) - 3:
        a, b, c, d = stack[i], stack[i + 1], stack[i + 2], stack[i + 3]
        range_ab = abs(b - a)
        range_bc = abs(c - b)
        range_cd = abs(d - c)

        if range_bc <= range_ab and range_bc <= range_cd:
            # B-C is a full cycle
            dod = range_bc
            if dod > 0.001:
                cycles.append((dod, True))
            # Remove B and C from stack
            stack.pop(i + 1)  # remove B
            stack.pop(i + 1)  # remove C
            i = max(0, i - 1)
        else:
            i += 1

    # Remaining points are half cycles
    for j in range(len(stack) - 1):
        dod = abs(stack[j + 1] - stack[j])
        if dod > 0.001:
            cycles.append((dod, False))

    return cycles


# ================================================================
# 2. DoD-dependent cycle life model
# ================================================================
def cycle_life_at_dod(dod, n_ref=5000, dod_ref=0.80, k=0.8):
    """Lithium-ion cycle life as function of depth-of-discharge.

    N_cycles(dod) = N_ref * (dod_ref / dod)^k
    N_ref = 5000 cycles at dod_ref = 0.80
    k = 0.8 (typical LFP exponent; Smith et al. 2017, Xu et al. 2020)

    Returns:
        Number of equivalent full cycles to end-of-life at given DoD.
    """
    dod_clipped = max(dod, 0.001)
    return n_ref * (dod_ref / dod_clipped) ** k


def compute_cycle_based_lifetime(soc_profile, n_ref=5000, dod_ref=0.80, k=0.8):
    """Compute effective battery lifetime from SOC profile using rainflow.

    Returns:
        (effective_lifetime_years, daily_damage, cycle_summary)
    """
    cycles = rainflow_cycles(soc_profile)
    if not cycles:
        return 15.0, 0.0, {'n_full': 0, 'n_half': 0, 'mean_dod': 0}

    daily_damage = 0.0
    for dod, is_full in cycles:
        n_life = cycle_life_at_dod(dod, n_ref, dod_ref, k)
        weight = 1.0 if is_full else 0.5
        daily_damage += weight / n_life

    if daily_damage <= 0:
        return 15.0, 0.0, {'n_full': 0, 'n_half': 0, 'mean_dod': 0}

    eff_life_days = 1.0 / daily_damage
    eff_life_years = eff_life_days / 365.0

    n_full = sum(1 for _, is_full in cycles if is_full)
    n_half = sum(1 for _, is_full in cycles if not is_full)
    mean_dod = np.mean([d for d, _ in cycles]) if cycles else 0

    return eff_life_years, daily_damage, {
        'n_full': n_full, 'n_half': n_half,
        'mean_dod': round(mean_dod, 4),
    }


# ================================================================
# 3. Run analysis
# ================================================================
print("=" * 70)
print("Battery Cycle-Depth Degradation Analysis")
print("=" * 70)

df = pd.read_csv(os.path.join(DATA_DIR_ABS, SCENARIOS_FILE))
SCENARIOS = [1, 17, 18]
STORAGE_LEVELS = [1, 5, 10, 20, 50]

results = []

for scenario_id in SCENARIOS:
    df_s = df[df['scenario_id'] == scenario_id].sort_values('hour')
    P_wind = df_s['wind_MW'].values
    P_pv = df_s['solar_MW'].values
    P_load = df_s['load_MW'].values

    for s_mwh in STORAGE_LEVELS:
        r = solve_dual_electrolyzer(
            P_wind, P_pv, P_load, 72,
            grid_connected=False,
            storage_cap_MWh=s_mwh,
            maximize_production=True,
            save_soc=True,
            time_limit_sec=60,
        )

        if r is None or r['soc_profile'] is None:
            print(f"  S{scenario_id} {s_mwh}MWh: NO SOC DATA")
            results.append({
                'scenario': scenario_id, 'storage_MWh': s_mwh,
                'feasible': False,
            })
            continue

        soc = r['soc_profile']
        eff_life, damage, cyc_info = compute_cycle_based_lifetime(soc)

        # Linear cost: 1000 yuan/kWh / (15 * 365) days
        daily_linear = s_mwh * 1000 * STORAGE_INVESTMENT / STORAGE_LIFE_DAYS
        # Cycle-based cost: 1000 yuan/kWh / (eff_life * 365) days
        daily_cycle = (s_mwh * 1000 * STORAGE_INVESTMENT /
                       (eff_life * 365) if eff_life > 0 else daily_linear * 10)

        ton_cost_linear = r['ton_cost']
        cost_delta = daily_cycle - daily_linear
        ton_cost_cycle = ton_cost_linear + cost_delta / max(r['daily_nh3'], 0.01)

        print(f"  S{scenario_id} {s_mwh:2d}MWh: linear={15:.0f}yr -> "
              f"cycle={eff_life:.1f}yr, "
              f"cycles={cyc_info['n_full']}F+{cyc_info['n_half']}H, "
              f"DoD_avg={cyc_info['mean_dod']:.2f}, "
              f"ton_cost {ton_cost_linear:.0f} -> {ton_cost_cycle:.0f} yuan/t")

        results.append({
            'scenario': scenario_id, 'storage_MWh': s_mwh,
            'feasible': True,
            'daily_nh3': r['daily_nh3'],
            'soc_profile': [round(v, 4) for v in soc],
            'linear_lifetime_yr': 15,
            'cycle_lifetime_yr': round(eff_life, 2),
            'cycle_lifetime_increase_pct': round((eff_life / 15 - 1) * 100, 1),
            'n_full_cycles': cyc_info['n_full'],
            'n_half_cycles': cyc_info['n_half'],
            'mean_dod': cyc_info['mean_dod'],
            'ton_cost_linear': round(ton_cost_linear, 2),
            'ton_cost_cycle': round(ton_cost_cycle, 2),
            'daily_storage_linear': round(daily_linear, 2),
            'daily_storage_cycle': round(daily_cycle, 2),
        })

# Summary
print("\n" + "=" * 70)
print("Summary for manuscript:")
print("=" * 70)
for scenario_id in SCENARIOS:
    print(f"\nS{scenario_id}:")
    for r in [x for x in results if x['scenario'] == scenario_id and x['feasible']]:
        print(f"  {r['storage_MWh']:2d} MWh: lifetime {r['cycle_lifetime_yr']:.1f} yr "
              f"({r['cycle_lifetime_increase_pct']:+.0f}% vs linear), "
              f"ton_cost {r['ton_cost_linear']:.0f} -> {r['ton_cost_cycle']:.0f} "
              f"(+{(r['ton_cost_cycle']/r['ton_cost_linear']-1)*100:.0f}%)")

# Save results
out_path = os.path.join(OUT_DIR_ABS, 'battery_degradation_results.json')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to: {out_path}")
print("Done.")
