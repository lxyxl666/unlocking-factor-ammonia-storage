"""Worst-case consecutive multi-day UF validation.

Finds the lowest-renewable consecutive N-day window from the 2025 NASA POWER
8760h data and runs a continuous MILP (with daily SOC coupling) to verify
whether the Unlocking Factor holds under extreme multi-day conditions.

Addresses reviewer concern: k-means typical days sever temporal continuity.
"""
import pandas as pd
import numpy as np
import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from pulp import (LpProblem, LpVariable, LpMinimize, lpSum, value,
                  PULP_CBC_CMD, LpBinary, LpContinuous)
from config import *
from dual_electrolyzer_milp import (compute_h2_breakpoints, ALKEL_H2_BREAKPOINTS,
                                     PEMEL_H2_BREAKPOINTS, N_BP)

# ================================================================
# 1. Load raw 8760h data
# ================================================================
DATA_DIR = r'C:\Users\31526\Desktop\diangong\paper_output\data_cleaned'
df_raw = pd.read_csv(os.path.join(DATA_DIR, 'nasa_power_2025_inner_mongolia.csv'))
df_raw['date'] = pd.to_datetime(df_raw['date'])
df_raw['hour_of_day'] = df_raw['hour']  # 0--23

print(f"Loaded {len(df_raw)} hourly records, {df_raw['date'].nunique()} days")

# ================================================================
# 2. Find worst consecutive N-day window
# ================================================================
# Rank each day by total renewable generation (wind_MW + pv_MW daily sum)
daily_ren = df_raw.groupby('date').agg(
    wind_daily=('wind_MW', 'sum'),
    pv_daily=('pv_MW', 'sum')
)
daily_ren['total_ren'] = daily_ren['wind_daily'] + daily_ren['pv_daily']

WINDOW_DAYS = 5  # can adjust: 3, 5, or 7

# Rolling sum: find window with lowest total renewable generation
rolling = daily_ren['total_ren'].rolling(WINDOW_DAYS).sum()
worst_end = rolling.idxmin()  # last date of worst window
all_dates = sorted(daily_ren.index)
end_idx = all_dates.index(worst_end)
start_idx = end_idx - WINDOW_DAYS + 1
worst_dates = all_dates[start_idx:end_idx + 1]

print(f"Worst {WINDOW_DAYS}-day window: {worst_dates[0].date()} to {worst_dates[-1].date()}")
print(f"Total renewable in this window: {rolling[worst_end]:.0f} MWh")
print(f"Daily breakdown:")
for d in worst_dates:
    print(f"  {d.date()}: wind={daily_ren.loc[d, 'wind_daily']:.0f} MWh, "
          f"pv={daily_ren.loc[d, 'pv_daily']:.0f} MWh")

# Extract hourly data for the worst window
mask = df_raw['date'].isin(worst_dates)
window = df_raw[mask].copy().sort_values(['date', 'hour'])
P_wind_raw = window['wind_MW'].values
P_pv_raw = window['pv_MW'].values

# Load profile (same for all days; use the standard load from data_cleaned)
# Try loading from load_actual.csv; otherwise construct 6 MW base load
load_path = os.path.join(DATA_DIR, 'load_actual.csv')
if os.path.exists(load_path):
    df_load = pd.read_csv(load_path)
    P_load_24 = df_load['load_MW'].values  # 24 hours
else:
    P_load_24 = np.full(24, 6.0)  # 6 MW default

# Repeat load profile for each day
P_load_raw = np.tile(P_load_24, WINDOW_DAYS)

T = len(P_wind_raw)
print(f"\nContinuous horizon: {T} hours ({WINDOW_DAYS} days)")

# ================================================================
# 3. Build multi-day MILP (adapted from dual_electrolyzer_milp.py)
# ================================================================
FEEDIN_PRICE = 0.3779
ALKEL_MIN_RATIO = 0.15
PEMEL_MIN_RATIO = 0.05
ALKEL_RAMP_MAX = 0.20
PEMEL_RAMP_MAX = 1.00
M_BIG = 1e6

def _tou_price_multi(t):
    """TOU price with daily wrapping for multi-day horizons."""
    h = t % 24
    if 10 <= h < 15 or 18 <= h < 21:
        return 0.8024  # peak
    elif 7 <= h < 10 or 15 <= h < 18 or 21 <= h < 23:
        return 0.6074  # standard
    else:
        return 0.3424  # valley

def build_multi_day_model(P_wind, P_pv, P_load, storage_cap_MWh,
                           maximize_production=True, daily_target_ub=72.0):
    """Build continuous multi-day MILP.

    Args:
        P_wind, P_pv, P_load: numpy arrays of length T (not necessarily 24)
        storage_cap_MWh: battery storage capacity
        maximize_production: maximize NH3 output
        daily_target_ub: upper bound on total NH3 production (tons)

    Returns:
        LpProblem instance
    """
    T_local = len(P_wind)
    prob = LpProblem("MultiDay_MILP", LpMinimize)
    K = N_BP

    # Decision variables
    r_alk = [LpVariable(f"ralk_{t}", lowBound=0, upBound=1) for t in range(T_local)]
    r_pem = [LpVariable(f"rpem_{t}", lowBound=0, upBound=1) for t in range(T_local)]
    r_nh3 = [LpVariable(f"rnh3_{t}", lowBound=0, upBound=1) for t in range(T_local)]
    y_alk = [LpVariable(f"yalk_{t}", cat=LpBinary) for t in range(T_local)]
    y_pem = [LpVariable(f"ypem_{t}", cat=LpBinary) for t in range(T_local)]
    y_nh3 = [LpVariable(f"ynh3_{t}", cat=LpBinary) for t in range(T_local)]
    s_alk = [LpVariable(f"salk_{t}", cat=LpBinary) for t in range(T_local)]
    s_pem = [LpVariable(f"spem_{t}", cat=LpBinary) for t in range(T_local)]
    s_nh3 = [LpVariable(f"snh3_{t}", cat=LpBinary) for t in range(T_local)]
    ramp_up = [LpVariable(f"rup_{t}", lowBound=0) for t in range(T_local)]
    ramp_dn = [LpVariable(f"rdn_{t}", lowBound=0) for t in range(T_local)]

    # SOS2 variables
    lam_alk = [[LpVariable(f"lam_alk_{t}_{k}", lowBound=0, upBound=1)
                 for k in range(K)] for t in range(T_local)]
    lam_pem = [[LpVariable(f"lam_pem_{t}_{k}", lowBound=0, upBound=1)
                 for k in range(K)] for t in range(T_local)]
    seg_alk = [[LpVariable(f"seg_alk_{t}_{k}", cat=LpBinary)
                 for k in range(K - 1)] for t in range(T_local)]
    seg_pem = [[LpVariable(f"seg_pem_{t}_{k}", cat=LpBinary)
                 for k in range(K - 1)] for t in range(T_local)]

    # Grid exchange (disabled for off-grid worst-case analysis)
    P_buy = [LpVariable(f"buy_{t}", lowBound=0) for t in range(T_local)]
    P_sell = [LpVariable(f"sell_{t}", lowBound=0) for t in range(T_local)]
    z_buy = [LpVariable(f"zbuy_{t}", cat=LpBinary) for t in range(T_local)]
    curtail = [LpVariable(f"curt_{t}", lowBound=0) for t in range(T_local)]

    # Storage variables
    has_storage = storage_cap_MWh > 0
    if has_storage:
        soc = [LpVariable(f"soc_{t}", lowBound=0, upBound=storage_cap_MWh)
               for t in range(T_local + 1)]
        chg = [LpVariable(f"chg_{t}", lowBound=0) for t in range(T_local)]
        dis = [LpVariable(f"dis_{t}", lowBound=0) for t in range(T_local)]

    # ---- Minimum load constraints ----
    for t in range(T_local):
        prob += r_alk[t] >= ALKEL_MIN_RATIO * y_alk[t]
        prob += r_alk[t] <= y_alk[t]
        prob += r_pem[t] >= PEMEL_MIN_RATIO * y_pem[t]
        prob += r_pem[t] <= y_pem[t]
        prob += r_nh3[t] >= NH3_MIN_RATIO * y_nh3[t]
        prob += r_nh3[t] <= y_nh3[t]

    # ---- Startup detection ----
    for t in range(T_local):
        if t == 0:
            prob += s_alk[t] >= y_alk[t] - 0  # assume off at start
            prob += s_pem[t] >= y_pem[t] - 0
            prob += s_nh3[t] >= y_nh3[t] - 0
        else:
            prob += s_alk[t] >= y_alk[t] - y_alk[t - 1]
            prob += s_pem[t] >= y_pem[t] - y_pem[t - 1]
            prob += s_nh3[t] >= y_nh3[t] - y_nh3[t - 1]

    # ---- ALKEL ramp rate limits ----
    for t in range(T_local):
        if t == 0:
            prob += r_alk[t] - 0 <= ALKEL_RAMP_MAX + M_BIG * s_alk[t] + ramp_up[t]
            prob += 0 - r_alk[t] <= ALKEL_RAMP_MAX + M_BIG * s_alk[t] + ramp_dn[t]
        else:
            prob += r_alk[t] - r_alk[t - 1] <= ALKEL_RAMP_MAX + M_BIG * s_alk[t] + ramp_up[t]
            prob += r_alk[t - 1] - r_alk[t] <= ALKEL_RAMP_MAX + M_BIG * s_alk[t] + ramp_dn[t]

    # ---- SOS2 piecewise-linear efficiency ----
    for t in range(T_local):
        r_break_alk = [bp[0] for bp in ALKEL_H2_BREAKPOINTS]
        h2_break_alk = [bp[1] for bp in ALKEL_H2_BREAKPOINTS]
        r_break_pem = [bp[0] for bp in PEMEL_H2_BREAKPOINTS]
        h2_break_pem = [bp[1] for bp in PEMEL_H2_BREAKPOINTS]

        # ALKEL SOS2: sum(lambda) == y_alk (0 when off, 1 when on)
        prob += lpSum(lam_alk[t][k] for k in range(K)) == y_alk[t]
        prob += lpSum(seg_alk[t][k] for k in range(K - 1)) == y_alk[t]
        prob += r_alk[t] == lpSum(lam_alk[t][k] * r_break_alk[k] for k in range(K))
        h2_alk_expr = lpSum(lam_alk[t][k] * h2_break_alk[k] for k in range(K))
        # SOS2 adjacency
        prob += lam_alk[t][0] <= seg_alk[t][0]
        for k in range(1, K - 1):
            prob += lam_alk[t][k] <= seg_alk[t][k - 1] + seg_alk[t][k]
        prob += lam_alk[t][K - 1] <= seg_alk[t][K - 2]

        # PEMEL SOS2: sum(lambda) == y_pem (0 when off, 1 when on)
        prob += lpSum(lam_pem[t][k] for k in range(K)) == y_pem[t]
        prob += lpSum(seg_pem[t][k] for k in range(K - 1)) == y_pem[t]
        prob += r_pem[t] == lpSum(lam_pem[t][k] * r_break_pem[k] for k in range(K))
        h2_pem_expr = lpSum(lam_pem[t][k] * h2_break_pem[k] for k in range(K))
        # SOS2 adjacency
        prob += lam_pem[t][0] <= seg_pem[t][0]
        for k in range(1, K - 1):
            prob += lam_pem[t][k] <= seg_pem[t][k - 1] + seg_pem[t][k]
        prob += lam_pem[t][K - 1] <= seg_pem[t][K - 2]

        # H2 balance: alk + pem output = NH3 requirement
        h2_needed = r_nh3[t] * AMMONIA_H2_NEED
        prob += h2_alk_expr + h2_pem_expr == h2_needed

    # ---- Power balance ----
    for t in range(T_local):
        P_eq = (r_alk[t] * ALKEL_P_RATED + r_pem[t] * PEMEL_P_RATED +
                r_nh3[t] * NH3_P_RATED)
        if has_storage:
            prob += (P_wind[t] + P_pv[t] + P_buy[t] + dis[t] ==
                     P_load[t] + P_eq + P_sell[t] + chg[t] + curtail[t])
        else:
            prob += (P_wind[t] + P_pv[t] + P_buy[t] ==
                     P_load[t] + P_eq + P_sell[t] + curtail[t])

    # ---- Grid constraints (off-grid: force P_buy = P_sell = 0) ----
    for t in range(T_local):
        prob += P_buy[t] <= M_BIG * z_buy[t]
        prob += P_sell[t] <= M_BIG * (1 - z_buy[t])
        prob += P_buy[t] == 0
        prob += P_sell[t] == 0

    # ---- Storage dynamics ----
    if has_storage:
        EFF_CHG = 0.90
        EFF_DIS = 0.90
        SELF_DISCHARGE = 0.998  # 1 - 0.002/h
        for t in range(T_local):
            prob += (soc[t + 1] ==
                     soc[t] * SELF_DISCHARGE + chg[t] * EFF_CHG - dis[t] / EFF_DIS)
        # Cyclic constraint: SOC returns to initial at end of window
        prob += soc[T_local] == soc[0]
        prob += soc[0] == storage_cap_MWh * 0.5  # start at 50%

    # ---- NH3 production ----
    total_nh3 = lpSum(r_nh3[t] * NH3_RATE for t in range(T_local))

    # ---- Objective ----
    if maximize_production:
        # Maximize NH3 output (minimize negative)
        startup_cost = lpSum(s_alk[t] * ALKEL_STARTUP_COST +
                             s_pem[t] * PEMEL_STARTUP_COST +
                             s_nh3[t] * NH3_STARTUP_COST for t in range(T_local))
        ramp_penalty = lpSum((ramp_up[t] + ramp_dn[t]) * 200 for t in range(T_local))
        om_cost = lpSum((r_alk[t] * ALKEL_P_RATED * ALKEL_OM +
                         r_pem[t] * PEMEL_P_RATED * PEMEL_OM) for t in range(T_local))
        prob += -total_nh3 * NH3_REF_VALUE + startup_cost + ramp_penalty + om_cost
    else:
        # Fixed target: daily_target_ub is t/d, convert to total over window
        prob += total_nh3 == daily_target_ub * WINDOW_DAYS
        # Minimize operating cost
        startup_cost = lpSum(s_alk[t] * ALKEL_STARTUP_COST +
                             s_pem[t] * PEMEL_STARTUP_COST +
                             s_nh3[t] * NH3_STARTUP_COST for t in range(T_local))
        ramp_penalty = lpSum((ramp_up[t] + ramp_dn[t]) * 200 for t in range(T_local))
        om_cost = lpSum((r_alk[t] * ALKEL_P_RATED * ALKEL_OM +
                         r_pem[t] * PEMEL_P_RATED * PEMEL_OM) for t in range(T_local))
        prob += startup_cost + ramp_penalty + om_cost

    return prob, total_nh3


def solve_multi_day_offgrid(P_wind, P_pv, P_load, storage_cap_MWh, daily_target):
    """Solve multi-day MILP in OFF-GRID mode with fixed NH3 target.
    Returns (feasible, metrics_dict)."""
    prob, total_nh3_expr = build_multi_day_model(
        P_wind, P_pv, P_load, storage_cap_MWh,
        maximize_production=False, daily_target_ub=daily_target
    )
    solver = PULP_CBC_CMD(msg=False, timeLimit=300)
    status = prob.solve(solver)

    if status != 1:
        return False, None

    T_local = len(P_wind)
    total_ren_available = P_wind.sum() + P_pv.sum()
    curtail_total = sum(value(prob.variablesDict()[f"curt_{t}"])
                        for t in range(T_local))
    ren_used = total_ren_available - curtail_total

    grid_import_total = sum(value(prob.variablesDict()[f"buy_{t}"])
                            for t in range(T_local))

    return True, {
        'total_nh3_t': daily_target * WINDOW_DAYS,
        'daily_nh3_t': daily_target,
        'ren_available_MWh': total_ren_available,
        'ren_used_MWh': ren_used,
        'curtailment_MWh': curtail_total,
        'grid_import_MWh': grid_import_total,
        'storage_MWh': storage_cap_MWh,
    }


# ================================================================
# 4. Run worst-case analysis — off-grid feasibility sweep
# ================================================================
print("\n" + "=" * 60)
print("Worst-case multi-day OFF-GRID feasibility analysis")
print("=" * 60)

storage_levels = [0, 1, 2, 3, 5, 10, 20, 50, 100, 200]
nh3_targets = [0, 5, 10, 15, 20, 30, 45, 63, 72]

# Build feasibility matrix: storage x NH3 target
print(f"\nFeasibility matrix (off-grid, {WINDOW_DAYS}-day worst window):")
print(f"{'Storage':>8s}", end="")
for tgt in nh3_targets:
    print(f"{tgt:>6d}", end="")
print(" t/d")
print("-" * (8 + 6 * len(nh3_targets)))

results = []
for s_mwh in storage_levels:
    print(f"{s_mwh:>6d} MWh", end=" ")
    for tgt in nh3_targets:
        feasible, res = solve_multi_day_offgrid(
            P_wind_raw, P_pv_raw, P_load_raw, s_mwh, tgt
        )
        if feasible:
            print(f"{'OK':>6s}", end="")
            results.append(res)
        else:
            print(f"{'--':>6s}", end="")
    print()

# ================================================================
# 5. Summarize findings
# ================================================================
print("\n" + "=" * 60)
print("Summary for manuscript:")
print("=" * 60)

# Find max NH3 at each storage level
print("\nMax off-grid NH3 production in worst 5-day window:")
for s_mwh in storage_levels:
    max_nh3 = 0
    for r in results:
        if r['storage_MWh'] == s_mwh and r['daily_nh3_t'] > max_nh3:
            max_nh3 = r['daily_nh3_t']
    if max_nh3 > 0:
        # Find ren_used for this case
        for r in results:
            if r['storage_MWh'] == s_mwh and r['daily_nh3_t'] == max_nh3:
                print(f"  {s_mwh:>6d} MWh: {max_nh3:.0f} t/d NH3, "
                      f"ren_used={r['ren_used_MWh']:.0f} MWh, "
                      f"curtail={r['curtailment_MWh']:.0f} MWh")
                break
    else:
        print(f"  {s_mwh:>6d} MWh: INFEASIBLE (cannot even meet base load)")

# Print comparison with typical-day results
print("\nComparison with typical-day results:")
print("  Typical-day S17 (best UF): UF_peak = 9.27 at 1 MWh, off-grid NH3 = 72 t/d")
print("  Typical-day S18:           UF_peak = 8.01 at 1 MWh, off-grid NH3 = 72 t/d")
print(f"  Worst-case {WINDOW_DAYS}-day window:   see above — extreme low-renewable period")

# Save results
out_path = os.path.join(os.path.dirname(__file__), '..', '..', 'results',
                         'journal', 'worst_case_window_results.json')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w') as f:
    json.dump({
        'window_days': WINDOW_DAYS,
        'window_start': str(worst_dates[0].date()),
        'window_end': str(worst_dates[-1].date()),
        'results': results
    }, f, indent=2, default=str)

print(f"\nResults saved to: {out_path}")
print("Done.")
