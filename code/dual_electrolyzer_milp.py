"""Dual-Electrolyzer Decoupled MILP Model (Plan A).

Models ALKEL and PEMEL as independently dispatchable assets with:
- Differentiated minimum loads (ALKEL 15%, PEMEL 5%)
- Ramp rate constraints (ALKEL 20%/h, PEMEL unconstrained)
- Startup costs (ALKEL 500 yuan/start, PEMEL 50 yuan/start)
- Piecewise-linear part-load efficiency curves (SOS2 via binary formulation)
"""
import pandas as pd
import numpy as np
import json
import os
from pulp import (LpProblem, LpVariable, LpMinimize, lpSum, value,
                  PULP_CBC_CMD, LpBinary, LpContinuous)

from config import *


# ================================================================
# Helper: efficiency piecewise linear breakpoints → H2 output values
# ================================================================
def compute_h2_breakpoints(rated_h2, eff_breakpoints):
    """Convert (load_pu, eff_pu) breakpoints to (load_pu, h2_kg_per_h) breakpoints.

    H2 at breakpoint k = load_pu * rated_h2 * eff_pu_at_k
    """
    return [(r, r * rated_h2 * eff) for r, eff in eff_breakpoints]


ALKEL_H2_BREAKPOINTS = compute_h2_breakpoints(280, ALKEL_EFF_BREAKPOINTS)
# [(0.15, 25.2), (0.40, 91.84), (0.70, 184.24), (1.00, 280.0)]

PEMEL_H2_BREAKPOINTS = compute_h2_breakpoints(320, PEMEL_EFF_BREAKPOINTS)
# [(0.05, 14.4), (0.30, 92.16), (0.60, 190.08), (1.00, 320.0)]

N_BP = 4  # number of breakpoints


# ================================================================
# Dual-Electrolyzer MILP
# ================================================================
def build_dual_electrolyzer_model(P_wind, P_pv, P_load, daily_target,
                                  grid_connected=True, storage_cap_MWh=0,
                                  maximize_production=False,
                                  price_params=None,
                                  min_load_params=None,
                                  nh3_params=None):
    """Build MILP for dual-electrolyzer dispatch.

    Args:
        P_wind, P_pv, P_load: 24-hour numpy arrays (MW)
        daily_target: daily ammonia production target (tons).
                      If maximize_production=True, this is the upper bound.
        grid_connected: if True, can buy/sell from grid
        storage_cap_MWh: battery storage capacity (0 = no storage)
        maximize_production: if True, maximize NH3 output instead of
                             meeting a fixed target (used for off-grid)
        price_params: dict with optional 'tou_peak', 'tou_standard', 'tou_valley',
                      'feedin_price', 'nh3_ref_value' overrides
        min_load_params: dict with optional 'alkel_min', 'pemel_min' overrides
        nh3_params: dict with optional 'nh3_min', 'nh3_startup_cost' overrides

    Returns:
        LpProblem instance (unsolved), or None on failure
    """
    # --- Resolve parameter overrides ---
    if price_params is None:
        price_params = {}
    if min_load_params is None:
        min_load_params = {}

    _feedin_price = price_params.get('feedin_price', FEEDIN_PRICE)
    _nh3_ref_value = price_params.get('nh3_ref_value', NH3_REF_VALUE)
    _tou_peak = price_params.get('tou_peak', 0.8024)
    _tou_standard = price_params.get('tou_standard', 0.6074)
    _tou_valley = price_params.get('tou_valley', 0.3424)
    _alkel_min = min_load_params.get('alkel_min', ALKEL_MIN_RATIO)
    _pemel_min = min_load_params.get('pemel_min', PEMEL_MIN_RATIO)

    if nh3_params is None:
        nh3_params = {}
    _nh3_min = nh3_params.get('nh3_min', PEMEL_MIN_RATIO)  # default: legacy (5%)
    _nh3_startup_cost = nh3_params.get('nh3_startup_cost', 0.0)

    def _tou_price(h):
        if 10 <= h < 15 or 18 <= h < 21:
            return _tou_peak
        elif 7 <= h < 10 or 15 <= h < 18 or 21 <= h < 23:
            return _tou_standard
        else:
            return _tou_valley

    prob = LpProblem("DualElectrolyzer_MILP", LpMinimize)

    T = 24
    M = M_BIG
    K = N_BP  # local alias

    # --- Decision Variables ---

    # Continuous power ratios
    r_alk = [LpVariable(f"ra_{t}", lowBound=0, upBound=1) for t in range(T)]
    r_pem = [LpVariable(f"rp_{t}", lowBound=0, upBound=1) for t in range(T)]
    r_nh3 = [LpVariable(f"rn_{t}", lowBound=0, upBound=1) for t in range(T)]

    # On/off binaries (independent per electrolyzer)
    y_alk = [LpVariable(f"ya_{t}", cat=LpBinary) for t in range(T)]
    y_pem = [LpVariable(f"yp_{t}", cat=LpBinary) for t in range(T)]
    y_nh3 = [LpVariable(f"yn_{t}", cat=LpBinary) for t in range(T)]

    # Startup indicators (1 if unit starts at t)
    s_alk = [LpVariable(f"sa_{t}", cat=LpBinary) for t in range(T)]
    s_pem = [LpVariable(f"sp_{t}", cat=LpBinary) for t in range(T)]
    s_nh3 = [LpVariable(f"sn_{t}", cat=LpBinary) for t in range(T)]

    # Ramp violation slack (ALKEL only)
    ramp_up = [LpVariable(f"rup_{t}", lowBound=0) for t in range(T)]
    ramp_dn = [LpVariable(f"rdn_{t}", lowBound=0) for t in range(T)]

    # Grid exchange
    P_buy = [LpVariable(f"b_{t}", lowBound=0) for t in range(T)]
    P_sell = [LpVariable(f"s_{t}", lowBound=0) for t in range(T)]
    z_buy = [LpVariable(f"zb_{t}", cat=LpBinary) for t in range(T)]

    # Storage (if applicable)
    curtail = [LpVariable(f"c_{t}", lowBound=0) for t in range(T)]

    if storage_cap_MWh > 0:
        soc = [LpVariable(f"soc_{t}", lowBound=0, upBound=storage_cap_MWh)
               for t in range(T + 1)]
        chg = [LpVariable(f"chg_{t}", lowBound=0) for t in range(T)]
        dis = [LpVariable(f"dis_{t}", lowBound=0) for t in range(T)]

    # --- SOS2 Efficiency: λ weights and segment binaries ---
    # For ALKEL
    lam_alk = [[LpVariable(f"la_{t}_{k}", lowBound=0, upBound=1)
                for k in range(K)] for t in range(T)]
    seg_alk = [[LpVariable(f"sa_{t}_{k}", cat=LpBinary)
                for k in range(K - 1)] for t in range(T)]

    # For PEMEL
    lam_pem = [[LpVariable(f"lp_{t}_{k}", lowBound=0, upBound=1)
                for k in range(K)] for t in range(T)]
    seg_pem = [[LpVariable(f"sp_{t}_{k}", cat=LpBinary)
                for k in range(K - 1)] for t in range(T)]

    # --- Constraints ---

    # SOS2: sum of lambdas = 1 when on, = 0 when off
    for t in range(T):
        # ALKEL
        prob += lpSum(lam_alk[t][k] for k in range(K)) == y_alk[t]
        # SOS2 adjacency: λ[k] ≤ seg[k-1] + seg[k] (with sentinels seg[-1]=seg[K]=0)
        prob += lam_alk[t][0] <= seg_alk[t][0]
        for k in range(1, K - 1):
            prob += lam_alk[t][k] <= seg_alk[t][k - 1] + seg_alk[t][k]
        prob += lam_alk[t][K - 1] <= seg_alk[t][K - 2]
        prob += lpSum(seg_alk[t][k] for k in range(K - 1)) == y_alk[t]

        # PEMEL
        prob += lpSum(lam_pem[t][k] for k in range(K)) == y_pem[t]
        prob += lam_pem[t][0] <= seg_pem[t][0]
        for k in range(1, K - 1):
            prob += lam_pem[t][k] <= seg_pem[t][k - 1] + seg_pem[t][k]
        prob += lam_pem[t][K - 1] <= seg_pem[t][K - 2]
        prob += lpSum(seg_pem[t][k] for k in range(K - 1)) == y_pem[t]

    # Load ratio and H2 output from SOS2 interpolation
    for t in range(T):
        # r_alk[t] = Σ λ * r_breakpoint; same for H2
        prob += r_alk[t] == lpSum(
            lam_alk[t][k] * ALKEL_H2_BREAKPOINTS[k][0] for k in range(K))
        h2_alk_t = lpSum(
            lam_alk[t][k] * ALKEL_H2_BREAKPOINTS[k][1] for k in range(K))

        prob += r_pem[t] == lpSum(
            lam_pem[t][k] * PEMEL_H2_BREAKPOINTS[k][0] for k in range(K))
        h2_pem_t = lpSum(
            lam_pem[t][k] * PEMEL_H2_BREAKPOINTS[k][1] for k in range(K))

        # H2 balance: ALKEL H2 + PEMEL H2 = NH3 H2 consumption
        prob += h2_alk_t + h2_pem_t == r_nh3[t] * AMMONIA_H2_NEED

    # Minimum load constraints (differentiated)
    for t in range(T):
        prob += r_alk[t] >= _alkel_min * y_alk[t]
        prob += r_alk[t] <= y_alk[t]
        prob += r_pem[t] >= _pemel_min * y_pem[t]
        prob += r_pem[t] <= y_pem[t]
        prob += r_nh3[t] >= _nh3_min * y_nh3[t]
        prob += r_nh3[t] <= y_nh3[t]

    # Startup detection
    for t in range(T):
        # ALKEL startup: y_alk[t] - y_alk[t-1], with y_alk[-1] = 0
        y_alk_prev = 0 if t == 0 else y_alk[t - 1]
        prob += s_alk[t] >= y_alk[t] - y_alk_prev

        y_pem_prev = 0 if t == 0 else y_pem[t - 1]
        prob += s_pem[t] >= y_pem[t] - y_pem_prev

        y_nh3_prev = 0 if t == 0 else y_nh3[t - 1]
        prob += s_nh3[t] >= y_nh3[t] - y_nh3_prev

    # Ramp rate constraints (ALKEL only, with slack penalty)
    # Relaxed when on/off transition occurs (M_ramp=1.0 covers full range [0,1])
    M_ramp = 1.0
    for t in range(1, T):
        # Ramp up relaxed if was off at t-1
        prob += r_alk[t] - r_alk[t - 1] <= ALKEL_RAMP_MAX + ramp_up[t] + M_ramp * (1 - y_alk[t - 1])
        # Ramp down relaxed if now off at t
        prob += r_alk[t - 1] - r_alk[t] <= ALKEL_RAMP_MAX + ramp_dn[t] + M_ramp * (1 - y_alk[t])

    # Power balance
    for t in range(T):
        equipment_power = (r_alk[t] * ALKEL_P_RATED +
                           r_pem[t] * PEMEL_P_RATED +
                           r_nh3[t] * NH3_P_RATED)

        if grid_connected:
            prob += (P_wind[t] + P_pv[t] + P_buy[t] ==
                     P_load[t] + equipment_power + P_sell[t])
            # No simultaneous buy/sell
            prob += P_buy[t] <= M * z_buy[t]
            prob += P_sell[t] <= M * (1 - z_buy[t])
        else:
            if storage_cap_MWh > 0:
                prob += (P_wind[t] + P_pv[t] + dis[t] ==
                         P_load[t] + equipment_power + chg[t] + curtail[t])
            else:
                # Off-grid no storage: curtailment absorbs excess renewable
                prob += (P_wind[t] + P_pv[t] ==
                         P_load[t] + equipment_power + curtail[t])

    # Storage constraints
    if storage_cap_MWh > 0 and not grid_connected:
        for t in range(T):
            prob += (soc[t + 1] == soc[t] * (1 - STORAGE_SELF_DISCHARGE) +
                     chg[t] * STORAGE_EFF_CHG - dis[t] / STORAGE_EFF_DIS)
        prob += soc[0] == soc[T]  # cyclic constraint

    # Production target
    total_nh3 = lpSum(r_nh3[t] * NH3_RATE for t in range(T))
    if maximize_production:
        prob += total_nh3 <= daily_target  # upper bound only; objective drives it up
    else:
        prob += total_nh3 == daily_target   # exact target for grid-connected

    # --- Objective Function ---
    ramp_penalty_cost = 200.0  # yuan per MW of ramp violation

    if grid_connected:
        prob += lpSum(
            P_buy[t] * _tou_price(t) * 1000
            - P_sell[t] * _feedin_price * 1000
            + r_alk[t] * ALKEL_P_RATED * ALKEL_OM * 1000
            + r_pem[t] * PEMEL_P_RATED * PEMEL_OM * 1000
            + r_nh3[t] * NH3_P_RATED * NH3_OM * 1000
            + s_alk[t] * ALKEL_STARTUP_COST
            + s_pem[t] * PEMEL_STARTUP_COST
            + s_nh3[t] * _nh3_startup_cost
            + ramp_up[t] * ramp_penalty_cost
            + ramp_dn[t] * ramp_penalty_cost
            for t in range(T)
        )
    elif maximize_production:
        # Off-grid production max: min(-total_nh3) = max(total_nh3)
        prob += - total_nh3
    else:
        # Off-grid economic: maximize NH3 value - O&M - startup - ramp
        prob += lpSum(
            - r_nh3[t] * NH3_RATE * _nh3_ref_value
            + r_alk[t] * ALKEL_P_RATED * ALKEL_OM * 1000
            + r_pem[t] * PEMEL_P_RATED * PEMEL_OM * 1000
            + r_nh3[t] * NH3_P_RATED * NH3_OM * 1000
            + s_alk[t] * ALKEL_STARTUP_COST
            + s_pem[t] * PEMEL_STARTUP_COST
            + s_nh3[t] * _nh3_startup_cost
            + ramp_up[t] * ramp_penalty_cost
            + ramp_dn[t] * ramp_penalty_cost
            for t in range(T)
        )

    return prob


# ================================================================
# Solve and extract results
# ================================================================
def solve_dual_electrolyzer(P_wind, P_pv, P_load, daily_target,
                            grid_connected=True, storage_cap_MWh=0,
                            maximize_production=False,
                            time_limit_sec=30,
                            price_params=None,
                            min_load_params=None,
                            nh3_params=None,
                            save_soc=False):
    """Build and solve the dual-electrolyzer MILP.

    Returns results dict or None if infeasible.
    If save_soc=True, result dict includes 'soc_profile' (list of T+1 values).
    """
    prob = build_dual_electrolyzer_model(
        P_wind, P_pv, P_load, daily_target,
        grid_connected, storage_cap_MWh, maximize_production,
        price_params, min_load_params, nh3_params)

    if prob is None:
        return None

    prob.solve(PULP_CBC_CMD(msg=False, timeLimit=time_limit_sec))

    if prob.status != 1:  # not optimal
        return None

    # Resolve price overrides for result computation
    if price_params is None:
        price_params = {}
    _feedin_price = price_params.get('feedin_price', FEEDIN_PRICE)
    _tou_peak = price_params.get('tou_peak', 0.8024)
    _tou_standard = price_params.get('tou_standard', 0.6074)
    _tou_valley = price_params.get('tou_valley', 0.3424)

    def _tou_price(h):
        if 10 <= h < 15 or 18 <= h < 21:
            return _tou_peak
        elif 7 <= h < 10 or 15 <= h < 18 or 21 <= h < 23:
            return _tou_standard
        else:
            return _tou_valley

    T = 24
    vd = prob.variablesDict()

    def val(name, default=0.0):
        v = vd.get(name)
        return value(v) if v is not None else default

    def ival(name, default=0):
        v = vd.get(name)
        return int(round(value(v))) if v is not None else default

    # Extract variable values
    ra = [val(f"ra_{t}") for t in range(T)]
    rp = [val(f"rp_{t}") for t in range(T)]
    rn = [val(f"rn_{t}") for t in range(T)]
    ya = [ival(f"ya_{t}") for t in range(T)]
    yp = [ival(f"yp_{t}") for t in range(T)]

    buy_vals = [val(f"b_{t}") for t in range(T)]
    sell_vals = [val(f"s_{t}") for t in range(T)]

    total_buy = sum(buy_vals)
    total_sell = sum(sell_vals)
    total_ren = float(sum(P_wind[t] + P_pv[t] for t in range(T)))
    total_load = float(sum(P_load[t] + ra[t]*ALKEL_P_RATED +
                           rp[t]*PEMEL_P_RATED + rn[t]*NH3_P_RATED
                           for t in range(T)))

    # Costs
    total_buy_cost = sum(buy_vals[t] * _tou_price(t) * 1000 for t in range(T))
    total_sell_rev = sum(sell_vals[t] * _feedin_price * 1000 for t in range(T))
    total_om = sum((ra[t]*ALKEL_P_RATED*ALKEL_OM +
                    rp[t]*PEMEL_P_RATED*PEMEL_OM +
                    rn[t]*NH3_P_RATED*NH3_OM) * 1000 for t in range(T))
    total_wind_cost = sum(P_wind[t] * WIND_LCOE * 1000 for t in range(T))
    total_pv_cost = sum(P_pv[t] * PV_LCOE * 1000 for t in range(T))
    total_cost = total_wind_cost + total_pv_cost + total_om + total_buy_cost - total_sell_rev

    # Green compliance
    self_use = ((total_load - total_sell - total_buy) / total_ren
                if total_ren > 0 else 0)
    green_ratio = ((total_ren - total_sell) / total_load
                   if total_load > 0 else 0)
    feedin_ratio = (total_sell / total_ren if total_ren > 0 else 0)

    avg_alk_load = float(np.mean([r for r, y in zip(ra, ya) if y > 0]) if any(ya) else 0)
    avg_pem_load = float(np.mean([r for r, y in zip(rp, yp) if y > 0]) if any(yp) else 0)
    alkel_share = float(sum(ra) / (sum(ra) + sum(rp)) if (sum(ra) + sum(rp)) > 0 else 0.5)

    daily_nh3 = sum(rn[t] * NH3_RATE for t in range(T))

    # Storage cost
    daily_storage_cost = 0.0
    if storage_cap_MWh > 0:
        daily_storage_inv = (storage_cap_MWh * 1000 * 1000 / STORAGE_LIFE_DAYS)
        daily_storage_om = storage_cap_MWh * 1000 * STORAGE_OM
        daily_storage_cost = daily_storage_inv + daily_storage_om

    total_cost_with_storage = total_cost + daily_storage_cost
    ton_cost_val = (total_cost_with_storage / daily_nh3 if daily_nh3 > 0
                    else float('inf'))

    # Curtailment
    total_curtail = sum(val(f"c_{t}") for t in range(T))
    ren_used = total_ren - total_curtail

    # SOC profile extraction (for degradation analysis)
    soc_profile = None
    if save_soc and storage_cap_MWh > 0:
        try:
            soc_profile = [val(f"soc_{t}") for t in range(T + 1)]
        except KeyError:
            soc_profile = None

    return {
        'daily_target': daily_target,
        'daily_nh3': round(daily_nh3, 4),
        'ton_cost': round(ton_cost_val, 2),
        'total_cost': round(total_cost_with_storage, 2),
        'total_buy_MWh': round(total_buy, 2),
        'total_sell_MWh': round(total_sell, 2),
        'total_ren_MWh': round(total_ren, 2),
        'total_load_MWh': round(total_load, 2),
        'curtailment_MWh': round(total_curtail, 2),
        'ren_used_MWh': round(ren_used, 2),
        'self_use_ratio': round(self_use, 4),
        'green_ratio': round(green_ratio, 4),
        'feedin_ratio': round(feedin_ratio, 4),
        'all_pass': bool(self_use >= 0.6 and green_ratio >= 0.3 and feedin_ratio < 0.2),
        'pass_count': sum([self_use >= 0.6, green_ratio >= 0.3, feedin_ratio < 0.2]),
        'alkel_ratio': [round(r, 4) for r in ra],
        'pemel_ratio': [round(r, 4) for r in rp],
        'nh3_ratio': [round(r, 4) for r in rn],
        'alkel_on': ya,
        'pemel_on': yp,
        'alkel_startups': int(sum(ival(f"sa_{t}") for t in range(T))),
        'pemel_startups': int(sum(ival(f"sp_{t}") for t in range(T))),
        'nh3_startups': int(sum(ival(f"sn_{t}") for t in range(T))),
        'avg_alkel_load': round(avg_alk_load, 4),
        'avg_pemel_load': round(avg_pem_load, 4),
        'alkel_energy_share': round(alkel_share, 4),
        'equipment_utilization': round(sum(rn) / (T * 1.0), 4),
        'daily_storage_cost': round(daily_storage_cost, 2),
        'cost_breakdown': {
            'wind_lcoe': round(total_wind_cost, 2),
            'pv_lcoe': round(total_pv_cost, 2),
            'om_total': round(total_om, 2),
            'grid_buy': round(total_buy_cost, 2),
            'grid_sell_rev': round(total_sell_rev, 2),
            'storage_daily': round(daily_storage_cost, 2),
        },
        'solve_status': prob.status,
        'soc_profile': soc_profile,
    }


# ================================================================
# Batch solve across all scenarios
# ================================================================
def run_all_scenarios(df_scenarios, production_levels=None):
    """Run dual-electrolyzer model on all 24 scenarios × all production levels.

    Returns:
        all_results: list of per-solve result dicts
        scenario_best: dict mapping scenario_id → best production result
        annual_summary: dict with annual totals and averages
    """
    if production_levels is None:
        production_levels = PRODUCTION_LEVELS

    scenario_ids = sorted(df_scenarios['scenario_id'].unique())
    all_results = []
    scenario_best = {}
    compliance = {'full': 0, 'partial': 0, 'none': 0}

    for sid in scenario_ids:
        df_s = df_scenarios[df_scenarios['scenario_id'] == sid].sort_values('hour')
        Pw = df_s['wind_MW'].values
        Ps = df_s['solar_MW'].values
        Pl = df_s['load_MW'].values

        best_for_sid = None
        for D in production_levels:
            res = solve_dual_electrolyzer(Pw, Ps, Pl, D, grid_connected=True)
            if res:
                res['scenario_id'] = int(sid)
                res['production'] = D
                all_results.append(res)

                if best_for_sid is None or res['ton_cost'] < best_for_sid['ton_cost']:
                    best_for_sid = res

        if best_for_sid:
            scenario_best[int(sid)] = best_for_sid
            if best_for_sid['all_pass']:
                compliance['full'] += 1
            elif best_for_sid['pass_count'] > 0:
                compliance['partial'] += 1
            else:
                compliance['none'] += 1

    annual_nh3 = sum(v['daily_target'] * DAYS_PER_SCENARIO
                     for v in scenario_best.values())
    annual_cost = sum(v['total_cost'] * DAYS_PER_SCENARIO
                      for v in scenario_best.values())
    avg_ton_cost = annual_cost / annual_nh3 if annual_nh3 > 0 else 0

    annual_summary = {
        'annual_nh3_tons': round(annual_nh3, 2),
        'annual_total_cost': round(annual_cost, 2),
        'avg_ton_cost': round(avg_ton_cost, 2),
        'compliance': compliance,
    }

    return all_results, scenario_best, annual_summary


if __name__ == '__main__':
    print("=" * 60)
    print("DUAL-ELECTROLYZER DECOUPLED MILP (PLAN A)")
    print("=" * 60)

    df_scenarios = pd.read_csv(os.path.join(DATA_DIR, SCENARIOS_FILE))

    all_results, scenario_best, annual = run_all_scenarios(df_scenarios)

    print(f"\nAnnual Summary:")
    print(f"  NH3: {annual['annual_nh3_tons']:.0f} tons")
    print(f"  Total cost: {annual['annual_total_cost']:.2f} yuan")
    print(f"  Avg ton cost: {annual['avg_ton_cost']:.2f} yuan/t")
    print(f"  Compliance: full={annual['compliance']['full']}, "
          f"partial={annual['compliance']['partial']}, "
          f"none={annual['compliance']['none']}")

    # Save
    with open(os.path.join(OUT_DIR, 'dual_electrolyzer_results.json'), 'w',
              encoding='utf-8') as f:
        json.dump({
            'all_results': all_results,
            'scenario_best': {str(k): v for k, v in scenario_best.items()},
            'annual_summary': annual,
        }, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {OUT_DIR}")
