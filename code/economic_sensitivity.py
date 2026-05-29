"""Economic sensitivity analysis (LCOA tornado chart).

Sweeps five economic parameters to determine which dominate the
levelized cost of ammonia (LCOA). Generates tornado chart data.

Parameters:
  1. Storage investment cost (500-2000 yuan/kWh) -- re-solve MILP
  2. Wind/PV LCOE (+-30%) -- post-solve
  3. NH3 selling price (+-30%) -- re-solve for off-grid
  4. Electrolyzer CAPEX (+-30%) -- post-solve
  5. Discount rate (5%-15%) -- post-solve
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
# 1. CAPEX and annualization
# ================================================================
# Capital costs (yuan/kW installed)
CAPEX = {
    'wind': 4000,
    'pv': 3500,
    'alkel': 3000,
    'pemel': 5000,
    'nh3_plant': 8000,
    'storage': STORAGE_INVESTMENT,  # yuan/kWh
}

LIFETIMES = {
    'wind': 25,
    'pv': 25,
    'alkel': 20,
    'pemel': 15,
    'nh3_plant': 25,
    'storage': STORAGE_LIFE_YEARS,
}

BASE_DISCOUNT = 0.08  # 8%


def crf(rate, years):
    """Capital recovery factor."""
    if rate == 0:
        return 1.0 / years
    return rate * (1 + rate) ** years / ((1 + rate) ** years - 1)


def annualize_capex(total_investment, lifetime_years, discount_rate):
    return total_investment * crf(discount_rate, lifetime_years)


def compute_lcoa(result, discount_rate, capex_mult=1.0, lcoe_mult=1.0):
    """Compute levelized cost of ammonia from a dispatch result.

    Args:
        result: solve_dual_electrolyzer result dict (day-level)
        discount_rate: decimal
        capex_mult: multiplier on all CAPEX values
        lcoe_mult: multiplier on wind/PV LCOE

    Returns:
        LCOA in yuan/ton, and cost breakdown dict.
    """
    daily_nh3 = result['daily_nh3']
    if daily_nh3 <= 0:
        return float('inf'), {}

    # Annualize CAPEX
    wind_capex_annual = annualize_capex(
        WIND_CAP_MW * 1000 * CAPEX['wind'] * capex_mult,
        LIFETIMES['wind'], discount_rate)
    pv_capex_annual = annualize_capex(
        PV_CAP_MW * 1000 * CAPEX['pv'] * capex_mult,
        LIFETIMES['pv'], discount_rate)
    alkel_capex_annual = annualize_capex(
        ALKEL_P_RATED * 1000 * CAPEX['alkel'] * capex_mult,
        LIFETIMES['alkel'], discount_rate)
    pemel_capex_annual = annualize_capex(
        PEMEL_P_RATED * 1000 * CAPEX['pemel'] * capex_mult,
        LIFETIMES['pemel'], discount_rate)
    nh3_capex_annual = annualize_capex(
        NH3_P_RATED * 1000 * CAPEX['nh3_plant'] * capex_mult,
        LIFETIMES['nh3_plant'], discount_rate)

    # Annualize storage
    storage_MWh = result.get('storage_MWh', 0)
    if 'storage_cap_MWh' in result:
        storage_MWh = result['storage_cap_MWh']
    storage_capex_annual = annualize_capex(
        storage_MWh * 1000 * CAPEX['storage'] * capex_mult,
        LIFETIMES['storage'], discount_rate) if storage_MWh > 0 else 0

    total_annual_capex = (wind_capex_annual + pv_capex_annual +
                          alkel_capex_annual + pemel_capex_annual +
                          nh3_capex_annual + storage_capex_annual)

    # Annual O&M, grid, storage (scale from daily values)
    # result['total_cost'] includes: wind_lcoe, pv_lcoe, om, grid_buy, grid_sell_rev, storage
    cb = result.get('cost_breakdown', {})
    annual_wind_lcoe = cb.get('wind_lcoe', 0) * 365 * lcoe_mult
    annual_pv_lcoe = cb.get('pv_lcoe', 0) * 365 * lcoe_mult
    annual_om = cb.get('om_total', 0) * 365
    annual_grid_buy = cb.get('grid_buy', 0) * 365
    annual_grid_sell_rev = cb.get('grid_sell_rev', 0) * 365
    annual_storage_om = cb.get('storage_daily', 0) * 365

    annual_nh3_tons = daily_nh3 * 365

    total_annual_cost = (total_annual_capex + annual_wind_lcoe + annual_pv_lcoe +
                         annual_om + annual_grid_buy - annual_grid_sell_rev +
                         annual_storage_om)

    lcoa = total_annual_cost / annual_nh3_tons

    return lcoa, {
        'capex_annual': total_annual_capex,
        'lcoe_annual': annual_wind_lcoe + annual_pv_lcoe,
        'om_annual': annual_om + annual_storage_om,
        'grid_net': annual_grid_buy - annual_grid_sell_rev,
    }


# ================================================================
# 2. Base case solve
# ================================================================
print("=" * 70)
print("Economic Sensitivity Analysis (LCOA Tornado)")
print("=" * 70)

df = pd.read_csv(os.path.join(DATA_DIR_ABS, SCENARIOS_FILE))
df_s = df[df['scenario_id'] == 1].sort_values('hour')
P_wind = df_s['wind_MW'].values
P_pv = df_s['solar_MW'].values
P_load = df_s['load_MW'].values

# Base case: S1 grid-connected at 36 t/d, 0 MWh storage (simple case)
print("\nSolving base case (S1, grid-connected, 36 t/d, 0 MWh storage)...")
base_r = solve_dual_electrolyzer(
    P_wind, P_pv, P_load, 36,
    grid_connected=True, storage_cap_MWh=0,
    time_limit_sec=60,
)

if base_r is None:
    print("BASE CASE INFEASIBLE -- aborting")
    sys.exit(1)

# Store storage capacity in result for LCOA computation
base_r['storage_MWh'] = 0
base_lcoa, base_breakdown = compute_lcoa(base_r, BASE_DISCOUNT)
print(f"Base LCOA: {base_lcoa:.1f} yuan/ton NH3")
print(f"  CAPEX annual: {base_breakdown['capex_annual']:.0f}")
print(f"  LCOE annual:  {base_breakdown['lcoe_annual']:.0f}")
print(f"  O&M annual:   {base_breakdown['om_annual']:.0f}")
print(f"  Grid net:     {base_breakdown['grid_net']:.0f}")

# ================================================================
# 3. Parameter sweeps
# ================================================================

SWEEP_POINTS = [0.7, 0.85, 1.0, 1.15, 1.3]

tornado_data = []

# 3.1 Storage investment cost (re-solve MILP with storage=5MWh)
print("\nSweep 1: Storage investment cost (500-2000 yuan/kWh)...")
for mult in SWEEP_POINTS:
    inv = STORAGE_INVESTMENT * mult
    # Use a temporary override: re-solve with storage=5 MWh and adjusted cost
    r = solve_dual_electrolyzer(
        P_wind, P_pv, P_load, 36,
        grid_connected=True, storage_cap_MWh=5,
        price_params={'feedin_price': FEEDIN_PRICE},
        time_limit_sec=60,
    )
    if r:
        r['storage_MWh'] = 5
        # Override storage investment for LCOA calculation
        orig_inv = CAPEX['storage']
        CAPEX['storage'] = inv
        lcoa, _ = compute_lcoa(r, BASE_DISCOUNT)
        CAPEX['storage'] = orig_inv
        tornado_data.append({
            'parameter': 'Storage investment',
            'multiplier': mult,
            'value': inv,
            'unit': 'yuan/kWh',
            'lcoa': round(lcoa, 1),
            'lcoa_delta_pct': round((lcoa / base_lcoa - 1) * 100, 2),
        })
        print(f"  {inv:.0f} yuan/kWh: LCOA={lcoa:.1f} yuan/t "
              f"({(lcoa/base_lcoa-1)*100:+.1f}%)")

# 3.2 Wind/PV LCOE (post-solve only)
print("\nSweep 2: Wind/PV LCOE (+-30%)...")
for mult in SWEEP_POINTS:
    lcoa, _ = compute_lcoa(base_r, BASE_DISCOUNT, lcoe_mult=mult)
    tornado_data.append({
        'parameter': 'Wind/PV LCOE',
        'multiplier': mult,
        'value': mult,
        'unit': 'x base',
        'lcoa': round(lcoa, 1),
        'lcoa_delta_pct': round((lcoa / base_lcoa - 1) * 100, 2),
    })
    print(f"  {mult:.2f}x: LCOA={lcoa:.1f} yuan/t "
          f"({(lcoa/base_lcoa-1)*100:+.1f}%)")

# 3.3 NH3 selling price (re-solve off-grid for max production)
print("\nSweep 3: NH3 selling price (+-30%)...")
for mult in SWEEP_POINTS:
    nh3_price = NH3_REF_VALUE * mult
    r = solve_dual_electrolyzer(
        P_wind, P_pv, P_load, 72,
        grid_connected=False, storage_cap_MWh=0,
        maximize_production=True,
        price_params={'nh3_ref_value': nh3_price},
        time_limit_sec=60,
    )
    if r:
        r['storage_MWh'] = 0
        lcoa, _ = compute_lcoa(r, BASE_DISCOUNT)
        tornado_data.append({
            'parameter': 'NH3 selling price',
            'multiplier': mult,
            'value': nh3_price,
            'unit': 'yuan/ton',
            'lcoa': round(lcoa, 1),
            'lcoa_delta_pct': round((lcoa / base_lcoa - 1) * 100, 2),
        })
        print(f"  {nh3_price:.0f} yuan/ton: LCOA={lcoa:.1f} yuan/t "
              f"({(lcoa/base_lcoa-1)*100:+.1f}%)")

# 3.4 Electrolyzer CAPEX (post-solve only)
print("\nSweep 4: Electrolyzer CAPEX (+-30%)...")
for mult in SWEEP_POINTS:
    lcoa, _ = compute_lcoa(base_r, BASE_DISCOUNT, capex_mult=mult)
    tornado_data.append({
        'parameter': 'Electrolyzer CAPEX',
        'multiplier': mult,
        'value': mult,
        'unit': 'x base',
        'lcoa': round(lcoa, 1),
        'lcoa_delta_pct': round((lcoa / base_lcoa - 1) * 100, 2),
    })
    print(f"  {mult:.2f}x: LCOA={lcoa:.1f} yuan/t "
          f"({(lcoa/base_lcoa-1)*100:+.1f}%)")

# 3.5 Discount rate (post-solve only)
print("\nSweep 5: Discount rate (5%-15%)...")
for rate in [0.05, 0.08, 0.10, 0.12, 0.15]:
    lcoa, _ = compute_lcoa(base_r, rate)
    tornado_data.append({
        'parameter': 'Discount rate',
        'multiplier': rate / BASE_DISCOUNT,
        'value': rate * 100,
        'unit': '%',
        'lcoa': round(lcoa, 1),
        'lcoa_delta_pct': round((lcoa / base_lcoa - 1) * 100, 2),
    })
    print(f"  {rate*100:.0f}%: LCOA={lcoa:.1f} yuan/t "
          f"({(lcoa/base_lcoa-1)*100:+.1f}%)")

# ================================================================
# 4. Tornado summary
# ================================================================
print("\n" + "=" * 70)
print("Tornado Chart Summary (LCOA sensitivity, ranked by impact):")
print("=" * 70)

# For each parameter, compute the range (max_lcoa - min_lcoa)
param_ranges = {}
for d in tornado_data:
    p = d['parameter']
    if p not in param_ranges:
        param_ranges[p] = {'min': float('inf'), 'max': float('-inf')}
    param_ranges[p]['min'] = min(param_ranges[p]['min'], d['lcoa_delta_pct'])
    param_ranges[p]['max'] = max(param_ranges[p]['max'], d['lcoa_delta_pct'])

for p in sorted(param_ranges, key=lambda x: param_ranges[x]['max'] - param_ranges[x]['min'], reverse=True):
    rng = param_ranges[p]
    print(f"  {p:<25s}: {rng['min']:+.1f}% to {rng['max']:+.1f}% "
          f"(span={rng['max']-rng['min']:.1f}%)")

# Save
out_path = os.path.join(OUT_DIR_ABS, 'economic_sensitivity_results.json')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w') as f:
    json.dump({
        'base_lcoa': round(base_lcoa, 1),
        'tornado_data': tornado_data,
        'param_ranges': {k: v for k, v in param_ranges.items()},
    }, f, indent=2, default=str)

print(f"\nResults saved to: {out_path}")
print("Done.")
