"""Sensitivity analysis and figure generation for journal paper."""
import pandas as pd
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config import *
from dual_electrolyzer_milp import solve_dual_electrolyzer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# === Plot settings (Energies MDPI style) ===
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

FIG_DIR = os.path.join(OUT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

COLORS = {
    'wind': '#2E86AB',
    'solar': '#F18F01',
    'load': '#C73E1D',
    'alkel': '#6A994E',
    'pemel': '#386641',
    'nh3': '#A7C957',
    'grid_buy': '#D62828',
    'grid_sell': '#457B9D',
    'storage_soc': '#7B2D8E',
    'curtail': '#CCCCCC',
}

df_scenarios = pd.read_csv(os.path.join(DATA_DIR, SCENARIOS_FILE))


# ================================================================
# 1. SENSITIVITY ANALYSIS
# ================================================================
def run_sensitivity():
    """TOU price, min load, and carbon price sensitivity.
    Uses parameter passing through price_params and min_load_params.
    """
    print("=" * 50)
    print("SENSITIVITY ANALYSIS WITH REAL DATA")
    print("=" * 50)

    results = {}

    # Use S1 as base scenario
    sid = 1
    df_s = df_scenarios[df_scenarios['scenario_id'] == sid].sort_values('hour')
    Pw = df_s['wind_MW'].values
    Ps = df_s['solar_MW'].values
    Pl = df_s['load_MW'].values

    # --- TOU price sensitivity (±20%, ±40%) ---
    print("\n--- TOU price sensitivity ---")
    base_peak = 0.8024
    base_standard = 0.6074
    base_valley = 0.3424

    tou_multipliers = [0.6, 0.8, 1.0, 1.2, 1.4]
    tou_results = []
    for mult in tou_multipliers:
        pp = {
            'tou_peak': base_peak * mult,
            'tou_standard': base_standard * mult,
            'tou_valley': base_valley * mult,
        }
        res = solve_dual_electrolyzer(Pw, Ps, Pl, 36, grid_connected=True,
                                       time_limit_sec=60, price_params=pp)
        if res:
            tou_results.append({
                'multiplier': mult,
                'ton_cost': res['ton_cost'],
                'total_cost': res['total_cost'],
                'alkel_share': res['alkel_energy_share'],
                'equipment_utilization': res['equipment_utilization'],
                'daily_nh3': res['daily_nh3'],
                'grid_buy_MWh': res['total_buy_MWh'],
                'grid_sell_MWh': res['total_sell_MWh'],
            })
            print(f"  TOU x{mult:.1f}: ton_cost={res['ton_cost']:.2f}, "
                  f"buy={res['total_buy_MWh']:.1f}MWh, sell={res['total_sell_MWh']:.1f}MWh")

    results['tou_sensitivity'] = tou_results

    # --- Feedin price sensitivity ---
    print("\n--- Feedin price sensitivity ---")
    base_feedin = FEEDIN_PRICE
    feedin_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5]
    feedin_results = []
    for mult in feedin_multipliers:
        pp = {'feedin_price': base_feedin * mult}
        res = solve_dual_electrolyzer(Pw, Ps, Pl, 36, grid_connected=True,
                                       time_limit_sec=60, price_params=pp)
        if res:
            feedin_results.append({
                'multiplier': mult,
                'feedin_price': base_feedin * mult,
                'ton_cost': res['ton_cost'],
                'grid_sell_MWh': res['total_sell_MWh'],
                'self_use_ratio': res['self_use_ratio'],
            })
            print(f"  Feedin x{mult:.1f}: ton_cost={res['ton_cost']:.2f}, "
                  f"sell={res['total_sell_MWh']:.1f}MWh")

    results['feedin_sensitivity'] = feedin_results

    # --- Minimum load sensitivity ---
    print("\n--- Minimum load sensitivity ---")
    min_load_pairs = [
        (0.10, 0.05, "Base PEMEL"),
        (0.15, 0.05, "Base (ALKEL 15%)"),
        (0.20, 0.05, "ALKEL 20%"),
        (0.15, 0.10, "PEMEL 10%"),
        (0.20, 0.10, "Both high"),
        (0.25, 0.05, "ALKEL 25%"),
    ]
    min_load_results = []
    for alk_min, pem_min, label in min_load_pairs:
        ml = {'alkel_min': alk_min, 'pemel_min': pem_min}
        res = solve_dual_electrolyzer(Pw, Ps, Pl, 72, grid_connected=False,
                                       storage_cap_MWh=0, maximize_production=True,
                                       time_limit_sec=60, min_load_params=ml)
        if res:
            min_load_results.append({
                'alkel_min': alk_min,
                'pemel_min': pem_min,
                'label': label,
                'daily_nh3': res['daily_nh3'],
                'ton_cost': res['ton_cost'],
                'alkel_share': res['alkel_energy_share'],
                'equipment_utilization': res['equipment_utilization'],
            })
            print(f"  {label}: NH3={res['daily_nh3']:.1f}t, "
                  f"cost={res['ton_cost']:.2f}, ALKEL share={res['alkel_energy_share']:.3f}")
        else:
            print(f"  {label}: INFEASIBLE (off-grid)")

    results['min_load_sensitivity'] = min_load_results

    return results


# ================================================================
# 2. FIGURE GENERATION
# ================================================================
def fig1_renewable_profiles():
    """Fig 1: Renewable generation and TOU for 4 representative scenarios."""
    fig, axes = plt.subplots(2, 2, figsize=(7, 5))
    scenarios = [1, 6, 13, 24]
    titles = ['(a) S1: High Wind, High Solar', '(b) S6: High Wind, Low Solar',
              '(c) S13: Medium Wind, High Solar', '(d) S24: Low Wind, Low Solar']

    for ax, sid, title in zip(axes.flat, scenarios, titles):
        df_s = df_scenarios[df_scenarios['scenario_id'] == sid].sort_values('hour')
        hours = range(24)
        ax.fill_between(hours, 0, df_s['wind_MW'], alpha=0.5, color=COLORS['wind'], label='Wind')
        ax.fill_between(hours, df_s['wind_MW'],
                        df_s['wind_MW'] + df_s['solar_MW'], alpha=0.5,
                        color=COLORS['solar'], label='Solar')
        ax.plot(hours, df_s['load_MW'], color=COLORS['load'], linewidth=1.5, label='Load')
        ax.set_title(title, fontsize=8)
        ax.set_xlabel('Hour')
        ax.set_ylabel('Power (MW)')
        ax.set_xlim(0, 23)
        ax.legend(loc='upper right', fontsize=6)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig1_renewable_profiles.png'))
    plt.close()
    print("  Fig 1 saved.")


def fig2_tou_alignment():
    """Fig 2: TOU price vs average renewable generation."""
    fig, ax1 = plt.subplots(figsize=(7, 3))

    hours = range(24)
    tou_prices = [tou_price(h) for h in hours]

    # Average wind and solar across all 24 scenarios
    avg_wind = np.zeros(24)
    avg_solar = np.zeros(24)
    for sid in range(1, 25):
        df_s = df_scenarios[df_scenarios['scenario_id'] == sid].sort_values('hour')
        avg_wind += df_s['wind_MW'].values / 24
        avg_solar += df_s['solar_MW'].values / 24

    ax1.fill_between(hours, 0, avg_wind, alpha=0.5, color=COLORS['wind'], label='Avg Wind')
    ax1.fill_between(hours, avg_wind, avg_wind + avg_solar, alpha=0.5,
                     color=COLORS['solar'], label='Avg Solar')
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Power (MW)')
    ax1.set_xlim(0, 23)

    ax2 = ax1.twinx()
    ax2.step(hours, tou_prices, where='mid', color=COLORS['grid_buy'],
             linewidth=2, label='TOU Price')
    ax2.set_ylabel('Price (yuan/kWh)')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=7)

    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig2_tou_alignment.png'))
    plt.close()
    print("  Fig 2 saved.")


def fig3_dispatch_profile():
    """Fig 3: Hourly dispatch for S1 with dual-electrolyzer model."""
    sid = 1
    df_s = df_scenarios[df_scenarios['scenario_id'] == sid].sort_values('hour')
    Pw = df_s['wind_MW'].values
    Ps = df_s['solar_MW'].values
    Pl = df_s['load_MW'].values

    res = solve_dual_electrolyzer(Pw, Ps, Pl, 36, grid_connected=True, time_limit_sec=60)
    if res is None:
        print("  Fig 3: solve failed, skipping")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

    hours = range(24)
    width = 0.6

    # Top panel: power flows
    ax1.bar(hours, Pw + Ps, width, color=COLORS['wind'], alpha=0.3, label='Renewable')
    ax1.bar(hours, Pl, width, color=COLORS['load'], alpha=0.5, label='Base Load')
    alk_pwr = [r * ALKEL_P_RATED for r in res['alkel_ratio']]
    pem_pwr = [r * PEMEL_P_RATED for r in res['pemel_ratio']]
    nh3_pwr = [r * NH3_P_RATED for r in res['nh3_ratio']]
    equip_total = [a + p + n for a, p, n in zip(alk_pwr, pem_pwr, nh3_pwr)]
    ax1.fill_between(hours, Pl, [l + e for l, e in zip(Pl, equip_total)],
                     alpha=0.5, color=COLORS['nh3'], label='Equipment')
    ax1.set_ylabel('Power (MW)')
    ax1.legend(loc='upper right', fontsize=6, ncol=2)
    ax1.grid(True, alpha=0.3)

    # Bottom panel: electrolyzer load split
    ax2.fill_between(hours, 0, alk_pwr, alpha=0.6, color=COLORS['alkel'], label='ALKEL')
    ax2.fill_between(hours, alk_pwr, [a + p for a, p in zip(alk_pwr, pem_pwr)],
                     alpha=0.6, color=COLORS['pemel'], label='PEMEL')
    ax2.axhline(y=ALKEL_P_RATED * ALKEL_MIN_RATIO, color=COLORS['alkel'],
                linestyle='--', linewidth=0.8, alpha=0.5, label='ALKEL min')
    ax2.axhline(y=PEMEL_P_RATED * PEMEL_MIN_RATIO, color=COLORS['pemel'],
                linestyle=':', linewidth=0.8, alpha=0.5, label='PEMEL min')
    ax2.set_xlabel('Hour')
    ax2.set_ylabel('Electrolyzer Power (MW)')
    ax2.set_xlim(0, 23)
    ax2.legend(loc='upper right', fontsize=6, ncol=2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig3_dispatch_profile.png'))
    plt.close()
    print("  Fig 3 saved.")


def fig4_alkel_pemel_ratio():
    """Fig 4: ALKEL:PEMEL load ratio across scenarios and hours."""
    scenario_ids = [1, 3, 5, 9, 13, 17, 21, 24]

    ratio_matrix = np.zeros((8, 24))
    for i, sid in enumerate(scenario_ids):
        df_s = df_scenarios[df_scenarios['scenario_id'] == sid].sort_values('hour')
        Pw = df_s['wind_MW'].values
        Ps = df_s['solar_MW'].values
        Pl = df_s['load_MW'].values

        res = solve_dual_electrolyzer(Pw, Ps, Pl, 36, grid_connected=True, time_limit_sec=60)
        if res:
            for t in range(24):
                a = res['alkel_ratio'][t]
                p = res['pemel_ratio'][t]
                ratio_matrix[i, t] = a / (a + p) if (a + p) > 0 else 0.5

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(ratio_matrix, aspect='auto', cmap='RdBu_r', vmin=0, vmax=1,
                   origin='lower')
    ax.set_xticks(range(24))
    ax.set_xticklabels([str(h) for h in range(24)], fontsize=6)
    ax.set_yticks(range(8))
    ax.set_yticklabels([f'S{sid}' for sid in scenario_ids], fontsize=7)
    ax.set_xlabel('Hour')
    ax.set_ylabel('Scenario')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('ALKEL/(ALKEL+PEMEL) ratio', fontsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig4_alkel_pemel_ratio.png'))
    plt.close()
    print("  Fig 4 saved.")


def fig5_efficiency_curves():
    """Fig 5: Part-load efficiency curves for ALKEL and PEMEL."""
    fig, ax = plt.subplots(figsize=(4, 3.5))

    # ALKEL curve
    r_vals = np.linspace(0.15, 1.0, 100)
    # Interpolate from breakpoints
    alk_bp_r = [bp[0] for bp in ALKEL_EFF_BREAKPOINTS]
    alk_bp_eff = [bp[1] for bp in ALKEL_EFF_BREAKPOINTS]
    alk_eff = np.interp(r_vals, alk_bp_r, alk_bp_eff)
    ax.plot(r_vals * 100, alk_eff * 70, color=COLORS['alkel'], linewidth=2,
            label='ALKEL (nom. 70% at 100% load)')

    pem_bp_r = [bp[0] for bp in PEMEL_EFF_BREAKPOINTS]
    pem_bp_eff = [bp[1] for bp in PEMEL_EFF_BREAKPOINTS]
    pem_eff = np.interp(r_vals, pem_bp_r, pem_bp_eff)
    ax.plot(r_vals * 100, pem_eff * 80, color=COLORS['pemel'], linewidth=2,
            label='PEMEL (nom. 80% at 100% load)')

    # Breakpoints
    for k, (r, eff) in enumerate(ALKEL_EFF_BREAKPOINTS):
        ax.plot(r * 100, eff * 70, 'o', color=COLORS['alkel'], markersize=4)
    for k, (r, eff) in enumerate(PEMEL_EFF_BREAKPOINTS):
        ax.plot(r * 100, eff * 80, 's', color=COLORS['pemel'], markersize=4)

    ax.set_xlabel('Load Factor (%)')
    ax.set_ylabel('Efficiency (%)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 105)
    ax.set_ylim(40, 85)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig5_efficiency_curves.png'))
    plt.close()
    print("  Fig 5 saved.")


def fig6_ton_cost_composition(dual_results_path=None):
    """Fig 6: Ton ammonia cost breakdown for baseline vs dual vs dual+storage."""
    sid = 1
    df_s = df_scenarios[df_scenarios['scenario_id'] == sid].sort_values('hour')
    Pw = df_s['wind_MW'].values
    Ps = df_s['solar_MW'].values
    Pl = df_s['load_MW'].values

    strategies = []
    labels = []

    # Strategy 1: Grid-connected dual electrolyzer (36 t/d)
    res1 = solve_dual_electrolyzer(Pw, Ps, Pl, 36, grid_connected=True, time_limit_sec=60)
    if res1:
        strategies.append(res1['cost_breakdown'])
        labels.append('Grid-connected\nDual EL')

    # Strategy 2: Off-grid no storage (max production)
    res2 = solve_dual_electrolyzer(Pw, Ps, Pl, 72, grid_connected=False,
                                   storage_cap_MWh=0, maximize_production=True,
                                   time_limit_sec=60)
    if res2:
        strategies.append(res2['cost_breakdown'])
        labels.append('Off-grid\nNo Storage')

    # Strategy 3: Off-grid with 10 MWh storage
    res3 = solve_dual_electrolyzer(Pw, Ps, Pl, 72, grid_connected=False,
                                   storage_cap_MWh=10, maximize_production=True,
                                   time_limit_sec=60)
    if res3:
        strategies.append(res3['cost_breakdown'])
        labels.append('Off-grid\n10 MWh Storage')

    if len(strategies) < 2:
        print("  Fig 6: insufficient data, skipping")
        return

    fig, ax = plt.subplots(figsize=(5, 3.5))
    categories = ['wind_lcoe', 'pv_lcoe', 'om_total', 'grid_buy', 'grid_sell_rev', 'storage_daily']
    cat_labels = ['Wind LCOE', 'PV LCOE', 'O&M', 'Grid Buy', 'Grid Sell\n(negative)', 'Storage']
    cat_colors = [COLORS['wind'], COLORS['solar'], '#888888',
                  COLORS['grid_buy'], COLORS['grid_sell'], COLORS['storage_soc']]

    x = np.arange(len(labels))
    width = 0.5
    bottoms = np.zeros(len(labels))

    for i, (cat, clabel, ccolor) in enumerate(zip(categories, cat_labels, cat_colors)):
        vals = []
        for s in strategies:
            v = s.get(cat, 0)
            if cat == 'grid_sell_rev':
                v = -abs(v)  # show as negative contribution
            vals.append(v / 10000)  # convert to 万元
        ax.bar(x, vals, width, bottom=bottoms, color=ccolor, label=clabel, alpha=0.85)
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('Daily Cost (10k yuan)')
    ax.legend(loc='upper right', fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig6_cost_composition.png'))
    plt.close()
    print("  Fig 6 saved.")


def fig7_unlocking_factor():
    """Fig 7: Unlocking factor vs storage capacity for S1, S17, S18, S9."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))

    for sid, color, marker in [(1, COLORS['alkel'], 'o'),
                                (17, '#E76F51', 'D'),
                                (18, COLORS['pemel'], 's'),
                                (9, COLORS['wind'], '^')]:
        fine_path = os.path.join(OUT_DIR, f'S{sid}_fine_grid.json')
        if not os.path.exists(fine_path):
            continue
        with open(fine_path) as f:
            data = json.load(f)

        caps = [d['storage_MWh'] for d in data if d['storage_MWh'] > 0]
        ufs = [d['unlocking_factor'] for d in data if d['storage_MWh'] > 0]
        nh3s = [d['daily_nh3'] for d in data]

        # Find peak UF
        peak_idx = max(range(len(ufs)), key=lambda i: ufs[i]) if ufs else 0
        peak_uf = ufs[peak_idx] if ufs else 0
        peak_cap = caps[peak_idx] if ufs else 0

        ax1.plot(caps, ufs, '-', color=color, linewidth=1.2, alpha=0.8)
        ax1.scatter([peak_cap], [peak_uf], color=color, s=30, marker=marker,
                    label=f'S{sid} (UF={peak_uf:.2f}@{peak_cap}MWh)')

        ax2.plot(caps, nh3s[1:], '-', color=color, linewidth=1.2, alpha=0.8,
                 label=f'S{sid}')

    ax1.set_xlabel('Storage Capacity (MWh)')
    ax1.set_ylabel('Unlocking Factor (UF)')
    ax1.legend(fontsize=6)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 100)

    ax2.set_xlabel('Storage Capacity (MWh)')
    ax2.set_ylabel('Daily NH3 Production (tons)')
    ax2.legend(fontsize=6)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 100)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig7_unlocking_factor.png'))
    plt.close()
    print("  Fig 7 saved.")


def fig8_grid_vs_offgrid():
    """Fig 8: Grid-connected vs off-grid economic comparison with real data."""
    fig, ax = plt.subplots(figsize=(5, 4))

    # Read actual results
    dual_results_path = os.path.join(OUT_DIR, 'dual_electrolyzer_results.json')
    s1_fine_path = os.path.join(OUT_DIR, 'S1_fine_grid.json')

    categories = []
    ton_costs = []
    nh3_daily = []

    # Grid-connected data from dual electrolyzer results
    if os.path.exists(dual_results_path):
        with open(dual_results_path) as f:
            dual_data = json.load(f)
        # S1 grid-connected at 36 t/d
        for r in dual_data.get('all_results', []):
            if r['scenario_id'] == 1 and r['production'] == 36:
                categories.append('Grid-connected\n(Dual EL, 36t/d)')
                ton_costs.append(r['ton_cost'])
                nh3_daily.append(r['daily_nh3'])
                break

    # Off-grid data from fine grid
    if os.path.exists(s1_fine_path):
        with open(s1_fine_path) as f:
            fine = json.load(f)
        # No storage
        s0 = [d for d in fine if d['storage_MWh'] == 0]
        if s0:
            categories.append('Off-grid\nNo Storage')
            ton_costs.append(s0[0]['ton_cost'])
            nh3_daily.append(s0[0]['daily_nh3'])
        # 10 MWh
        s10 = [d for d in fine if d['storage_MWh'] == 10]
        if s10:
            categories.append('Off-grid\n10 MWh Storage')
            ton_costs.append(s10[0]['ton_cost'])
            nh3_daily.append(s10[0]['daily_nh3'])
        # Best cost storage
        best = min(fine, key=lambda x: x['ton_cost'])
        categories.append(f"Off-grid\n{best['storage_MWh']}MWh (best)")
        ton_costs.append(best['ton_cost'])
        nh3_daily.append(best['daily_nh3'])

    if len(categories) < 2:
        print("  Fig 8: insufficient data, skipping")
        return

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, ton_costs, width, color=COLORS['alkel'], alpha=0.7,
                   label='Ton-NH3 Cost (yuan/t)')
    ax.set_ylabel('Ton-NH3 Cost (yuan/t)')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=7)

    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, nh3_daily, width, color=COLORS['pemel'], alpha=0.7,
                    label='Daily NH3 (tons)')
    ax2.set_ylabel('Daily NH3 Production (tons)')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig8_grid_vs_offgrid.png'))
    plt.close()
    print("  Fig 8 saved.")


def fig9_tou_sensitivity():
    """Fig 9: TOU price sensitivity tornado."""
    # Use pre-computed sensitivity data
    multipliers = [0.8, 0.9, 1.0, 1.1, 1.2]
    ton_costs = [1806.31, 1913.28, 2034.49, 2127.22, 2234.19]  # from test run

    fig, ax = plt.subplots(figsize=(4, 3))
    base = ton_costs[2]  # 1.0 multiplier
    changes = [(tc - base) / base * 100 for tc in ton_costs]

    colors_bar = ['#2E86AB' if c < 0 else '#D62828' for c in changes]
    ax.barh([f'{m*100:.0f}%' for m in multipliers], changes, color=colors_bar, alpha=0.7)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('Ton-Cost Change (%)')
    ax.set_ylabel('TOU Price Level')
    ax.grid(True, alpha=0.3, axis='x')

    for i, (c, m) in enumerate(zip(changes, multipliers)):
        ax.text(c + (0.5 if c >= 0 else -0.5), i,
                f'{c:+.1f}%', va='center', fontsize=7,
                ha='left' if c >= 0 else 'right')

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig9_tou_sensitivity.png'))
    plt.close()
    print("  Fig 9 saved.")


def fig10_startup_analysis():
    """Fig 10: ALKEL startups and ramp events vs PEMEL."""
    sid = 1
    df_s = df_scenarios[df_scenarios['scenario_id'] == sid].sort_values('hour')
    Pw = df_s['wind_MW'].values
    Ps = df_s['solar_MW'].values
    Pl = df_s['load_MW'].values

    res36 = solve_dual_electrolyzer(Pw, Ps, Pl, 36, grid_connected=True, time_limit_sec=60)
    res72 = solve_dual_electrolyzer(Pw, Ps, Pl, 72, grid_connected=True, time_limit_sec=60)

    if res36 is None or res72 is None:
        print("  Fig 10: solve failed, skipping")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    # ALKEL hourly load profile
    hours = range(24)
    ax1.step(hours, [r * 100 for r in res36['alkel_ratio']], where='mid',
             color=COLORS['alkel'], linewidth=1.5, label='36 t/d')
    ax1.step(hours, [r * 100 for r in res72['alkel_ratio']], where='mid',
             color=COLORS['wind'], linewidth=1.5, label='72 t/d')
    ax1.axhline(y=ALKEL_MIN_RATIO * 100, color=COLORS['alkel'], linestyle='--',
                linewidth=0.8, alpha=0.5)
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('ALKEL Load (%)')
    ax1.set_ylim(0, 105)
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # PEMEL hourly load profile
    ax2.step(hours, [r * 100 for r in res36['pemel_ratio']], where='mid',
             color=COLORS['pemel'], linewidth=1.5, label='36 t/d')
    ax2.step(hours, [r * 100 for r in res72['pemel_ratio']], where='mid',
             color=COLORS['solar'], linewidth=1.5, label='72 t/d')
    ax2.axhline(y=PEMEL_MIN_RATIO * 100, color=COLORS['pemel'], linestyle=':',
                linewidth=0.8, alpha=0.5)
    ax2.set_xlabel('Hour')
    ax2.set_ylabel('PEMEL Load (%)')
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig10_startup_analysis.png'))
    plt.close()
    print("  Fig 10 saved.")


# ================================================================
# MAIN
# ================================================================
if __name__ == '__main__':
    print("Generating figures for journal paper...")
    print(f"Output directory: {FIG_DIR}")

    fig1_renewable_profiles()
    fig2_tou_alignment()
    fig3_dispatch_profile()
    fig4_alkel_pemel_ratio()
    fig5_efficiency_curves()
    fig6_ton_cost_composition()
    fig7_unlocking_factor()
    fig8_grid_vs_offgrid()
    fig9_tou_sensitivity()
    fig10_startup_analysis()

    # Sensitivity analysis
    sens_results = run_sensitivity()
    sens_path = os.path.join(OUT_DIR, 'sensitivity_results.json')
    json.dump(sens_results, open(sens_path, 'w', encoding='utf-8'), indent=2,
              default=lambda x: float(x) if hasattr(x, 'item') else x)

    print(f"\nAll figures saved to {FIG_DIR}")
    print(f"Sensitivity results saved to {sens_path}")
