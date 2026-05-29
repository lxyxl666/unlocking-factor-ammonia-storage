"""Supplemental figures for journal paper — all from saved data, no MILP solves."""
import pandas as pd, numpy as np, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from config import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})

FIG_DIR = os.path.join(OUT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

COLORS = {
    'wind': '#2E86AB', 'solar': '#F18F01', 'load': '#C73E1D',
    'alkel': '#6A994E', 'pemel': '#386641', 'nh3': '#A7C957',
    'grid_buy': '#D62828', 'grid_sell': '#457B9D', 'storage_soc': '#7B2D8E',
    'curtail': '#CCCCCC',
}


def fig_uf_histogram():
    """UF distribution across all scenarios at each storage level."""
    uf_csv = os.path.join(OUT_DIR, 'unlocking_factor_all.csv')
    if not os.path.exists(uf_csv):
        print("  UF histogram: no data, skipping")
        return

    df = pd.read_csv(uf_csv)
    df_nz = df[df['storage_MWh'] > 0]

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.8))

    # Panel 1: UF histogram at 10 MWh
    df_10 = df_nz[df_nz['storage_MWh'] == 10]
    ax = axes[0]
    ax.hist(df_10['unlocking_factor'].dropna(), bins=15, color=COLORS['alkel'],
            alpha=0.7, edgecolor='black', linewidth=0.3)
    ax.axvline(x=1.0, color=COLORS['grid_buy'], linestyle='--', linewidth=0.8,
               label='UF=1 (breakeven)')
    ax.set_xlabel('UF at 10 MWh')
    ax.set_ylabel('Count')
    ax.legend(fontsize=6)
    ax.set_title('(a) UF at 10 MWh', fontsize=8)

    # Panel 2: UF by scenario type
    ax = axes[1]
    scenario_ufs = df_nz.groupby('scenario_id')['unlocking_factor'].max()
    feasible = scenario_ufs.dropna()
    colors = [COLORS['alkel'] if v >= 1 else COLORS['grid_sell'] for v in feasible.values]
    ax.bar(range(len(feasible)), feasible.values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.3)
    ax.axhline(y=1.0, color=COLORS['grid_buy'], linestyle='--', linewidth=0.8)
    ax.set_xlabel('Scenario ID')
    ax.set_ylabel('Max UF')
    ax.set_title('(b) Max UF by Scenario', fontsize=8)
    ax.set_xticks(range(0, len(feasible), 3))
    ax.set_xticklabels([str(int(feasible.index[i])) for i in range(0, len(feasible), 3)])

    # Panel 3: UF decay with storage
    ax = axes[2]
    for sid in [1, 6, 17, 18]:
        df_s = df_nz[df_nz['scenario_id'] == sid].sort_values('storage_MWh')
        ax.plot(df_s['storage_MWh'], df_s['unlocking_factor'], '-', linewidth=1.2,
                alpha=0.8, label=f'S{sid}')

    ax.set_xlabel('Storage Capacity (MWh)')
    ax.set_ylabel('Unlocking Factor')
    ax.legend(fontsize=6)
    ax.set_title('(c) UF Decay Curves', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_uf_histogram.png'))
    plt.close()
    print("  UF histogram saved.")


def fig_nh3_vs_storage():
    """NH3 production vs storage capacity for key scenarios."""
    uf_csv = os.path.join(OUT_DIR, 'unlocking_factor_all.csv')
    if not os.path.exists(uf_csv):
        return

    df = pd.read_csv(uf_csv)

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    for sid, color, label in [(1, COLORS['alkel'], 'S1: High W, High S'),
                                (6, COLORS['pemel'], 'S6: High W, Low S'),
                                (17, '#E76F51', 'S17: Mid W, High S'),
                                (18, '#F4A261', 'S18: Mid W, Mid S'),
                                (10, COLORS['wind'], 'S10: Mid W, High S')]:
        df_s = df[df['scenario_id'] == sid].sort_values('storage_MWh')
        ax.plot(df_s['storage_MWh'], df_s['daily_nh3'], '-o', color=color,
                linewidth=1.2, markersize=3, alpha=0.8, label=label)

    ax.set_xlabel('Storage Capacity (MWh)')
    ax.set_ylabel('Daily NH3 Production (tons)')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_nh3_vs_storage.png'))
    plt.close()
    print("  NH3 vs storage saved.")


def fig_cost_vs_green():
    """Ton-cost vs green compliance trade-off (Pareto front)."""
    # Read dual electrolyzer results
    dual_path = os.path.join(OUT_DIR, 'dual_electrolyzer_results.json')
    if not os.path.exists(dual_path):
        print("  Cost vs green: no data, skipping")
        return

    with open(dual_path) as f:
        data = json.load(f)

    all_results = data.get('all_results', [])
    if not all_results:
        return

    df = pd.DataFrame(all_results)

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    # Color by pass count
    pass_colors = {0: '#D62828', 1: '#F18F01', 2: '#2E86AB', 3: '#6A994E'}
    pass_labels = {0: 'None', 1: '1/3', 2: '2/3', 3: 'Full (3/3)'}

    for pc in sorted(df['pass_count'].unique()):
        df_pc = df[df['pass_count'] == pc]
        ax.scatter(df_pc['green_ratio'], df_pc['ton_cost'],
                   c=pass_colors.get(int(pc), '#888888'),
                   label=pass_labels.get(int(pc), str(pc)),
                   alpha=0.6, s=15, edgecolors='none')

    ax.set_xlabel('Green Electricity Ratio')
    ax.set_ylabel('Ton-NH3 Cost (yuan/t)')
    ax.legend(fontsize=6, title='Compliance')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_cost_vs_green.png'))
    plt.close()
    print("  Cost vs green saved.")


def fig_grid_vs_offgrid_all():
    """Grid vs off-grid comparison across all feasible scenarios."""
    dual_path = os.path.join(OUT_DIR, 'dual_electrolyzer_results.json')
    uf_csv = os.path.join(OUT_DIR, 'unlocking_factor_all.csv')

    if not os.path.exists(dual_path) or not os.path.exists(uf_csv):
        return

    with open(dual_path) as f:
        dual_data = json.load(f)

    df_uf = pd.read_csv(uf_csv)

    # Grid-connected best per scenario
    df_all = pd.DataFrame(dual_data.get('all_results', []))
    grid_best = df_all.loc[df_all.groupby('scenario_id')['ton_cost'].idxmin()]

    # Off-grid best per scenario (from UF data)
    off_best = df_uf.loc[df_uf.groupby('scenario_id')['ton_cost'].idxmin()]

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))

    # Panel 1: Ton cost comparison
    ax = axes[0]
    common_sids = sorted(set(grid_best['scenario_id']) & set(off_best['scenario_id']))
    x = np.arange(len(common_sids))
    width = 0.35

    grid_costs = [grid_best[grid_best['scenario_id'] == s]['ton_cost'].values[0] for s in common_sids]
    off_costs = [off_best[off_best['scenario_id'] == s]['ton_cost'].values[0] for s in common_sids]

    ax.bar(x - width/2, grid_costs, width, color=COLORS['alkel'], alpha=0.7, label='Grid-connected')
    ax.bar(x + width/2, off_costs, width, color=COLORS['grid_sell'], alpha=0.7, label='Off-grid')
    ax.set_xlabel('Scenario')
    ax.set_ylabel('Ton Cost (yuan/t)')
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(s)) for s in common_sids], fontsize=6)
    ax.legend(fontsize=7)
    ax.set_title('(a) Ton-NH3 Cost', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 2: NH3 difference
    ax = axes[1]
    grid_nh3 = [grid_best[grid_best['scenario_id'] == s]['daily_nh3'].values[0] for s in common_sids]
    off_nh3 = [off_best[off_best['scenario_id'] == s]['daily_nh3'].values[0] for s in common_sids]

    ax.bar(x - width/2, grid_nh3, width, color=COLORS['alkel'], alpha=0.7, label='Grid-connected')
    ax.bar(x + width/2, off_nh3, width, color=COLORS['grid_sell'], alpha=0.7, label='Off-grid')
    ax.set_xlabel('Scenario')
    ax.set_ylabel('Daily NH3 (tons)')
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(s)) for s in common_sids], fontsize=6)
    ax.legend(fontsize=7)
    ax.set_title('(b) Daily NH3 Production', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_grid_vs_offgrid_all.png'))
    plt.close()
    print("  Grid vs offgrid (all scenarios) saved.")


def fig_wind_solar_resource_map():
    """Wind-solar resource space with scenario categorization."""
    df = pd.read_csv(os.path.join(DATA_DIR, SCENARIOS_FILE))

    fig, ax = plt.subplots(figsize=(5, 3.5))

    scenarios_agg = {}
    for sid in sorted(df['scenario_id'].unique()):
        df_s = df[df['scenario_id'] == sid]
        w_total = df_s['wind_MW'].sum()
        s_total = df_s['solar_MW'].sum()
        scenarios_agg[sid] = {'wind': w_total, 'solar': s_total}

    wind_vals = [v['wind'] for v in scenarios_agg.values()]
    solar_vals = [v['solar'] for v in scenarios_agg.values()]

    # Check infeasible off-grid
    uf_csv = os.path.join(OUT_DIR, 'unlocking_factor_all.csv')
    infeasible_sids = set()
    if os.path.exists(uf_csv):
        df_uf = pd.read_csv(uf_csv)
        feasible_sids = set(df_uf['scenario_id'].unique())
        infeasible_sids = set(range(1, 25)) - feasible_sids

    for sid, v in scenarios_agg.items():
        color = COLORS['grid_buy'] if sid in infeasible_sids else COLORS['alkel']
        marker = 'x' if sid in infeasible_sids else 'o'
        size = 80 if sid in infeasible_sids else 50
        ax.scatter(v['wind'], v['solar'], c=color, s=size, marker=marker, alpha=0.7,
                   edgecolors='black', linewidth=0.3)
        ax.annotate(str(sid), (v['wind'], v['solar']), fontsize=6,
                    textcoords="offset points", xytext=(4, 4))

    ax.set_xlabel('Daily Wind Energy (MWh)')
    ax.set_ylabel('Daily Solar Energy (MWh)')
    ax.grid(True, alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['alkel'],
               markersize=8, label='Off-grid feasible'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor=COLORS['grid_buy'],
               markersize=8, label='Off-grid infeasible'),
    ]
    ax.legend(handles=legend_elements, fontsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_resource_map.png'))
    plt.close()
    print("  Resource map saved.")


if __name__ == '__main__':
    print("Generating supplemental figures...")
    fig_uf_histogram()
    fig_nh3_vs_storage()
    fig_cost_vs_green()
    fig_grid_vs_offgrid_all()
    fig_wind_solar_resource_map()
    print(f"Done. Figures saved to {FIG_DIR}")
