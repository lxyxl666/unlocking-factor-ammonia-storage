"""
Nature-style figure redraw — wind-solar-ammonia-battery paper.
All figures use STRICT blue-gray palette. Max 4 panels per figure.
Includes 2 standalone 3D figures.

Figures generated:
  Fig 1 (1 panel):  Renewable profiles (2×2 grid)
  Fig 2 (2 panels): Efficiency curves + Resource map
  Fig 3 (2 panels): Dispatch stack + ALKEL:PEMEL ratio heatmap
  Fig 4 (2 panels): Cost composition + Grid vs off-grid all scenarios
  Fig 5 (4 panels): Unlocking Factor core evidence
  Fig 6 (1 panel):  3D — UF surface across (scenario x storage)
  Fig 7 (1 panel):  3D — NH3 production landscape
  Fig 8 (4 panels): Economics, sensitivity & robustness
"""

import pandas as pd
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config import *

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D

# ══════════════════════════════════════════════════════════════════════
# STRICT BLUE-GRAY PALETTE — no green, red, orange, yellow, purple
# ══════════════════════════════════════════════════════════════════════
BLUE = {
    "navy":    "#1B2D45",
    "dark":    "#253D5B",
    "steel":   "#325C8A",
    "mid":     "#4B7FB5",
    "sky":     "#72A3CE",
    "light":   "#9DC3E0",
    "pale":    "#C5DBEE",
    "ice":     "#E2EDF6",
}

GRAY = {
    "charcoal": "#32363B",
    "dark":     "#555A60",
    "mid":      "#7D8288",
    "light":    "#A5AAB0",
    "pale":     "#C8CCD0",
    "ice":      "#E4E6E8",
    "bg":       "#F4F5F6",
}

# Data series color assignment (all blue/gray)
C = {
    "wind":       BLUE["steel"],
    "solar":      BLUE["sky"],
    "alkel":      BLUE["navy"],
    "pemel":      BLUE["mid"],
    "nh3":        GRAY["charcoal"],
    "load":       GRAY["mid"],
    "grid_buy":   GRAY["dark"],
    "grid_sell":  BLUE["light"],
    "battery":    GRAY["light"],
    "curtail":    GRAY["pale"],
    "uf_high":    BLUE["dark"],
    "uf_mid":     BLUE["mid"],
    "uf_low":     BLUE["pale"],
    "edge":       GRAY["charcoal"],
    "text":       GRAY["dark"],
    "zero_line":  GRAY["mid"],
    "bg_panel":   GRAY["bg"],
    "pos":        GRAY["dark"],
    "neg":        BLUE["steel"],
}

# ── Matplotlib RC ──────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.6,
    "axes.edgecolor": GRAY["charcoal"],
    "xtick.color": GRAY["dark"],
    "ytick.color": GRAY["dark"],
    "text.color": GRAY["charcoal"],
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "grid.color": GRAY["pale"],
    "grid.alpha": 0.5,
    "grid.linewidth": 0.4,
})

OUT_DIR_FIG = os.path.join(OUT_DIR, "figures")
os.makedirs(OUT_DIR_FIG, exist_ok=True)

# ── Data ────────────────────────────────────────────────────────────
df_sc = pd.read_csv(os.path.join(DATA_DIR, SCENARIOS_FILE))
uf_csv = os.path.join(OUT_DIR, "unlocking_factor_all.csv")
df_uf = pd.read_csv(uf_csv) if os.path.exists(uf_csv) else None

dual_path = os.path.join(OUT_DIR, "dual_electrolyzer_results.json")
with open(dual_path) as f:
    dual_data = json.load(f)
df_dual = pd.DataFrame(dual_data.get("all_results", []))

rob_path = os.path.join(OUT_DIR, "robustness_check.json")
rob_data = json.load(open(rob_path)) if os.path.exists(rob_path) else {}


def load_json(name):
    p = os.path.join(OUT_DIR, name)
    return json.load(open(p)) if os.path.exists(p) else None


def load_fine_grid(sid):
    p = os.path.join(OUT_DIR, f"S{sid}_fine_grid.json")
    return json.load(open(p)) if os.path.exists(p) else None


def save_pub(fig, name):
    fig.savefig(os.path.join(OUT_DIR_FIG, f"{name}.svg"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR_FIG, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR_FIG, f"{name}.tiff"), dpi=600, bbox_inches="tight")
    print(f"  {name} saved")


def panel_label(ax, label, x=-0.10, y=1.04):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="left", color=GRAY["charcoal"])


def panel_label_3d(ax, label, x=-0.02, y=1.02):
    ax.text2D(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold",
              va="bottom", ha="left", color=GRAY["charcoal"])


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 1: Renewable Profiles (1 panel, 2×2 grid)                  ║
# ╚═════════════════════════════════════════════════════════════════════╝

def fig1():
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 3.2))
    fig.subplots_adjust(hspace=0.28, wspace=0.22)

    scenarios = [1, 6, 17, 24]
    titles = ["S1: High Wind, High Solar", "S6: High Wind, Low Solar",
              "S17: Mid Wind, High Solar", "S24: Low Wind, Low Solar"]
    hours = np.arange(24)

    for i, (sid, title) in enumerate(zip(scenarios, titles)):
        axi = axes[i // 2, i % 2]
        dfs = df_sc[df_sc["scenario_id"] == sid].sort_values("hour")
        ymax = (dfs["wind_MW"] + dfs["solar_MW"]).max() * 1.15
        axi.fill_between(hours, 0, dfs["wind_MW"], alpha=0.55, color=C["wind"])
        axi.fill_between(hours, dfs["wind_MW"], dfs["wind_MW"] + dfs["solar_MW"],
                         alpha=0.45, color=C["solar"])
        axi.plot(hours, dfs["load_MW"], color=C["load"], linewidth=1.0, linestyle="--")
        axi.set_title(title, fontsize=6.5, pad=3, color=GRAY["charcoal"])
        axi.set_xlim(0, 23)
        axi.set_ylim(0, ymax)
        axi.tick_params(labelsize=5)
        if i >= 2:
            axi.set_xlabel("Hour", fontsize=6)
        if i % 2 == 0:
            axi.set_ylabel("Power (MW)", fontsize=6)

    leg_handles = [
        Patch(facecolor=C["wind"], alpha=0.55, label="Wind"),
        Patch(facecolor=C["solar"], alpha=0.45, label="Solar"),
        Line2D([0], [0], color=C["load"], linewidth=1.0, linestyle="--", label="Load"),
    ]
    fig.legend(handles=leg_handles, fontsize=6.5, handlelength=1.2, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.02), frameon=False)

    save_pub(fig, "fig1_renewable_profiles")
    plt.close()
    print("Fig 1 done.")


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 2: Efficiency Curves & Resource Map (2 panels)             ║
# ╚═════════════════════════════════════════════════════════════════════╝

def fig2():
    fig = plt.figure(figsize=(7.2, 3.3))
    gs = GridSpec(1, 2, figure=fig, wspace=0.38)

    # (a) Part-load efficiency curves
    ax_a = fig.add_subplot(gs[0, 0])
    r_vals = np.linspace(0.05, 1.0, 200)
    alk_bp = np.array(ALKEL_EFF_BREAKPOINTS)
    pem_bp = np.array(PEMEL_EFF_BREAKPOINTS)
    alk_eff_pu = np.interp(r_vals, alk_bp[:, 0], alk_bp[:, 1])
    pem_eff_pu = np.interp(r_vals, pem_bp[:, 0], pem_bp[:, 1])

    ax_a.plot(r_vals * 100, alk_eff_pu * 70, color=C["alkel"], linewidth=1.6,
              label="ALKEL (nom. 70%)")
    ax_a.plot(r_vals * 100, pem_eff_pu * 80, color=C["pemel"], linewidth=1.6,
              label="PEMEL (nom. 80%)")
    ax_a.scatter(alk_bp[:, 0] * 100, alk_bp[:, 1] * 70, c=C["alkel"], s=14,
                 zorder=5, edgecolors="white", linewidth=0.3)
    ax_a.scatter(pem_bp[:, 0] * 100, pem_bp[:, 1] * 80, c=C["pemel"], s=14,
                 marker="s", zorder=5, edgecolors="white", linewidth=0.3)
    # Annotate min loads
    ax_a.axvline(x=15, color=C["alkel"], linestyle="--", linewidth=0.5, alpha=0.5)
    ax_a.text(15.5, 42, "ALKEL min 15%", fontsize=5, color=C["alkel"], rotation=90)
    ax_a.axvline(x=5, color=C["pemel"], linestyle="--", linewidth=0.5, alpha=0.5)
    ax_a.text(5.5, 68, "PEMEL min 5%", fontsize=5, color=C["pemel"], rotation=90)
    ax_a.set_xlabel("Load Factor (%)")
    ax_a.set_ylabel("Efficiency (%)")
    ax_a.legend(fontsize=6, handlelength=1.2, loc="lower right")
    ax_a.set_xlim(0, 105)
    ax_a.set_ylim(38, 85)
    panel_label(ax_a, "a")

    # (b) Wind-solar resource map
    ax_b = fig.add_subplot(gs[0, 1])
    scenario_agg = {}
    for sid in sorted(df_sc["scenario_id"].unique()):
        dfs = df_sc[df_sc["scenario_id"] == sid]
        scenario_agg[sid] = (dfs["wind_MW"].sum(), dfs["solar_MW"].sum())

    infeasible = set()
    if df_uf is not None:
        feasible = set(df_uf["scenario_id"].unique())
        infeasible = set(range(1, 25)) - feasible

    for sid, (w, s) in scenario_agg.items():
        if sid in infeasible:
            ax_b.scatter(w, s, c=GRAY["pale"], s=40, alpha=0.8, marker="x",
                         linewidth=0.8)
        else:
            ax_b.scatter(w, s, c=C["wind"], s=30, alpha=0.7, marker="o",
                         edgecolors=GRAY["charcoal"], linewidth=0.3)
    ax_b.set_xlabel("Daily Wind Energy (MWh)")
    ax_b.set_ylabel("Daily Solar Energy (MWh)")
    # Legend
    leg_b_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C["wind"],
               markersize=7, linestyle="", label="Off-grid feasible"),
        Line2D([0], [0], marker="x", color=GRAY["pale"], markersize=7,
               markeredgewidth=0.8, linestyle="", label="Off-grid infeasible"),
    ]
    ax_b.legend(handles=leg_b_elements, fontsize=6, handlelength=1.2, ncol=2,
               loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    panel_label(ax_b, "b")

    save_pub(fig, "fig2_efficiency_resource_map")
    plt.close()
    print("Fig 2 done.")


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 3: Dual-Electrolyzer Dispatch (2 panels)                   ║
# ╚═════════════════════════════════════════════════════════════════════╝

def fig3():
    fig = plt.figure(figsize=(7.2, 5.0))
    gs = GridSpec(2, 1, figure=fig, hspace=0.38, height_ratios=[1, 0.9])

    s1_rows = df_dual[(df_dual["scenario_id"] == 1) & (df_dual["production"] == 36)]
    if len(s1_rows) == 0:
        print("  Fig 3: No S1 data, skipping")
        plt.close()
        return
    s1 = s1_rows.iloc[0]
    hours = np.arange(24)

    # (a) Hourly dispatch stack
    ax_a = fig.add_subplot(gs[0, 0])
    dfs1 = df_sc[df_sc["scenario_id"] == 1].sort_values("hour")
    Pw, Ps, Pl = dfs1["wind_MW"].values, dfs1["solar_MW"].values, dfs1["load_MW"].values
    alk_pwr = np.array(s1["alkel_ratio"]) * ALKEL_P_RATED
    pem_pwr = np.array(s1["pemel_ratio"]) * PEMEL_P_RATED
    nh3_pwr = np.array(s1["nh3_ratio"]) * NH3_P_RATED

    ax_a.fill_between(hours, 0, Pw, alpha=0.35, color=C["wind"])
    ax_a.fill_between(hours, Pw, Pw + Ps, alpha=0.3, color=C["solar"])
    ax_a.fill_between(hours, Pw + Ps, Pw + Ps + alk_pwr, alpha=0.55, color=C["alkel"])
    ax_a.fill_between(hours, Pw + Ps + alk_pwr,
                      Pw + Ps + alk_pwr + pem_pwr, alpha=0.55, color=C["pemel"])
    ax_a.fill_between(hours, Pw + Ps + alk_pwr + pem_pwr,
                      Pw + Ps + alk_pwr + pem_pwr + nh3_pwr, alpha=0.4, color=C["nh3"])
    ax_a.plot(hours, Pl, color=C["load"], linewidth=1.0, linestyle="--")

    ax_a.set_ylabel("Power (MW)")
    ax_a.set_xlim(0, 23)
    ax_a.set_title("S1: High Wind, High Solar — Grid-connected 36 t/d", fontsize=7.5,
                   pad=4, color=GRAY["charcoal"])
    leg_a = [
        Patch(facecolor=C["wind"], alpha=0.5, label="Wind"),
        Patch(facecolor=C["solar"], alpha=0.5, label="Solar"),
        Patch(facecolor=C["alkel"], alpha=0.7, label="ALKEL"),
        Patch(facecolor=C["pemel"], alpha=0.7, label="PEMEL"),
        Patch(facecolor=C["nh3"], alpha=0.6, label="NH$_3$ Synth"),
        Line2D([0], [0], color=C["load"], linewidth=1.0, linestyle="--", label="Load"),
    ]
    ax_a.legend(handles=leg_a, fontsize=5.5, ncol=6, handlelength=1.0,
               loc="lower center", bbox_to_anchor=(0.5, -0.28))
    panel_label(ax_a, "a")

    # (b) ALKEL:PEMEL ratio heatmap
    ax_b = fig.add_subplot(gs[1, 0])
    scenario_ids = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    ratio_mat = np.zeros((len(scenario_ids), 24))
    for i, sid in enumerate(scenario_ids):
        rows = df_dual[(df_dual["scenario_id"] == sid) & (df_dual["production"] == 36)]
        if len(rows) > 0:
            r = rows.iloc[0]
            for t in range(24):
                a = r["alkel_ratio"][t]
                p = r["pemel_ratio"][t]
                ratio_mat[i, t] = a / (a + p) if (a + p) > 0 else 0.5

    im = ax_b.imshow(ratio_mat, aspect="auto", cmap="Blues", vmin=0, vmax=1,
                     origin="lower", interpolation="bilinear")
    ax_b.set_xticks(range(0, 24, 4))
    ax_b.set_xticklabels([str(h) for h in range(0, 24, 4)])
    ax_b.set_yticks(range(len(scenario_ids)))
    ax_b.set_yticklabels([f"S{sid}" for sid in scenario_ids], fontsize=5.5)
    ax_b.set_xlabel("Hour")
    cbar = plt.colorbar(im, ax=ax_b, shrink=0.85)
    cbar.set_label("ALKEL / (ALKEL+PEMEL)", fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    panel_label(ax_b, "b")

    save_pub(fig, "fig3_dispatch_ratio")
    plt.close()
    print("Fig 3 done.")


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 4: Economic Analysis (2 panels)                            ║
# ╚═════════════════════════════════════════════════════════════════════╝

def fig4():
    fig = plt.figure(figsize=(7.2, 3.3))
    gs = GridSpec(1, 2, figure=fig, wspace=0.38)

    # Pre-compute S1 wind/solar totals from scenario data
    s1_sc = df_sc[df_sc["scenario_id"] == 1]
    s1_wind_MWh = s1_sc["wind_MW"].sum()
    s1_solar_MWh = s1_sc["solar_MW"].sum()
    STORAGE_COST_PER_MWH_DAY = 1000.0 / STORAGE_LIFE_DAYS  # yuan/MWh/day

    # (a) Cost composition — 3 strategies, stacked bars
    ax_a = fig.add_subplot(gs[0, 0])
    s1_rows = df_dual[(df_dual["scenario_id"] == 1) & (df_dual["production"] == 36)]
    strategies_data = []
    labels_c = []

    if len(s1_rows) > 0:
        strategies_data.append(s1_rows.iloc[0]["cost_breakdown"])
        labels_c.append("Grid\n36 t/d")

    if df_uf is not None:
        s1_uf0 = df_uf[(df_uf["scenario_id"] == 1) & (df_uf["storage_MWh"] == 0)]
        if len(s1_uf0) > 0:
            r0 = s1_uf0.iloc[0]
            wind_c0 = s1_wind_MWh * 1000 * WIND_LCOE
            pv_c0 = s1_solar_MWh * 1000 * PV_LCOE
            total_c0 = r0["ton_cost"] * r0["daily_nh3"]
            cb0 = {
                "wind_lcoe": wind_c0,
                "pv_lcoe": pv_c0,
                "om_total": total_c0 - wind_c0 - pv_c0,
                "grid_buy": 0,
                "grid_sell_rev": 0,
                "storage_daily": 0,
            }
            strategies_data.append(cb0)
            labels_c.append("Off-grid\n0 MWh")

        s1_uf10 = df_uf[(df_uf["scenario_id"] == 1) & (df_uf["storage_MWh"] == 10)]
        if len(s1_uf10) > 0:
            r10 = s1_uf10.iloc[0]
            wind_c10 = s1_wind_MWh * 1000 * WIND_LCOE
            pv_c10 = s1_solar_MWh * 1000 * PV_LCOE
            stor_c10 = 10 * STORAGE_COST_PER_MWH_DAY
            total_c10 = r10["ton_cost"] * r10["daily_nh3"]
            cb10 = {
                "wind_lcoe": wind_c10,
                "pv_lcoe": pv_c10,
                "om_total": total_c10 - wind_c10 - pv_c10 - stor_c10,
                "grid_buy": 0,
                "grid_sell_rev": 0,
                "storage_daily": stor_c10,
            }
            strategies_data.append(cb10)
            labels_c.append("Off-grid\n10 MWh")

    categories = ["wind_lcoe", "pv_lcoe", "om_total", "grid_buy", "grid_sell_rev", "storage_daily"]
    cat_colors = [C["wind"], C["solar"], GRAY["mid"], GRAY["dark"], BLUE["light"], BLUE["sky"]]
    x_c = np.arange(len(labels_c))
    width_c = 0.55
    bottoms = np.zeros(len(labels_c))

    for cat, ccolor in zip(categories, cat_colors):
        vals = [s.get(cat, 0) / 10000 for s in strategies_data]
        if cat == "grid_sell_rev":
            vals = [-abs(v) for v in vals]
        ax_a.bar(x_c, vals, width_c, bottom=bottoms, color=ccolor, alpha=0.78,
                 edgecolor="white", linewidth=0.3)
        bottoms += np.array(vals)

    ax_a.set_xticks(x_c)
    ax_a.set_xticklabels(labels_c, fontsize=6.5)
    ax_a.set_ylabel("Daily Cost (10k yuan)")
    # Legend
    leg_cats = [
        Patch(facecolor=C["wind"], alpha=0.8, label="Wind LCOE"),
        Patch(facecolor=C["solar"], alpha=0.8, label="PV LCOE"),
        Patch(facecolor=GRAY["mid"], alpha=0.8, label="O&M"),
        Patch(facecolor=GRAY["dark"], alpha=0.8, label="Grid Buy"),
        Patch(facecolor=BLUE["light"], alpha=0.8, label="Grid Sell (neg.)"),
        Patch(facecolor=BLUE["sky"], alpha=0.8, label="Storage"),
    ]
    ax_a.legend(handles=leg_cats, fontsize=5, ncol=3, handlelength=1.0,
               loc="upper center", bbox_to_anchor=(0.5, -0.12))
    panel_label(ax_a, "a")

    # (b) Grid vs off-grid across all scenarios
    ax_b = fig.add_subplot(gs[0, 1])
    if df_dual is not None and len(df_dual) > 0 and df_uf is not None:
        grid_best = df_dual.loc[df_dual.groupby("scenario_id")["ton_cost"].idxmin()]
        off_best = df_uf.loc[df_uf.groupby("scenario_id")["ton_cost"].idxmin()]
        common = sorted(set(grid_best["scenario_id"]) & set(off_best["scenario_id"]))
        x_d = np.arange(len(common))
        w_d = 0.35
        g_costs = [grid_best[grid_best["scenario_id"] == s]["ton_cost"].values[0] for s in common]
        o_costs = [off_best[off_best["scenario_id"] == s]["ton_cost"].values[0] for s in common]

        ax_b.bar(x_d - w_d / 2, g_costs, w_d, color=C["alkel"], alpha=0.7,
                 label="Grid-connected", edgecolor="white", linewidth=0.2)
        ax_b.bar(x_d + w_d / 2, o_costs, w_d, color=C["pemel"], alpha=0.7,
                 label="Off-grid (best storage)", edgecolor="white", linewidth=0.2)
        ax_b.set_xticks(x_d)
        ax_b.set_xticklabels([str(int(s)) for s in common], fontsize=5.5)
        ax_b.set_xlabel("Scenario")
        ax_b.set_ylabel("Ton Cost (yuan/t)")
        ax_b.legend(fontsize=6, handlelength=1.0,
                   loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    panel_label(ax_b, "b")

    save_pub(fig, "fig4_cost_comparison")
    plt.close()
    print("Fig 4 done.")


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 5: Unlocking Factor — Core Evidence (4 panels, 2x2)        ║
# ╚═════════════════════════════════════════════════════════════════════╝

def fig5():
    fig = plt.figure(figsize=(7.2, 6.0))
    gs = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    uf_scenarios = [1, 6, 17, 18, 20]
    uf_linestyles = {1: "-", 6: (0, (3, 2)), 17: "--", 18: "-.", 20: ":"}
    uf_colors = {1: BLUE["pale"], 6: BLUE["sky"], 17: BLUE["mid"], 18: BLUE["steel"], 20: BLUE["navy"]}
    uf_markers = {1: "o", 6: "v", 17: "D", 18: "s", 20: "^"}
    uf_labels = {
        1: "S1: High W, High S",
        6: "S6: High W, Low S",
        17: "S17: Mid W, High S",
        18: "S18: Mid W, Mid S",
        20: "S20: Mid W, Low S",
    }

    # (a) UF vs storage — key scenarios
    ax_a = fig.add_subplot(gs[0, 0])
    for sid in uf_scenarios:
        fg = load_fine_grid(sid)
        if fg is None:
            continue
        caps = [d["storage_MWh"] for d in fg if d["storage_MWh"] > 0]
        ufs = [d["unlocking_factor"] for d in fg if d["storage_MWh"] > 0]
        if not ufs:
            continue
        peak_i = max(range(len(ufs)), key=lambda i: ufs[i])
        ax_a.plot(caps, ufs, uf_linestyles[sid], color=uf_colors[sid],
                  linewidth=1.1, alpha=0.85)
        ax_a.scatter([caps[peak_i]], [ufs[peak_i]], color=uf_colors[sid],
                     s=35, marker=uf_markers[sid], zorder=5,
                     edgecolors="white", linewidth=0.3,
                     label=f"{uf_labels[sid]} (peak {ufs[peak_i]:.1f})")

    ax_a.axhline(y=1, color=GRAY["mid"], linestyle="--", linewidth=0.6, alpha=0.5)
    ax_a.set_xlabel("Storage Capacity (MWh)")
    ax_a.set_ylabel("Unlocking Factor (UF)")
    ax_a.legend(fontsize=5.5, handlelength=1.0, ncol=2,
               loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    ax_a.set_xlim(0, 60)
    panel_label(ax_a, "a")

    # (b) UF distribution at 10 MWh
    ax_b = fig.add_subplot(gs[0, 1])
    if df_uf is not None:
        df_10 = df_uf[df_uf["storage_MWh"] == 10]
        uf_vals = df_10["unlocking_factor"].dropna()
        ax_b.hist(uf_vals, bins=16, color=BLUE["mid"], alpha=0.65,
                  edgecolor="white", linewidth=0.3)
        ax_b.axvline(x=1.0, color=GRAY["mid"], linestyle="--", linewidth=0.7)
        ymax_b = ax_b.get_ylim()[1]
        ax_b.text(1.0, ymax_b * 0.98, "UF=1", fontsize=5.5,
                  color=GRAY["dark"], ha="left", va="top")
        mean_uf = uf_vals.mean()
        ax_b.axvline(x=mean_uf, color=BLUE["navy"], linestyle="-", linewidth=0.7)
        ax_b.text(mean_uf, ymax_b * 0.88,
                  f"mean={mean_uf:.2f}", fontsize=5.5, color=BLUE["navy"], ha="left", va="top")
        ax_b.set_xlabel("UF at 10 MWh")
        ax_b.set_ylabel("Number of Scenarios")
    panel_label(ax_b, "b")

    # (c) NH3 production vs storage — saturation
    ax_c = fig.add_subplot(gs[1, 0])
    if df_uf is not None:
        for sid in [1, 6, 17, 18, 20]:
            dfs = df_uf[df_uf["scenario_id"] == sid].sort_values("storage_MWh")
            ax_c.plot(dfs["storage_MWh"], dfs["daily_nh3"], "-",
                      color=uf_colors.get(sid, BLUE["mid"]),
                      linewidth=1.1, alpha=0.8, label=uf_labels.get(sid, f"S{sid}"))
        ax_c.set_xlabel("Storage Capacity (MWh)")
        ax_c.set_ylabel("Daily NH$_3$ (tons)")
        ax_c.legend(fontsize=5.5, handlelength=1.0, ncol=2,
                   loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
        ax_c.set_xlim(0, 100)
    panel_label(ax_c, "c")

    # (d) Peak UF by scenario (bar chart)
    ax_d = fig.add_subplot(gs[1, 1])
    if df_uf is not None:
        df_nz = df_uf[df_uf["storage_MWh"] > 0]
        scenario_ufs = df_nz.groupby("scenario_id")["unlocking_factor"].max()
        feasible = scenario_ufs.dropna()
        bar_colors = [BLUE["navy"] if v >= 2 else BLUE["mid"] if v >= 1 else BLUE["pale"]
                      for v in feasible.values]
        ax_d.bar(range(len(feasible)), feasible.values, color=bar_colors, alpha=0.75,
                 edgecolor="white", linewidth=0.2)
        ax_d.axhline(y=1.0, color=GRAY["mid"], linestyle="--", linewidth=0.6, alpha=0.5)
        ax_d.set_xlabel("Scenario ID")
        ax_d.set_ylabel("Peak UF")
        ax_d.set_xticks(range(0, len(feasible), 3))
        ax_d.set_xticklabels([str(int(feasible.index[i])) for i in range(0, len(feasible), 3)],
                             fontsize=5.5)
    panel_label(ax_d, "d")

    save_pub(fig, "fig5_unlocking_factor")
    plt.close()
    print("Fig 5 done.")


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 6: 3D — Unlocking Factor Surface                           ║
# ╚═════════════════════════════════════════════════════════════════════╝

def fig6_3d_uf_surface():
    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    if df_uf is not None:
        df_3d = df_uf[(df_uf["storage_MWh"] > 0) & (df_uf["storage_MWh"] <= 50)].copy()
        sids_3d = sorted(df_3d["scenario_id"].unique())
        caps_3d = sorted(df_3d["storage_MWh"].unique())
        X, Y = np.meshgrid(sids_3d, caps_3d)
        Z = np.full(X.shape, np.nan)
        for i, sid in enumerate(sids_3d):
            for j, cap in enumerate(caps_3d):
                row = df_3d[(df_3d["scenario_id"] == sid) & (df_3d["storage_MWh"] == cap)]
                if len(row) > 0:
                    Z[j, i] = row.iloc[0]["unlocking_factor"]

        surf = ax.plot_surface(X, Y, Z, cmap="Blues_r", alpha=0.88,
                               edgecolor="none", linewidth=0, antialiased=True,
                               vmin=0, vmax=10)

        ax.set_xlabel("Scenario ID", labelpad=8, color=GRAY["charcoal"])
        ax.set_ylabel("Storage (MWh)", labelpad=8, color=GRAY["charcoal"])
        ax.set_zlabel("Unlocking Factor", labelpad=6, color=GRAY["charcoal"])
        ax.view_init(elev=26, azim=-58)
        ax.tick_params(labelsize=6, pad=3, colors=GRAY["dark"])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor(GRAY["pale"])
        ax.yaxis.pane.set_edgecolor(GRAY["pale"])
        ax.zaxis.pane.set_edgecolor(GRAY["pale"])

        cbar = fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.10)
        cbar.set_label("UF", fontsize=7, color=GRAY["charcoal"])
        cbar.ax.tick_params(labelsize=5.5, colors=GRAY["dark"])

    ax.set_title("Unlocking Factor UF(scenario, storage)", fontsize=9,
                 fontweight="bold", color=GRAY["charcoal"], pad=12)
    panel_label_3d(ax, "a")

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.96)
    base = os.path.join(OUT_DIR_FIG, "fig6_3d_uf_surface")
    fig.savefig(f"{base}.svg")
    fig.savefig(f"{base}.pdf")
    fig.savefig(f"{base}.tiff", dpi=600)
    print("  fig6_3d_uf_surface saved")
    plt.close()
    print("Fig 6 done.")


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 7: 3D — NH3 Production Landscape                           ║
# ╚═════════════════════════════════════════════════════════════════════╝

def fig7_3d_nh3_landscape():
    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    if df_uf is not None:
        df_3d = df_uf[df_uf["storage_MWh"] <= 50].copy()
        sids_3d = sorted(df_3d["scenario_id"].unique())
        caps_3d = sorted(df_3d["storage_MWh"].unique())
        X, Y = np.meshgrid(sids_3d, caps_3d)
        Z = np.full(X.shape, np.nan)
        for i, sid in enumerate(sids_3d):
            for j, cap in enumerate(caps_3d):
                row = df_3d[(df_3d["scenario_id"] == sid) & (df_3d["storage_MWh"] == cap)]
                if len(row) > 0:
                    Z[j, i] = row.iloc[0]["daily_nh3"]

        surf = ax.plot_surface(X, Y, Z, cmap="Blues_r", alpha=0.88,
                               edgecolor="none", linewidth=0, antialiased=True)
        z_min = np.nanmin(Z)

        ax.set_xlabel("Scenario ID", labelpad=8, color=GRAY["charcoal"])
        ax.set_ylabel("Storage (MWh)", labelpad=8, color=GRAY["charcoal"])
        ax.set_zlabel("Daily NH$_3$ (tons)", labelpad=6, color=GRAY["charcoal"])
        ax.view_init(elev=26, azim=-58)
        ax.tick_params(labelsize=6, pad=3, colors=GRAY["dark"])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor(GRAY["pale"])
        ax.yaxis.pane.set_edgecolor(GRAY["pale"])
        ax.zaxis.pane.set_edgecolor(GRAY["pale"])

        cbar = fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.10)
        cbar.set_label("NH$_3$ (t/d)", fontsize=7, color=GRAY["charcoal"])
        cbar.ax.tick_params(labelsize=5.5, colors=GRAY["dark"])

    ax.set_title("NH$_3$ Production Landscape", fontsize=9,
                 fontweight="bold", color=GRAY["charcoal"], pad=12)
    panel_label_3d(ax, "a")

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.96)
    base = os.path.join(OUT_DIR_FIG, "fig7_3d_nh3_landscape")
    fig.savefig(f"{base}.svg")
    fig.savefig(f"{base}.pdf")
    fig.savefig(f"{base}.tiff", dpi=600)
    print("  fig7_3d_nh3_landscape saved")
    plt.close()
    print("Fig 7 done.")


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 8: Economics, Sensitivity & Robustness (4 panels, 2x2)     ║
# ╚═════════════════════════════════════════════════════════════════════╝

def fig8():
    fig = plt.figure(figsize=(7.2, 6.0))
    gs = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    # (a) TOU & Feed-in price sensitivity
    ax_a = fig.add_subplot(gs[0, 0])
    tou_items = [("TOU x0.6", -590), ("TOU x0.8", -296), ("TOU x1.0 (base)", 0),
                 ("TOU x1.2", 92), ("TOU x1.4", 95)]
    feedin_items = [("Feedin x0.5", 3106), ("Feedin x0.75", 1379), ("Feedin x1.0 (base)", 0),
                    ("Feedin x1.25", -1957), ("Feedin x1.5", -3814)]

    all_items = feedin_items + tou_items
    names = [x[0] for x in all_items]
    shifts = [x[1] for x in all_items]
    bar_colors = [C["neg"] if s < 0 else C["pos"] for s in shifts]

    ax_a.barh(names, shifts, color=bar_colors, alpha=0.72, edgecolor="white", linewidth=0.3)
    ax_a.axvline(x=0, color=GRAY["charcoal"], linewidth=0.6)
    ax_a.set_xlabel("Ton-Cost Change (yuan/t)")
    ax_a.tick_params(labelsize=5.5)
    panel_label(ax_a, "a")

    # (b) Cost vs green compliance scatter
    ax_b = fig.add_subplot(gs[0, 1])
    if df_dual is not None and len(df_dual) > 0:
        pc_colors = {0: GRAY["pale"], 1: BLUE["light"], 2: BLUE["sky"], 3: BLUE["steel"]}
        pc_sizes = {0: 6, 1: 8, 2: 10, 3: 14}
        pc_labels = {0: "0/3", 1: "1/3", 2: "2/3", 3: "3/3 (Full)"}
        for pc in sorted(df_dual["pass_count"].unique()):
            df_pc = df_dual[df_dual["pass_count"] == pc]
            ax_b.scatter(df_pc["green_ratio"] * 100, df_pc["ton_cost"],
                         c=pc_colors.get(int(pc), GRAY["mid"]),
                         s=pc_sizes.get(int(pc), 8),
                         label=pc_labels.get(int(pc), str(pc)),
                         alpha=0.55, edgecolors="none")
        ax_b.set_xlabel("Green Electricity Ratio (%)")
        ax_b.set_ylabel("Ton-NH$_3$ Cost (yuan/t)")
        ax_b.legend(fontsize=5.5, handlelength=1.0, title="Compliance",
                    title_fontsize=6,
                    loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4)
    panel_label(ax_b, "b")

    # (c) ALKEL startup analysis across scenarios
    ax_c = fig.add_subplot(gs[1, 0])
    if df_dual is not None and len(df_dual) > 0:
        df_36 = df_dual[df_dual["production"] == 36]
        sids_sorted = sorted(df_36["scenario_id"].unique())
        startups_alk = [df_36[df_36["scenario_id"] == s]["alkel_startups"].values[0]
                        if len(df_36[df_36["scenario_id"] == s]) > 0 else 0
                        for s in sids_sorted]
        startups_pem = [df_36[df_36["scenario_id"] == s]["pemel_startups"].values[0]
                        if len(df_36[df_36["scenario_id"] == s]) > 0 else 0
                        for s in sids_sorted]
        x_c = np.arange(len(sids_sorted))
        w_c = 0.35
        ax_c.bar(x_c - w_c / 2, startups_alk, w_c, color=C["alkel"], alpha=0.7,
                 label="ALKEL", edgecolor="white", linewidth=0.2)
        ax_c.bar(x_c + w_c / 2, startups_pem, w_c, color=C["pemel"], alpha=0.7,
                 label="PEMEL", edgecolor="white", linewidth=0.2)
        ax_c.set_xticks(x_c)
        ax_c.set_xticklabels([str(int(s)) for s in sids_sorted], fontsize=5)
        ax_c.set_xlabel("Scenario")
        ax_c.set_ylabel("Daily Startups")
        ax_c.legend(fontsize=6, handlelength=1.0,
                   loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    panel_label(ax_c, "c")

    # (d) Cross-location UF validation + robustness
    ax_d = fig.add_subplot(gs[1, 1])
    # Cross-location bar
    locs = ["Inner\nMongolia", "Jiuquan\nGansu"]
    uf_peaks = [9.27, 7.80]
    uf_second = [8.01, 6.79]
    x_d = np.arange(len(locs))
    w_d = 0.3
    ax_d.bar(x_d - w_d / 2, uf_peaks, w_d, color=BLUE["navy"], alpha=0.8,
             label="Peak UF", edgecolor="white", linewidth=0.2)
    ax_d.bar(x_d + w_d / 2, uf_second, w_d, color=BLUE["sky"], alpha=0.7,
             label="Second-highest UF", edgecolor="white", linewidth=0.2)
    for i, (v1, v2) in enumerate(zip(uf_peaks, uf_second)):
        ax_d.text(i - w_d / 2, v1 + 0.2, str(v1), ha="center", fontsize=6,
                  fontweight="bold", color=BLUE["navy"])
        ax_d.text(i + w_d / 2, v2 + 0.2, str(v2), ha="center", fontsize=6,
                  color=BLUE["sky"])
    ax_d.set_xticks(x_d)
    ax_d.set_xticklabels(locs, fontsize=6)
    ax_d.set_ylabel("Unlocking Factor")
    ax_d.legend(fontsize=5.5, handlelength=1.0, loc="upper right")
    ax_d.set_ylim(0, 11.5)
    panel_label(ax_d, "d")

    save_pub(fig, "fig8_economics_robustness")
    plt.close()
    print("Fig 8 done.")


# ══════════════════════════════════════════════════════════════════════
# ╔═════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 9: 3D Bar Charts — UF & Cost Landscape                     ║
# ╚═════════════════════════════════════════════════════════════════════╝

def fig9_3d_bars():
    fig = plt.figure(figsize=(9.0, 4.2))
    gs = GridSpec(1, 2, figure=fig, wspace=0.22,
                  left=0.06, right=0.82, top=0.92, bottom=0.10)

    # ── (a) 3D bar: Peak UF by scenario x storage ────────────────────
    ax_a = fig.add_subplot(gs[0, 0], projection="3d")
    bar_sids = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    bar_caps = [0, 5, 10, 20, 30, 50]
    xpos_a = np.arange(len(bar_sids))
    ypos_a = np.arange(len(bar_caps))
    xpos_a, ypos_a = np.meshgrid(xpos_a, ypos_a)
    xpos_a, ypos_a = xpos_a.ravel(), ypos_a.ravel()
    z_a = np.zeros_like(xpos_a, dtype=float)
    dz_a = np.zeros_like(xpos_a, dtype=float)

    for idx, sid in enumerate(bar_sids):
        for jdx, cap in enumerate(bar_caps):
            if cap == 0:
                val = 0.0
            else:
                row = df_uf[(df_uf["scenario_id"] == sid) & (df_uf["storage_MWh"] == cap)]
                val = row.iloc[0]["unlocking_factor"] if len(row) > 0 else 0.0
            flat = idx * len(bar_caps) + jdx
            z_a[flat] = 0.0
            dz_a[flat] = val

    dx = dy = 0.6
    # Color bars by UF value
    norm_a = plt.Normalize(0, max(dz_a.max(), 1))
    colors_a = plt.cm.Blues_r(norm_a(dz_a))
    # Ensure zero-value bars are visible with pale color
    for i in range(len(dz_a)):
        if dz_a[i] < 0.01:
            colors_a[i] = mpl.colors.to_rgba(GRAY["ice"], 0.5)

    ax_a.bar3d(xpos_a, ypos_a, z_a, dx, dy, dz_a, color=colors_a,
               alpha=0.85, edgecolor="white", linewidth=0.15)
    ax_a.set_xticks(range(len(bar_sids)))
    ax_a.set_xticklabels([str(s) for s in bar_sids], fontsize=5.5, color=GRAY["dark"])
    ax_a.set_yticks(range(len(bar_caps)))
    ax_a.set_yticklabels([str(c) for c in bar_caps], fontsize=5.5, color=GRAY["dark"])
    ax_a.set_xlabel("Scenario ID", labelpad=10, color=GRAY["charcoal"], fontsize=6.5)
    ax_a.set_ylabel("Storage (MWh)", labelpad=10, color=GRAY["charcoal"], fontsize=6.5)
    ax_a.set_zlabel("Unlocking Factor", labelpad=2, color=GRAY["charcoal"], fontsize=6.5)
    ax_a.view_init(elev=28, azim=-52)
    ax_a.tick_params(labelsize=5.5, pad=2, colors=GRAY["dark"])
    ax_a.xaxis.pane.fill = False
    ax_a.yaxis.pane.fill = False
    ax_a.zaxis.pane.fill = False
    ax_a.xaxis.pane.set_edgecolor(GRAY["pale"])
    ax_a.yaxis.pane.set_edgecolor(GRAY["pale"])
    ax_a.zaxis.pane.set_edgecolor(GRAY["pale"])
    ax_a.set_title("Peak UF by Scenario × Storage", fontsize=7.5,
                   fontweight="bold", color=GRAY["charcoal"], pad=10)
    panel_label_3d(ax_a, "a")

    # ── (b) 3D bar: Grid-connected ton-cost by scenario ──────────────
    ax_b = fig.add_subplot(gs[0, 1], projection="3d")
    sids_b = sorted(df_dual["scenario_id"].unique()) if df_dual is not None else bar_sids
    prods_b = [12, 24, 36, 48]  # NH3 production targets
    xpos_b = np.arange(len(sids_b))
    ypos_b = np.arange(len(prods_b))
    xpos_b, ypos_b = np.meshgrid(xpos_b, ypos_b)
    xpos_b, ypos_b = xpos_b.ravel(), ypos_b.ravel()
    dz_b = np.zeros_like(xpos_b, dtype=float)

    if df_dual is not None:
        for idx, sid in enumerate(sids_b):
            for jdx, prod in enumerate(prods_b):
                row = df_dual[(df_dual["scenario_id"] == sid) & (df_dual["production"] == prod)]
                val = row.iloc[0]["ton_cost"] if len(row) > 0 else 0.0
                dz_b[idx * len(prods_b) + jdx] = val

    dx_b = dy_b = 0.6
    norm_b = plt.Normalize(min(dz_b.max() * 0.5, dz_b[dz_b > 0].min()),
                           dz_b.max() if dz_b.max() > 0 else 1)
    colors_b = plt.cm.Blues_r(norm_b(dz_b))
    for i in range(len(dz_b)):
        if dz_b[i] < 1:
            colors_b[i] = mpl.colors.to_rgba(GRAY["ice"], 0.5)

    ax_b.bar3d(xpos_b, ypos_b, np.zeros_like(dz_b), dx_b, dy_b, dz_b,
               color=colors_b, alpha=0.82, edgecolor="white", linewidth=0.12)
    ax_b.set_xticks(range(0, len(sids_b), 3))
    ax_b.set_xticklabels([str(int(sids_b[i])) for i in range(0, len(sids_b), 3)],
                         fontsize=5, color=GRAY["dark"])
    ax_b.set_yticks(range(len(prods_b)))
    ax_b.set_yticklabels([str(p) for p in prods_b], fontsize=5.5, color=GRAY["dark"])
    ax_b.set_xlabel("Scenario ID", labelpad=10, color=GRAY["charcoal"], fontsize=6.5)
    ax_b.set_ylabel("NH$_3$ Target (t/d)", labelpad=10, color=GRAY["charcoal"], fontsize=6.5)
    ax_b.set_zlabel("Ton Cost", labelpad=2, color=GRAY["charcoal"], fontsize=6.5)
    ax_b.view_init(elev=28, azim=-52)
    ax_b.tick_params(labelsize=5.5, pad=2, colors=GRAY["dark"])
    ax_b.xaxis.pane.fill = False
    ax_b.yaxis.pane.fill = False
    ax_b.zaxis.pane.fill = False
    ax_b.xaxis.pane.set_edgecolor(GRAY["pale"])
    ax_b.yaxis.pane.set_edgecolor(GRAY["pale"])
    ax_b.zaxis.pane.set_edgecolor(GRAY["pale"])
    ax_b.set_title("Grid-Connected Ton-Cost by Scenario", fontsize=7.5,
                   fontweight="bold", color=GRAY["charcoal"], pad=10)
    panel_label_3d(ax_b, "b")
    base = os.path.join(OUT_DIR_FIG, "fig9_3d_bars")
    fig.savefig(f"{base}.svg", bbox_inches="tight", pad_inches=0.35)
    fig.savefig(f"{base}.pdf", bbox_inches="tight", pad_inches=0.35)
    fig.savefig(f"{base}.tiff", dpi=600, bbox_inches="tight", pad_inches=0.35)
    print("  fig9_3d_bars saved")
    plt.close()
    print("Fig 9 done.")


# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("REDRAWING ALL 9 FIGURES — Strict blue-gray palette")
    print(f"Output: {OUT_DIR_FIG}")
    print("=" * 60)

    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6_3d_uf_surface()
    fig7_3d_nh3_landscape()
    fig8()
    fig9_3d_bars()

    print(f"\nDone. {9} figures saved to {OUT_DIR_FIG}")
