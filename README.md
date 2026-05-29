# Small-Capacity Battery Storage as Production Enabler in Off-Grid Wind-Solar-Ammonia Systems

**Xiaoguang Xu, Xiaoyang Li** — School of Electrical Engineering, Anhui Polytechnic University

This repository contains the code, data, results, and manuscript for our MILP-based optimization study of dual-electrolyzer (ALKEL+PEMEL) wind-solar-ammonia systems with battery storage. The key contribution is the **Unlocking Factor (UF)** — a metric quantifying how small-capacity battery storage enables ammonia production that would otherwise be infeasible.

## Directory Structure

```
.
├── manuscript.tex                  # LaTeX manuscript (JRSE/AIP format)
├── manuscript.pdf                  # Compiled PDF
├── manuscriptNotes.bib             # Bibliography
├── fig1_system_architecture.drawio # Figure 1 source (draw.io)
├── README.md
├── Figures/
│   ├── fig1*.pdf/svg/tiff          # Figure 1: System architecture & renewable profiles
│   ├── fig2*.pdf/svg/tiff          # Figure 2: Efficiency curves & resource map
│   ├── fig3*.pdf/svg/tiff          # Figure 3: Dispatch ratio
│   ├── fig4*.pdf/svg/tiff          # Figure 4: Cost comparison
│   ├── fig5*.pdf/svg/tiff          # Figure 5: Unlocking factor
│   ├── fig6*.pdf/svg/tiff          # Figure 6: 3D UF surface
│   ├── fig7*.pdf/svg/tiff          # Figure 7: 3D NH3 production landscape
│   ├── fig8*.pdf/svg/tiff          # Figure 8: Economics & robustness
│   ├── fig9*.pdf/svg/tiff          # Figure 9: 3D bars
│   ├── fig_S1_sos2.tex/pdf         # Supplementary: SOS2 formulation
│   ├── fig_S2_uf_surface.pdf       # Supplementary: UF surface (all scenarios)
│   ├── fig_S3_nh3_landscape.pdf    # Supplementary: NH3 landscape
│   ├── table_S1.tex                # Supplementary: 24-scenario summary
│   ├── table_S2.tex                # Supplementary: Robustness check
│   └── generate_supplementary.py   # Script to regenerate S1-S3
├── code/
│   ├── config.py                   # Global parameters & constants
│   ├── dual_electrolyzer_milp.py   # Core MILP model (Pyomo + Gurobi)
│   ├── run_all_2025.py             # Main batch run script
│   ├── run_fine_grid.py            # Fine-grid storage sweep
│   ├── run_jiuquan_full.py         # Jiuquan (Gansu) validation run
│   ├── combined_optimization.py    # Grid-connected + off-grid optimizer
│   ├── storage_unlocking.py        # UF computation via bisection
│   ├── robustness_check.py         # +/-15% wind/solar perturbation
│   ├── nh3_constraints.py          # NH3 synthesis flexibility analysis
│   ├── battery_degradation.py      # Cycle-depth degradation analysis
│   ├── economic_sensitivity.py     # LCOA tornado chart
│   ├── run_worst_case_window.py    # Multi-day worst-case window
│   ├── redraw_all_figures_nature.py# Figure generation (Nature style)
│   ├── sensitivity_and_figures.py  # Sensitivity analysis & figures
│   ├── figures_supplement.py       # Supplementary figures
│   ├── download_real_data.py       # NASA POWER data downloader
│   ├── download_jiuquan.py         # Jiuquan data downloader
│   ├── debug_s5.py                 # Debug script for scenario S5
│   └── test_single.py              # Single-scenario test runner
├── data/
│   ├── all_24_scenarios.csv        # 24 wind-solar-load scenarios (8760h)
│   ├── all_24_scenarios_real.csv   # Real data variant
│   ├── nasa_power_2025_inner_mongolia.csv  # NASA POWER 2025 data
│   ├── nasa_power_2024_inner_mongolia.csv  # NASA POWER 2024 data
│   ├── nasa_power_2023_inner_mongolia.csv  # NASA POWER 2023 data
│   ├── wind_scenarios.csv          # Wind cluster profiles
│   ├── solar_scenarios.csv         # Solar cluster profiles
│   ├── wind_solar_typical.csv      # Typical wind+solar day
│   ├── load_actual.csv             # Actual load data (Inner Mongolia)
│   ├── equipment_params.csv        # Electrolyzer, NH3 plant parameters
│   ├── storage_ammonia_params.csv  # Battery & ammonia storage params
│   ├── tou_price.csv               # Time-of-use electricity price
│   ├── grid_feedin_price.csv       # Feed-in tariff
│   ├── data_source_metadata.json   # Data provenance
│   ├── all_24_scenarios_jiuquan_gansu.csv  # Jiuquan validation data
│   └── nasa_power_2023_jiuquan_gansu.csv   # Jiuquan NASA data
├── results/
│   ├── dual_electrolyzer_results.json       # Main optimization results
│   ├── unlocking_factor_results.json        # UF calculation results
│   ├── unlocking_factor_all.csv             # UF across all scenarios
│   ├── combined_optimization_results.json   # Grid + off-grid combined
│   ├── combined_optimization_all.csv        # Combined results (all params)
│   ├── robustness_check.json                # +/-15% perturbation results
│   ├── sensitivity_results.json             # Sensitivity analysis
│   ├── nh3_constraints_results.json         # NH3 flexibility results
│   ├── battery_degradation_results.json     # Battery degradation results
│   ├── economic_sensitivity_results.json    # LCOA sensitivity results
│   ├── worst_case_window_results.json       # Multi-day window results
│   ├── jiuquan_full_results.json            # Jiuquan validation
│   ├── uf_coarse_jiuquan_gansu.csv          # Jiuquan UF coarse sweep
│   ├── S1_fine_grid.json / S9 / S13 / S17 / S18 / S20  # Fine-grid results
│   └── ...
└──
```

## Requirements

- Python 3.10+
- Pyomo (MILP modeling)
- Gurobi 10+ with valid license (commercial solver)
- pandas, numpy, matplotlib, scipy
- LaTeX (TeX Live or MiKTeX) for manuscript compilation

## Reproducing Results

1. **Download data**: `python code/download_real_data.py`
2. **Generate scenarios**: `python code/run_all_2025.py`
3. **UF analysis**: `python code/storage_unlocking.py`
4. **Robustness check**: `python code/robustness_check.py`
5. **Generate figures**: `python code/redraw_all_figures_nature.py`
6. **Compile manuscript**: `pdflatex manuscript.tex && bibtex manuscript && pdflatex manuscript.tex && pdflatex manuscript.tex`

## Key Parameters

| Parameter | Value |
|-----------|-------|
| Location | Inner Mongolia, China (42°N, 113°E) |
| Wind capacity | Based on cluster (see Table S1) |
| Solar capacity | Based on cluster (see Table S1) |
| Electrolyzers | ALKEL (1 MW rated) + PEMEL (1 MW rated) |
| NH3 target | 36 t/d (grid-connected), max feasible (off-grid) |
| Storage range | 0–100 MWh |
| Battery cost | 1500 CNY/kWh |
| Time horizon | 24 h, hourly resolution |

## License

This repository accompanies a manuscript submitted to *Journal of Renewable and Sustainable Energy* (AIP Publishing). Code and data are provided for reproducibility purposes.
