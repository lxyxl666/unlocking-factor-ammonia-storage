"""Centralized parameters for dual-electrolyzer + storage MILP model.
All parameters cite sources where applicable.
"""
import os

# === Paths ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data_cleaned')
OUT_DIR = os.path.join(BASE_DIR, 'results', 'journal')
os.makedirs(OUT_DIR, exist_ok=True)

# Data file selection: set to 'real' for NASA POWER data, 'competition' for original
DATA_SOURCE = 'real'
if DATA_SOURCE == 'real':
    SCENARIOS_FILE = 'all_24_scenarios_real.csv'
else:
    SCENARIOS_FILE = 'all_24_scenarios.csv'

# === Equipment Rated Power (at 72 t/d capacity, alpha=2) ===
WIND_CAP_MW = 40       # Wind farm installed capacity
PV_CAP_MW = 64         # PV farm installed capacity
ALKEL_P_RATED = 20     # ALKEL rated power (MW)
PEMEL_P_RATED = 20     # PEMEL rated power (MW)
NH3_P_RATED = 1.5      # Ammonia synthesis rated power (MW)
LOAD_PEAK_MW = 6       # Conventional load peak
NH3_RATE = 3.0         # t NH3/h at full power (72 t/d)

# === Electrolyzer Characteristics (literature-based) ===

# ALKEL:  Almatec 20MW stack data + Gabrielli2018 review
# PEMEL:  Siemens Silyzer 300 data + Zhang2020 IEEE TSTE
ALKEL_H2_RATE = 280    # kg H2/h at rated power (20MW * 14 kg/MWh)
PEMEL_H2_RATE = 320    # kg H2/h at rated power (20MW * 16 kg/MWh)
# Verified: ALKEL 20MW * 1/50 kWh/kg * 0.70 = 280 kg/h; PEMEL 20MW * 1/50 * 0.80 = 320 kg/h
AMMONIA_H2_NEED = 600  # kg H2/h (0.2 kgH2/kgNH3 * 3000 kgNH3/h)

# Differentiated minimum loads
ALKEL_MIN_RATIO = 0.15  # [Sayed2023] ALKEL 15-20% min load for gas crossover safety
PEMEL_MIN_RATIO = 0.05  # [Buttler2018] PEMEL can operate at 5-10% min load

# Ramp rates (fraction of rated power per hour)
ALKEL_RAMP_MAX = 0.20   # [Sayed2023] ALKEL ~10-20%/min → 20%/h is conservative
PEMEL_RAMP_MAX = 1.00   # PEMEL effectively unconstrained at hourly resolution

# Startup costs (cold start energy penalty)
ALKEL_STARTUP_COST = 500.0  # yuan/start [Bertuccioli2014] 0.5-2h cold start at rated power
PEMEL_STARTUP_COST = 50.0   # yuan/start [Buttler2018] <5min cold start, negligible

# === NH3 Synthesis Operational Parameters ===
NH3_MIN_RATIO = 0.40       # Haber-Bosch min load (catalyst temperature maintenance) [Cheema2018]
NH3_STARTUP_COST = 2000.0  # yuan/start (catalyst bed heating, ~2-4 h energy penalty)

# === Part-load Efficiency (SOS2 piecewise linear, 4 segments) ===
# ALKEL: efficiency drops significantly at low load
#  eta(r) = eta_nom * piecewise_linear(r)
# Points: (r_min, eta_pu), ..., (1.0, eta_pu)
ALKEL_EFF_BREAKPOINTS = [
    (0.15, 0.60),   # 60% of nominal at 15% load
    (0.40, 0.82),   # 82% at 40% load
    (0.70, 0.94),   # 94% at 70% load
    (1.00, 1.00),   # 100% (nominal = 70% absolute) at full load
]

PEMEL_EFF_BREAKPOINTS = [
    (0.05, 0.90),   # 90% of nominal at 5% load (nearly flat)
    (0.30, 0.96),   # 96% at 30% load
    (0.60, 0.99),   # 99% at 60% load
    (1.00, 1.00),   # 100% (nominal = 80% absolute) at full load
]

# === O&M Costs ===
ALKEL_OM = 0.10   # yuan/kWh_input [IEA2024]
PEMEL_OM = 0.15   # yuan/kWh_input [IEA2024]
NH3_OM = 0.002    # yuan/kWh [competition data]

# === LCOE (levelized cost of energy) ===
WIND_LCOE = 0.15   # yuan/kWh [competition data, consistent with NEA 2024]
PV_LCOE = 0.12     # yuan/kWh

# === Grid Pricing ===
FEEDIN_PRICE = 0.3779  # yuan/kWh (coal benchmark)


def tou_price(h):
    """Time-of-use purchase price (yuan/kWh)."""
    if 10 <= h < 15 or 18 <= h < 21:
        return 0.8024  # Peak
    elif 7 <= h < 10 or 15 <= h < 18 or 21 <= h < 23:
        return 0.6074  # Standard
    else:
        return 0.3424  # Valley


# === Storage Parameters ===
STORAGE_INVESTMENT = 1000   # yuan/kWh installed [NEA2022]
STORAGE_OM = 0.01           # yuan/kWh dispatched
STORAGE_EFF_CHG = 0.90      # Round-trip: 0.90 * 0.90 = 0.81
STORAGE_EFF_DIS = 0.90
STORAGE_SELF_DISCHARGE = 0.002  # per hour
STORAGE_LIFE_YEARS = 15
STORAGE_LIFE_DAYS = STORAGE_LIFE_YEARS * 365

# NH3 reference value for economic dispatch (from Q3 grid-connected annual average)
NH3_REF_VALUE = 4138.32  # yuan/ton

# === Production Levels ===
PRODUCTION_LEVELS = [72, 63, 54, 45, 36]

# === Scenarios ===
# 6 wind scenarios x 4 solar scenarios = 24 combinations
# Each scenario represents 15 days for annual statistics
DAYS_PER_SCENARIO = 15
TOTAL_SCENARIOS = 24
T = 24  # time periods (hours)

# === Big-M Constant ===
M_BIG = 150  # MW, sufficient upper bound for power exchange
