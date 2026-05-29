"""Master pipeline: run all computations with 2025 data.
Sequence: dual EL -> UF coarse -> UF fine -> robustness -> combined -> figures
"""
import subprocess, sys, os, time

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPTS_DIR)

steps = [
    ("Inner Mongolia Dual EL", "dual_electrolyzer_milp.py"),
    ("Inner Mongolia UF Coarse", "storage_unlocking.py"),
    ("Inner Mongolia UF Fine", "run_fine_grid.py"),
    ("Jiuquan Full Pipeline", "run_jiuquan_full.py"),
    ("Combined Optimization (3 models)", "combined_optimization.py"),
    ("Robustness Check (+-15%)", "robustness_check.py"),
    ("Main Figures", "sensitivity_and_figures.py"),
    ("Supplemental Figures", "figures_supplement.py"),
]

total = len(steps)
t0 = time.time()

for i, (name, script) in enumerate(steps, 1):
    print(f"\n{'='*60}")
    print(f"[{i}/{total}] {name} ({script})")
    print(f"{'='*60}")
    ts = time.time()
    result = subprocess.run([sys.executable, script], capture_output=False,
                           timeout=7200)
    if result.returncode != 0:
        print(f"  WARNING: {name} exited with code {result.returncode}")
    elapsed = time.time() - ts
    print(f"  Done in {elapsed:.0f}s ({elapsed/60:.1f}min)")

total_time = time.time() - t0
print(f"\n{'='*60}")
print(f"ALL DONE in {total_time:.0f}s ({total_time/60:.1f}min)")
print(f"{'='*60}")
