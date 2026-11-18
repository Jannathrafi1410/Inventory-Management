"""
Math-only test runner that:
1. Sets BLAS thread counts to 1 BEFORE importing anything else.
2. Directly runs the pure-math validation cases for P3 (optimizer) anchors.
3. Writes results to docs/math_test_results.txt.
Run with: python run_math_tests.py
"""
import os
# Must be set before any BLAS library loads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import sys
import traceback
from pathlib import Path

results = []

def check(name, expr, expected=True, atol=1e-4):
    try:
        val = expr()
        ok = abs(float(val) - float(expected)) <= atol if not isinstance(expected, bool) else bool(val) == expected
        status = "PASS" if ok else "FAIL"
        results.append(f"{status}  {name}: got={val}, expected={expected}")
        return ok
    except Exception as e:
        results.append(f"ERROR {name}: {e}")
        traceback.print_exc()
        return False

print("=" * 60)
print("Pure-math test suite (P1/P3 anchors + special cases)")
print("=" * 60)

# ---- Import math modules (no LightGBM, no scipy.stats needed) ----
try:
    from optimizer.cell import Cell
    from optimizer.fractile import optimal_fractile
    from optimizer.profit import expected_profit
    from optimizer.waterfill import waterfill, _quantile_from_sorted
    import numpy as np
    results.append("PASS  core imports: Cell, optimal_fractile, expected_profit, waterfill")
except Exception as e:
    results.append(f"ERROR  core imports failed: {e}")
    traceback.print_exc()
    sys.exit(1)

print("Core imports OK")

# ---- Anchor: DXB-NYE hard buyout (before carry, lock_days=0) ----
cell_nye = Cell("DXB-NYE", 21000, 19845, 13495, 13495, 0.0, 0.55, 900, 120, "hard_buyout", "GCC")
check("DXB-NYE delta_m", lambda: cell_nye.delta_m(), 7274.75, atol=0.01)
check("DXB-NYE c_o",     lambda: cell_nye.c_o(),     13495.0,  atol=0.01)
check("DXB-NYE F(lock=0)",
      lambda: optimal_fractile(cell_nye, lock_days=0, r_annual=0.12),
      0.3502, atol=5e-4)

# ---- Anchor: DXB-DEC soft release (before carry, lock_days=0) ----
cell_dec = Cell("DXB-DEC", 14500, 13702, 9592, 9592, 0.70, 0.78, 0, 150, "soft_release", "GCC")
check("DXB-DEC delta_m", lambda: cell_dec.delta_m(), 4285.56, atol=0.1)
check("DXB-DEC c_o",     lambda: cell_dec.c_o(),     2877.6,  atol=0.1)
check("DXB-DEC F(lock=0)",
      lambda: optimal_fractile(cell_dec, lock_days=0, r_annual=0.12),
      0.5982, atol=5e-4)

# ---- Special case: free allotment -> F=1.0 ----
cell_free = Cell("FREE", 12000, 11340, 9500, 0.0, 1.0, 0.82, 0, 60, "free_allotment", "GCC")
check("free allotment c_o=0",  lambda: cell_free.c_o(), 0.0)
check("free allotment F=1.0",  lambda: optimal_fractile(cell_free), 1.0)

# ---- Special case: a=1, c_d=c_b -> delta_m=0 -> F=0 ----
cell_noadv = Cell("NOADV", 100.0, 80.0, 80.0, 80.0, 0.0, 1.0, 0.0, 100, "hard_buyout", "G")
check("a=1,c_d=c_b delta_m=0", lambda: cell_noadv.delta_m(), 0.0)
check("a=1,c_d=c_b F=0",       lambda: optimal_fractile(cell_noadv), 0.0)

# ---- Special case: classic newsvendor (a=0,g=0,hard buyout) -> F=(p-c_b)/p ----
cell_nv = Cell("NV", 100.0, 90.0, 70.0, 70.0, 0.0, 0.0, 0.0, 100, "hard_buyout", "G")
check("newsvendor F=(p-c_b)/p",
      lambda: optimal_fractile(cell_nv, lock_days=0),
      (100.0 - 70.0) / 100.0, atol=1e-6)

# ---- Water-filling: spend never exceeds budget ----
rng = np.random.default_rng(7)
wf_cells = [
    Cell("A", 12000, 11340, 9000, 9000,  0.7, 0.8,  0, 100, "soft_release",   "G1"),
    Cell("B", 15000, 14175,11000, 11000, 0.0, 0.7, 200,  80, "hard_buyout",    "G1"),
    Cell("C",  9000,  8505, 7000,    0, 1.0, 0.85,  0,  50, "free_allotment", "G2"),
    Cell("D", 20000, 18900,15000, 15000, 0.7, 0.75,  0,  60, "soft_release",   "G2"),
]
demand_wf = np.column_stack([
    rng.lognormal(np.log(60), 0.35, 2000),
    rng.lognormal(np.log(40), 0.4,  2000),
    rng.lognormal(np.log(30), 0.3,  2000),
    rng.lognormal(np.log(35), 0.3,  2000),
])
budget = 800_000.0
res = waterfill(wf_cells, demand_wf, budget=budget)
check("waterfill spend<=budget",    lambda: res.spend, budget, atol=budget)  # any value <= budget passes
check("waterfill spend<=budget (exact)", lambda: res.spend <= budget + 1e-6, True)

# lambda=0 when budget non-binding
res_large = waterfill(wf_cells, demand_wf, budget=1e12)
check("lambda=0 non-binding budget", lambda: abs(res_large.lambda_star) < 1e-9, True)

# tiny budget: free cell (C, k=0) still gets allocated
res_tiny = waterfill(wf_cells, demand_wf, budget=50_000.0)
check("tiny budget free-cell still allocated", lambda: res_tiny.quantities[2] > 0, True)
check("tiny budget spend<=50k", lambda: res_tiny.spend <= 50_000.0 + 1e-6, True)

# ---- Brute-force vs fractile within 2% ----
rng2 = np.random.default_rng(1234)
demand_bf = rng2.lognormal(np.log(70), 0.35, 5000)
cell_bf = Cell("BF", 14500, 13702, 9592, 9592, 0.70, 0.78, 0, 200, "soft_release", "G")
f_bf = optimal_fractile(cell_bf)
sd = np.sort(demand_bf)
idx = max(0, min(int(np.ceil(f_bf * len(sd))) - 1, len(sd) - 1))
q_star = float(min(cell_bf.cap, sd[idx]))
grid = np.linspace(0, cell_bf.cap, 400)
profits = np.array([expected_profit(cell_bf, q, demand_bf) for q in grid])
q_grid = float(grid[np.argmax(profits)])
err = abs(q_star - q_grid) / max(abs(q_grid), 1.0)
check("brute-force vs fractile <=2%", lambda: err <= 0.02, True)

# ---- Naive newsvendor worse than correct fractile on DXB-DEC ----
rng3 = np.random.default_rng(99)
demand_comp = rng3.lognormal(np.log(110), 0.30, 4000)
f_correct = optimal_fractile(cell_dec)
sd3 = np.sort(demand_comp)
q_correct = float(min(cell_dec.cap, sd3[max(0, min(int(np.ceil(f_correct*len(sd3)))-1, len(sd3)-1))]))
f_naive = float(np.clip((cell_dec.p - cell_dec.c_b) / cell_dec.p, 0.0, 1.0))
q_naive  = float(min(cell_dec.cap, sd3[max(0, min(int(np.ceil(f_naive*len(sd3)))-1, len(sd3)-1))]))
prof_correct = expected_profit(cell_dec, q_correct, demand_comp)
prof_naive   = expected_profit(cell_dec, q_naive,   demand_comp)
check("correct_fractile > naive_newsvendor on DXB-DEC", lambda: prof_correct > prof_naive, True)

# ---- Leakage check: ground_truth not referenced in forecast/ or optimizer/ ----
root = Path(__file__).resolve().parent
offenders = []
for pkg in ("forecast", "optimizer"):
    pkg_dir = root / pkg
    if not pkg_dir.exists():
        continue
    for py in pkg_dir.rglob("*.py"):
        text = py.read_text(encoding="utf-8", errors="ignore")
        if "ground_truth" in text:
            offenders.append(str(py.relative_to(root)))
check("no ground_truth in forecast/ or optimizer/", lambda: len(offenders) == 0, True)
if offenders:
    results.append(f"  offenders: {offenders}")

# ---- Write results ----
out_path = root / "docs" / "math_test_results.txt"
out_path.parent.mkdir(exist_ok=True)

passed = sum(1 for r in results if r.startswith("PASS"))
failed = sum(1 for r in results if r.startswith("FAIL"))
errors = sum(1 for r in results if r.startswith("ERROR"))

summary = f"\nResults: {passed} passed, {failed} failed, {errors} errors\n"

print()
for r in results:
    print(r)
print(summary)

out_path.write_text("\n".join(results) + summary, encoding="utf-8")
print(f"Results written to {out_path}")

sys.exit(0 if (failed == 0 and errors == 0) else 1)
