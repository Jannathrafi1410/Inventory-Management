"""
Sim-layer tests: schema, censoring, margin, correlation, reproducibility, leakage guard.
Runs without scipy (no BLAS deadlock).
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys, hashlib, traceback
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd

results = []

def check(name, val, expected=True):
    ok = bool(val) == bool(expected) if isinstance(expected, bool) else abs(float(val)-float(expected)) < 1e-9
    status = "PASS" if ok else "FAIL"
    results.append(f"{status}  {name}")
    if not ok:
        results.append(f"       got={val}, expected={expected}")
    return ok

def check_range(name, val, lo, hi):
    ok = lo <= float(val) <= hi
    status = "PASS" if ok else "FAIL"
    results.append(f"{status}  {name}: {val:.4f} in [{lo},{hi}]")
    return ok

print("=" * 60)
print("Sim layer tests")
print("=" * 60)

from sim.generator import generate_synthetic_data
from sim.schema import SCHEMAS
from sim.cells import get_candidate_cells

# ---- 1. Generate data into temp dir ----
with tempfile.TemporaryDirectory() as tmp:
    tables = generate_synthetic_data(master_seed=123, out_dir=tmp)

    # 1a. Required tables and columns
    for table, cols in SCHEMAS.items():
        if table not in tables:
            results.append(f"FAIL  missing table: {table}")
            continue
        df = tables[table]
        for col in cols:
            if col not in df.columns:
                results.append(f"FAIL  {table} missing column {col}")
    check("ground_truth table present", "ground_truth" in tables)
    gt = tables["ground_truth"]
    for col in ["latent_demand","capacity","observed_sales","dynamic_margin","fallback_availability"]:
        check(f"ground_truth has column {col}", col in gt.columns)

    # 1b. Censoring: observed_sales <= capacity always
    check("observed_sales <= capacity (all rows)",
          (gt["observed_sales"] <= gt["capacity"]).all())
    check("observed_sales <= latent_demand (all rows)",
          (gt["observed_sales"] <= np.floor(gt["latent_demand"] + 1e-9)).all())

    # 1c. Dynamic margin 5.5% ± 0.3pp
    mean_margin = float(gt["dynamic_margin"].mean())
    check_range("mean dynamic margin 5.5% ±0.3pp", mean_margin, 0.052, 0.058)

    # 1d. Sell-out rate in sane range 10-45%
    sell_out_rate = float(gt["sell_out"].mean())
    check_range("sell-out rate 10-45%", sell_out_rate, 0.10, 0.45)

    # 1e. Within-group correlation > across-group correlation
    pivot = gt.pivot_table(index="date", columns="cell_id", values="latent_demand")
    cell_to_group = gt.drop_duplicates("cell_id").set_index("cell_id")["region_group"].to_dict()
    cells_list = list(pivot.columns)
    within, across = [], []
    for i in range(len(cells_list)):
        for j in range(i+1, len(cells_list)):
            a_id, b_id = cells_list[i], cells_list[j]
            s = pivot[[a_id, b_id]].dropna()
            if len(s) < 10:
                continue
            corr = float(s[a_id].corr(s[b_id]))
            if np.isnan(corr):
                continue
            if cell_to_group[a_id] == cell_to_group[b_id]:
                within.append(corr)
            else:
                across.append(corr)
    if within and across:
        wm, am = np.mean(within), np.mean(across)
        check("within-group corr > across-group + 0.05",
              wm > am + 0.05)
        results.append(f"       within={wm:.3f}, across={am:.3f}")
    else:
        results.append("FAIL  not enough pairs for correlation test")

    # 1f. Inventory commits censored to observed sales
    commits = tables["inventory_commit"]
    check("units_sold <= units in inventory_commit",
          (commits["units_sold"] <= commits["units"]).all())

    # 1g. Rooms_left >= 0 in quotes
    quotes = tables["quote"]
    check("rooms_left >= 0 in quote", (quotes["rooms_left"] >= 0).all())

# ---- 2. Byte-identical regeneration ----
def hash_tables(tables_dict):
    h = hashlib.sha256()
    for name in sorted(tables_dict.keys()):
        if name == "_out_dir": continue
        df = tables_dict[name]
        h.update(name.encode())
        h.update(df.to_csv(index=False).encode())
    return h.hexdigest()

with tempfile.TemporaryDirectory() as t1:
    with tempfile.TemporaryDirectory() as t2:
        t1_tables = generate_synthetic_data(master_seed=77, out_dir=t1)
        t2_tables = generate_synthetic_data(master_seed=77, out_dir=t2)
        check("byte-identical regeneration (same seed)",
              hash_tables(t1_tables) == hash_tables(t2_tables))
        t3_tables = generate_synthetic_data(master_seed=99, out_dir=t2)
        check("different seeds produce different output",
              hash_tables(t1_tables) != hash_tables(t3_tables))

# ---- 3. Leakage guard ----
root = Path(__file__).resolve().parent
offenders = []
for pkg in ("forecast", "optimizer"):
    pkg_dir = root / pkg
    for py in pkg_dir.rglob("*.py"):
        text = py.read_text(encoding="utf-8", errors="ignore")
        if "ground_truth" in text:
            offenders.append(str(py.relative_to(root)))
check("no ground_truth in forecast/ or optimizer/", len(offenders) == 0)

# ---- 4. Cell count in range 20-40 ----
all_cells = get_candidate_cells()
check_range("candidate cell count 20-40", len(all_cells), 20, 40)

# ---- Summary ----
passed = sum(1 for r in results if r.startswith("PASS"))
failed = sum(1 for r in results if r.startswith("FAIL"))
errors = sum(1 for r in results if r.startswith("ERROR"))
summary = f"\nResults: {passed} passed, {failed} failed, {errors} errors\n"
print()
for r in results:
    print(r)
print(summary)

out_path = root / "docs" / "sim_test_results.txt"
out_path.write_text("\n".join(results) + summary, encoding="utf-8")
print(f"Results written to {out_path}")
sys.exit(0 if (failed == 0 and errors == 0) else 1)
