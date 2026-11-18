"""
Sim static-analysis tests that do NOT import pandas or scipy.
Verifies: schema definitions, cell count, cell economics, leakage guard, seed utility.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
from pathlib import Path
import numpy as np

results = []

def check(name, val, expected=True):
    ok = bool(val) == bool(expected) if isinstance(expected, bool) else abs(float(val)-float(expected)) <= 1e-9
    status = "PASS" if ok else "FAIL"
    results.append(f"{status}  {name}")
    if not ok:
        results.append(f"       got={val!r}, expected={expected!r}")
    return ok

def check_range(name, val, lo, hi):
    ok = lo <= float(val) <= hi
    status = "PASS" if ok else "FAIL"
    results.append(f"{status}  {name}: {val} in [{lo}, {hi}]")
    return ok

print("=" * 60)
print("Sim static + seed tests (no pandas/scipy)")
print("=" * 60)

# ---- 1. Schema definitions ----
from sim.schema import SCHEMAS
check("search_event schema has required cols",
      all(c in SCHEMAS["search_event"] for c in ["search_id","ts","agent_id","as_of_ts","destination_city"]))
check("quote schema has required cols",
      all(c in SCHEMAS["quote"] for c in ["search_id","hotel_id","rate_plan","is_sold_out","rooms_left","as_of_ts"]))
check("booking schema has required cols",
      all(c in SCHEMAS["booking"] for c in ["booking_id","search_id","checkin","margin","lead_time_days","as_of_ts"]))
check("cancellation schema has required cols",
      all(c in SCHEMAS["cancellation"] for c in ["booking_id","ts_cancelled","days_before_checkin","as_of_ts"]))
check("inventory_commit schema has required cols",
      all(c in SCHEMAS["inventory_commit"] for c in
          ["commit_id","units","unit_net_cost","cash_prepaid","refund_frac","units_sold","units_spoiled","as_of_ts"]))
check("external_signal schema has required cols",
      all(c in SCHEMAS["external_signal"] for c in ["date","geo_id","signal_type","value","as_of_ts"]))
check("6 required tables in schema", len(SCHEMAS) == 6)

# ---- 2. Candidate cells ----
from sim.cells import get_candidate_cells, CANDIDATE_CELLS
cells = get_candidate_cells()
check_range("cell count 20-40", len(cells), 20, 40)

# Anchor cells present
ids = [c.cell_id for c in cells]
check("DXB-NYE anchor cell present", "DXB-NYE" in ids)
check("DXB-DEC anchor cell present", "DXB-DEC" in ids)
check("AIR-BOMDXB anchor cell present", "AIR-BOMDXB" in ids)

# DXB-NYE economics match spec exactly
nye = next(c for c in cells if c.cell_id == "DXB-NYE")
check("DXB-NYE p=21000",    nye.p == 21000)
check("DXB-NYE c_d=19845",  nye.c_d == 19845)
check("DXB-NYE c_b=13495",  nye.c_b == 13495)
check("DXB-NYE k=13495",    nye.k == 13495)
check("DXB-NYE r=0.0",      nye.r == 0.0)
check("DXB-NYE a=0.55",     abs(nye.a - 0.55) < 1e-9)
check("DXB-NYE g=900",      nye.g == 900)
check("DXB-NYE contract=hard_buyout", nye.contract_type == "hard_buyout")

# DXB-DEC economics match spec
dec = next(c for c in cells if c.cell_id == "DXB-DEC")
check("DXB-DEC p=14500",   dec.p == 14500)
check("DXB-DEC c_b=9592",  dec.c_b == 9592)
check("DXB-DEC r=0.70",    abs(dec.r - 0.70) < 1e-9)
check("DXB-DEC a=0.78",    abs(dec.a - 0.78) < 1e-9)
check("DXB-DEC contract=soft_release", dec.contract_type == "soft_release")

# free-allotment cells have k=0, r=1
free_cells = [c for c in cells if c.contract_type == "free_allotment"]
check("at least 2 free-allotment cells", len(free_cells) >= 2)
for fc in free_cells:
    check(f"{fc.cell_id} k=0", fc.k == 0.0)
    check(f"{fc.cell_id} r=1.0", abs(fc.r - 1.0) < 1e-9)

# cell groups defined
groups = set(c.region_group for c in cells)
check("multiple region groups (shocks correlated by group)", len(groups) >= 3)
results.append(f"       region groups: {sorted(groups)}")

# product types
types = set(c.product_type for c in cells)
check("hotel cells present", "hotel" in types)
check("air cells present",   "air"   in types)

# ---- 3. Seed utility ----
from config.settings import stable_label_seed, get_rng

# Same (master_seed, label) → same seed
s1 = stable_label_seed(42, "test")
s2 = stable_label_seed(42, "test")
check("stable_label_seed deterministic", s1 == s2)

# Different labels → different seeds
s3 = stable_label_seed(42, "other")
check("different labels → different seeds", s1 != s3)

# Different master seeds → different
s4 = stable_label_seed(99, "test")
check("different master seeds → different", s1 != s4)

# get_rng produces reproducible streams
rng_a = get_rng(42, "stream_x")
rng_b = get_rng(42, "stream_x")
va = rng_a.normal(size=5)
vb = rng_b.normal(size=5)
check("get_rng same seed/label → identical stream", np.allclose(va, vb))

# Different labels → independent streams
rng_c = get_rng(42, "stream_y")
vc = rng_c.normal(size=5)
check("get_rng different labels → independent", not np.allclose(va, vc))

# ---- 4. Assumptions YAML loads and has required keys ----
from config.settings import load_assumptions
cfg = load_assumptions()
check("carry annual_rate=0.12", abs(cfg.carry.annual_rate.value - 0.12) < 1e-9)
check("carry lock_days=120",    cfg.carry.lock_days.value == 120)
check("market dynamic_margin=0.055", abs(cfg.market.dynamic_margin.value - 0.055) < 1e-9)
check("risk alpha=0.95", abs(cfg.risk.alpha.value - 0.95) < 1e-9)
check("risk beta=0.5",   abs(cfg.risk.beta.value - 0.5)  < 1e-9)
check("4 contract types", len(cfg.contracts) == 4)
check("free_allotment k=0", cfg.contracts["free_allotment"].k.value == 0.0)
check("hard_buyout r=0",    cfg.contracts["hard_buyout"].r.value == 0.0)
check("soft_release r=0.7", abs(cfg.contracts["soft_release"].r.value - 0.7) < 1e-9)

# ---- 5. Leakage guard ----
root = Path(__file__).resolve().parent
offenders = []
for pkg in ("forecast", "optimizer"):
    pkg_dir = root / pkg
    for py in pkg_dir.rglob("*.py"):
        text = py.read_text(encoding="utf-8", errors="ignore")
        if "ground_truth" in text:
            offenders.append(str(py.relative_to(root)))
check("no ground_truth in forecast/ or optimizer/", len(offenders) == 0)
if offenders:
    results.append(f"  offenders: {offenders}")

# ---- 6. Doc files required by spec ----
doc_files = [
    "docs/architecture.md", "docs/assumptions.md", "docs/limitations.md",
    "docs/deck_outline.md", "docs/video_script.md", "docs/demo_script.md",
    "docs/baseline_metrics.md",
]
for f in doc_files:
    check(f"doc file exists: {f}", (root / f).exists())

# ---- Summary ----
passed = sum(1 for r in results if r.startswith("PASS"))
failed = sum(1 for r in results if r.startswith("FAIL"))
errors = sum(1 for r in results if r.startswith("ERROR"))
summary = f"\nResults: {passed} passed, {failed} failed, {errors} errors\n"

print()
for r in results:
    print(r)
print(summary)

out_path = root / "docs" / "sim_static_test_results.txt"
out_path.write_text("\n".join(results) + summary, encoding="utf-8")
print(f"Results written to {out_path}")
sys.exit(0 if (failed == 0 and errors == 0) else 1)
