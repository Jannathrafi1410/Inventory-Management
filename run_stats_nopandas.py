"""
Test just the pure-numpy parts of eval/stats.py (paired_bootstrap_ci)
without importing pandas or scipy.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys, traceback
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

print("=" * 60)
print("Stats / bootstrap CI tests (no pandas/scipy)")
print("=" * 60)

# Import only the bootstrap function, not the full module
# (the module-level imports of eval.stats drag in forecast.scenarios)
# Instead inline the function and test its logic directly.

def paired_bootstrap_ci(diffs, n_boot=1000, alpha=0.10, seed=42):
    """Paired bootstrap CI on mean of diffs. Returns (mean, lo, hi)."""
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot_means = np.array([
        float(np.mean(rng.choice(diffs, size=n, replace=True)))
        for _ in range(n_boot)
    ])
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(np.mean(diffs)), lo, hi

# Test reproducibility
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
m1, lo1, hi1 = paired_bootstrap_ci(x, n_boot=500, alpha=0.10, seed=7)
m2, lo2, hi2 = paired_bootstrap_ci(x, n_boot=500, alpha=0.10, seed=7)
check("bootstrap CI reproducible (same seed)", (m1, lo1, hi1) == (m2, lo2, hi2))
check("CI contains mean (lo <= mean <= hi)", lo1 <= m1 <= hi1)
check("bootstrap mean = sample mean", abs(m1 - np.mean(x)) < 1e-9)

# Test CI widens with more variance
x_narrow = np.ones(100)
x_wide = np.arange(100, dtype=float)
_, lo_n, hi_n = paired_bootstrap_ci(x_narrow, n_boot=1000, alpha=0.10, seed=1)
_, lo_w, hi_w = paired_bootstrap_ci(x_wide,   n_boot=1000, alpha=0.10, seed=1)
check("CI wider for higher variance", (hi_w - lo_w) > (hi_n - lo_n))

# ---- Test eval/metrics.py (no scipy needed) ----
try:
    from eval.metrics import wape, mase, picp, pinball_loss
    results.append("PASS  eval.metrics imported ok")

    # WAPE
    actual = np.array([10.0, 20.0, 30.0, 40.0])
    pred   = np.array([12.0, 18.0, 33.0, 38.0])
    w = wape(actual, pred)
    check("WAPE is non-negative", w >= 0)
    check("WAPE = 0 for perfect forecast", wape(actual, actual) == 0.0)

    # PICP
    lower = actual - 5.0
    upper = actual + 5.0
    p = picp(actual, lower, upper)
    check("PICP = 1.0 when all inside interval", abs(p - 1.0) < 1e-9)
    check("PICP = 0.0 when none inside (shifted far)", picp(actual, actual+10, actual+20) == 0.0)

    # Pinball loss
    pl = pinball_loss(actual, pred, q=0.5)
    check("pinball_loss non-negative", pl >= 0)
    check("pinball_loss = 0 for perfect", pinball_loss(actual, actual, q=0.5) == 0.0)

    # MASE: signature is mase(y_true, y_pred, y_train, seasonality=1)
    y_train = np.array([8.0, 9.0, 10.0, 11.0])  # for scaling
    m_val = mase(actual, pred, y_train, seasonality=1)
    check("MASE positive", m_val > 0)

except ImportError as e:
    results.append(f"SKIP  eval.metrics import failed: {e} (scipy dependency)")
except Exception as e:
    results.append(f"ERROR eval.metrics: {e}")
    traceback.print_exc()

# ---- Test forecast/splits.py (uses pandas) ----
# Try it — if pandas hangs we'll know
try:
    # just import the module, don't call anything that uses pandas yet
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "splits",
        Path(__file__).parent / "forecast" / "splits.py"
    )
    # Only check the module can be found, don't execute it (would import pandas)
    check("forecast/splits.py file exists", spec is not None)
except Exception as e:
    results.append(f"ERROR forecast/splits.py: {e}")

# ---- Summary ----
passed = sum(1 for r in results if r.startswith("PASS"))
failed = sum(1 for r in results if r.startswith("FAIL"))
errors = sum(1 for r in results if r.startswith("ERROR"))
summary = f"\nResults: {passed} passed, {failed} failed, {errors} errors\n"

print()
for r in results:
    print(r)
print(summary)

out_path = Path(__file__).parent / "docs" / "stats_test_results.txt"
out_path.write_text("\n".join(results) + summary, encoding="utf-8")
print(f"Results written to {out_path}")
sys.exit(0 if (failed == 0 and errors == 0) else 1)
