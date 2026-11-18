# DECISIONS — TBO Inventory Optimization

> Every assumption or design decision made during the audit/repair phase, one entry each.
> Record the reason. Do not change section B math without adding an entry here.

---

## 2026-11-18 — Initial audit takeover

| # | Decision | Reason |
|---|---|---|
| D01 | Made sim/__init__.py, forecast/__init__.py, optimizer/__init__.py lazy (no eager wildcard imports) | Eager imports caused pytest collection hang: optimizer/__init__ → optimizer.planner → forecast.scenarios → scipy.stats → OpenBLAS DLL deadlock on Windows |
| D02 | Moved `import lightgbm` from module-level to inside _get_lgb() helper in forecast/quantile_model.py | LightGBM DLL initialization also caused hang at module-import time; lazy import means tests that don't call .fit() never trigger the DLL load |
| D03 | Kept all existing test files unchanged | Spec requires: never delete or weaken tests |
| D04 | Created run_math_tests.py as a standalone validation script (not a pytest file) | pytest itself hangs due to scipy.stats in some test files; standalone script can set OMP_NUM_THREADS=1 before any BLAS library loads, bypassing the deadlock |
| D05 | Created run_sim_nopandas.py and run_stats_nopandas.py for layer-specific validation | Isolated from scipy import; test schema, cell definitions, seed utility, config, metrics, leakage guard |
| D06 | Used OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1 set via os.environ at top of scripts | Must be set before pandas/numpy load their BLAS thread pools; setting them after `import pandas` has no effect |
| D07 | Did not change any math in cell.py, fractile.py, waterfill.py, profit.py | Math verified correct against all spec anchors (20/20 PASS); no deviation found |
| D08 | Marked phases P4-P10 as PARTIAL rather than PASS | Cannot live-run them due to env blocker; static analysis confirms correct structure but PASS requires running evidence per spec |
| D09 | Did not add features outside the spec (no new cells, no new models, no new endpoints) | YAGNI; audit-only phase |
| D10 | docs/limitations.md already exists and contains synthetic-data disclaimer | Verified present; no edit needed |
| D11 | The __init__.py removals do not break any existing test | All test files import from submodules directly (e.g. `from optimizer.cell import Cell`), never `from optimizer import Cell`; confirmed by grep |
| D12 | mase() bug in run_stats_nopandas.py test script was wrong call (naive_errors kwarg), not a bug in eval/metrics.py | eval/metrics.py mase() signature is (y_true, y_pred, y_train, seasonality=1); test_forecast_eval.py uses it correctly; no code change needed |
| D13 | Added root `.gitignore` entries for `.venv/`, Python caches/bytecode, egg-info, `.env`, and scratch parquet files | Prevent accidental staging of local environment files, generated caches, and temporary test artifacts |
