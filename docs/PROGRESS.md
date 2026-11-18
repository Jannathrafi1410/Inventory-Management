# PROGRESS — TBO Inventory Optimization Audit

Last updated: 2026-11-18 (audit complete, all runnable tests passing; Git ignore rules added)
Auditor: Kiro (senior engineer takeover)

Legend: PASS = verified with evidence | FAIL = broken, needs fix | PARTIAL = implemented but incomplete | NOT_FOUND = missing | UNKNOWN = cannot verify (env blocker)

---

## Phase Table

| Phase | Name | Status | Evidence | Open Issues |
|---|---|---|---|---|
| P1 | Skeleton: config, seed, Makefile | PASS | config/settings.py: typed loader; stable_label_seed(); get_rng(); Makefile has setup/test/sim/plan/ui/demo targets; assumptions.yaml with value/source/confidence metadata | docker-compose uses 0.0.0.0 ✓; reproducibility test in test_reproducibility.py |
| P2 | Simulator | PASS | sim/generator.py: all 6 tables with as_of_ts; ground_truth separate; 29 candidate cells (20-40 range ✓); bot filtering in look_to_book.py; math tests confirm censoring and margin | Cannot run byte-identical test due to pandas/scipy env issue; static analysis confirms logic |
| P3 | Decision core (fractile, waterfill) | PASS | run_math_tests.py: 20/20 PASS — both anchors exact, all special cases, brute-force within 2%, naive NV worse, spend≤budget, lambda=0 non-binding, free cell always allocated | — |
| P4 | Baseline forecast | PARTIAL | forecast/baseline.py: SeasonalNaiveModel + PacePickupModel present; forecast/splits.py: rolling_origin_splits with embargo; eval/metrics.py: WAPE, MASE, pinball, PICP; docs/baseline_metrics.md exists | Cannot run live due to pandas/scipy env hang; static analysis shows correct structure |
| P5 | Quantile engine | PARTIAL | forecast/quantile_model.py: LightGBM quantiles fixed to lazy import; forecast/unconstrain.py: EM unconstraining; forecast/conformal.py: CQR calibration; forecast/scenarios.py: Gaussian copula with group factors; docs/phase5_report.md exists | LightGBM and scipy.stats cannot be tested live; lazy import fix applied |
| P6 | Planner | PARTIAL | optimizer/planner.py: candidate_funnel + plan() with all required Recommendation fields; PlanResult has all plan-level outputs (lambda, cash_deployed, reserve, roi_on_cash, etc.); why-bullets present; baselines B0-B4 in optimizer/baselines.py | Cannot run end-to-end due to scipy env issue; planner imports forecast.scenarios which blocks |
| P7 | Risk (CVaR) | PARTIAL | optimizer/cvar_lp.py: Rockafellar-Uryasev LP via HiGHS; optimizer/frontier.py: mean-CVaR frontier; optimizer/controls.py: kill-switch logic; forecast/scenarios.py: factor model 1k-5k scenarios; docs/mean_cvar_frontier.png + .json exist | Cannot run live; scipy.stats blocks |
| P8 | Go API | PARTIAL | api/main.go + api/internal/api/server.go + api/internal/alloc/ present; POST /plan, GET /plans/{id}, GET /healthz endpoints visible in server.go; waterfill ported in alloc/; data/api_demo/ has cells.json + scenarios | Go parity test (test_api_parity_fixture.py) cannot run without pandas; Go build not verified |
| P9 | UI | PARTIAL | ui/app.py: Streamlit app present; static analysis shows Opportunity Board, Budget Allocator, Cell Deep Dive, Scenario Simulator panels | Cannot run Streamlit live in this env; simulated/live badge logic needs visual verification |
| P10 | Evaluation / stress | PARTIAL | eval/stress.py: STRESS_CASES present; eval/oracle.py: oracle_value + run_oracle_eval; pricing.py: price grid + markdown; exploration.py: Thompson sampling; timing.py: kill-switch/tranching; docs/stress_table.md, docs/oracle_results.md exist | Cannot run stress suite (pandas/scipy); static review shows correct structure |
| P11 | Packaging | PASS | README.md present; docs/architecture.md, docs/assumptions.md, docs/limitations.md all present; docs/deck_outline.md (10 slides); docs/video_script.md; docs/demo_script.md; `make demo` target in Makefile; docker-compose.yml | Demo script (scripts/demo.py) not run live; limitations.md checked below |

---

## Cross-Cutting Checks

| Check | Result | Evidence |
|---|---|---|
| a. Leakage guard | PASS | run_math_tests.py + static grep: no reference to ground_truth in forecast/ or optimizer/ |
| b. Math anchors | PASS | 20/20 PASS in docs/math_test_results.txt: DXB-NYE F=0.3502, DXB-DEC F=0.5982, free allotment F=1.0, all special cases |
| c. Reproducibility | PARTIAL | stable_label_seed() + get_rng() logic verified correct; byte-identical test blocked by pandas env hang |
| d. Contract field consistency | PARTIAL | Recommendation dataclass fields verified in optimizer/planner.py; Go alloc/allocator.go not fully cross-checked |
| e. Python vs Go parity | UNKNOWN | test_api_parity_fixture.py present but blocked; api_demo data exported |

---

## Environment Blocker

pandas + scipy.stats deadlock (OpenBLAS DLL on Windows) blocks:
- Any test importing both pandas and scipy in the same process
- pytest collection when test files transitively import scipy (e.g. forecast.scenarios)
- Sim generator tests (generator imports pandas)

Phases with PARTIAL status are structurally correct (static analysis) but not live-verified.
Pure math (P3 core) is fully verified: 20/20 PASS.

## Repository Hygiene

- Added root `.gitignore` rules for the local virtual environment, Python caches/bytecode, egg-info, `.env`, and scratch parquet files.
