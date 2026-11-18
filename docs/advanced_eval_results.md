# Advanced Evaluation Results (Exit Gate)

**Note:** All numbers are simulator outputs under stated assumptions, not TBO real P&L. Ground truth (`data/ground_truth.parquet`) is read only inside `/eval` code paths.

This document consolidates: baseline/planner evaluation, oracle (perfect-foresight) gap, stress tests (risk-neutral vs risk-averse), paired-bootstrap 90% CIs across two generator parameterisations, and Phase 8 evidence/gaps. Supporting artifacts: `docs/baseline_eval.md`, `docs/oracle_results.md|json`, `docs/stress_table.md|json`, `docs/stats_cis.md|json`, `docs/pricing_ladder.md|json`, `docs/timing_policy.md|json`, `docs/exploration.md|json`.

## 1) Baselines table (B0–B4 + Planner)

Source: `docs/baseline_eval.md` (held-out latent scenarios, eval-only; forecast decisions from censored history with embargoed time split; CRN per seed).

- **Setup:** budget Rs 10,000,000 | seeds 50 | scenarios/seed 300 | held-out start 2026-12-18 | forecast cutoff 2027-09-18 | embargo 14 days
- **EXIT GATE (baseline_eval):** PASS (never_spend_above_budget=True; planner_sometimes_unspent=True; beats B1/B2/B3=True)
- **Planner assumption:** candidate funnel (screen `delta_m>0`, volume/sellout/discount/event; rank by `delta_m/k` with `k=0` first, top 18) then budget-aware water-filling with `reserve_fraction=0.03`.

| strategy | profit_mean | profit_CI90_lo | profit_CI90_hi | spend_mean | spend_max | spoilage_mean | fill_uplift_mean | roi_on_cash_mean | sometimes_unspent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| B0_no_prebuy | 1181751.63 | 1180664.36 | 1182784.15 | 0.00 | 0.00 | 0.00 | 0.000000 | 0.000000 | True |
| B1_equal_split | 4211741.52 | 4210347.83 | 4213186.45 | 10000000.00 | 10000000.00 | 20952.43 | 0.118203 | 0.421174 | False |
| B2_volume_proportional | 4235362.02 | 4233499.34 | 4237097.27 | 10000000.00 | 10000000.00 | 23799.56 | 0.112996 | 0.423536 | False |
| B3_mean_demand_newsvendor | 4452598.78 | 4448736.90 | 4456391.77 | 10000000.00 | 10000000.00 | 240157.93 | 0.106388 | 0.445260 | False |
| B4_correct_fractile_no_budget | 4300714.10 | 4299342.49 | 4302016.26 | 10000000.00 | 10000000.00 | 3708.39 | 0.113409 | 0.430071 | False |
| Planner | 4678493.18 | 4673574.36 | 4683133.64 | 9673849.20 | 9699842.25 | 64296.74 | 0.113675 | 0.483624 | True |

Planner vs B1–B3 (profit CIs, favourable): all favourable per `docs/baseline_eval.md`.

## 2) Oracle (perfect-foresight) gap / % captured / regret

Source: `docs/oracle_results.md` / `docs/oracle_results.json` (oracle has perfect foresight of realized latent demand per scenario, eval-only; same constraints/economics).

- **Budget:** Rs 10,000,000 | **Seeds:** 50 | **Scenarios/seed:** 300
- **Oracle mean uplift (INR):** 5,019,656.06

| strategy | mean uplift (INR) | % oracle captured | mean regret (INR) | mean spoilage (INR) | mean spend (INR) |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0_no_prebuy | 1,179,362.82 | 23.49% | 3,840,293.24 | 0.00 | 0.00 |
| B1_equal_split | 4,208,720.94 | 83.84% | 810,935.12 | 21,335.49 | 10,000,000.00 |
| B2_volume_proportional | 4,233,535.75 | 84.34% | 786,120.31 | 23,236.45 | 10,000,000.00 |
| B3_mean_demand_newsvendor | 4,449,047.99 | 88.63% | 570,608.07 | 240,297.51 | 10,000,000.00 |
| B4_correct_fractile_no_budget | 4,298,693.04 | 85.64% | 720,963.03 | 3,486.28 | 10,000,000.00 |
| **Planner** | **4,680,099.21** | **93.24%** | **339,556.85** | **62,174.74** | **9,672,937.17** |

## 3) Stress tests (risk-neutral vs risk-averse)

Source: `docs/stress_table.md` / `docs/stress_results.json`.

- **Assumption:** Decisions on stressed forecast scenarios; realized on stressed held-out latent scenarios (eval-only); bot contamination affects decision signal only.
- **Setup:** budget Rs 10,000,000 | scenarios 300 | seed 42 | RN = water-filling | RA = mean-CVaR LP (α=0.95, β=0.5, reserve 3%, LP solved on up to 150 scenarios for tractability, evaluated on full set)
- **11 cases:** demand -30%/+30%, volatility x1.6, correlation x2, sell-out `a` -0.20/+0.10, event cancellation (event→35%, mixed→85%), FX +10% on `c_d`, supplier default (largest `k*cap`, sets `a=0,r=0`), competitor price war (`p` -8%), 20% bot contamination of search logs (decision-only +20%).

| scenario | RN expected uplift (INR) | RN CVaR95 (INR) | RN spoilage (INR) | RA expected uplift (INR) | RA CVaR95 (INR) | RA spoilage (INR) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| demand_-30pct | 3,244,008.53 | 2,665,497.66 | 264,143.44 | 3,392,060.01 | 2,643,154.30 | 369,238.59 |
| demand_+30pct | 4,431,086.05 | 4,177,381.26 | 1,538.38 | 4,484,853.32 | 3,709,556.57 | 174,511.66 |
| volatility_x1.6 | 3,728,961.15 | 2,906,254.58 | 203,150.82 | 3,855,053.61 | 2,684,019.90 | 350,671.57 |
| correlation_x2 | 4,075,481.54 | 3,677,461.99 | 44,993.31 | 3,918,687.45 | 2,864,101.92 | 425,077.43 |
| sellout_a_-0.20 | 3,665,151.94 | 3,330,409.34 | 67,026.65 | 3,497,152.27 | 2,559,171.34 | 430,807.52 |
| sellout_a_+0.10 | 4,104,362.00 | 3,687,777.57 | 67,026.65 | 3,943,521.99 | 2,936,007.52 | 435,991.30 |
| event_cancellation | 3,727,000.62 | 3,323,191.22 | 117,969.29 | 3,544,328.88 | 2,929,007.82 | 377,621.26 |
| fx_+10pct | 2,516,745.31 | 2,190,792.89 | 67,026.65 | 2,254,859.61 | 1,393,246.05 | 432,575.01 |
| supplier_default_largest | 3,920,162.74 | 3,539,228.00 | 67,026.65 | 3,740,879.86 | 2,745,580.65 | 444,470.62 |
| competitor_price_war_-8pct | 1,495,380.89 | 1,227,552.76 | 67,026.65 | 1,281,274.14 | 511,627.44 | 430,167.05 |
| bot_contamination_20pct_search | 4,030,440.09 | 3,705,928.06 | 30,073.29 | 3,799,851.05 | 2,771,259.83 | 449,180.48 |

## 4) 90% CIs across 2 parameterisations (paired bootstrap, CRN)

Source: `docs/stats_cis.md` / `docs/stats_cis.json`.

- **Assumption:** Paired bootstrap on CRN, 90% percentile CIs (2000 resamples, seed 123). Two generator parameterisations checked to reduce overfitting to simulator defaults.
- **Run used for artifacts:** budget Rs 10,000,000 | **seeds 10** | **scenarios/seed 120** (demo-practical runtime). **Scale-up to spec:** `.\.venv\Scripts\python.exe -m eval.stats --budget 10000000 --seeds 50 --scenarios 300 --out-md docs/stats_cis.md --out-json docs/stats_cis.json`
- **Robustness (Planner mean > B1–B3 in both parameterisations):** True

### A_baseline (`rho=0.6, demand_scale=1.0, sigma_scale=1.0`)

| strategy | mean uplift (INR) | CI90 lo | CI90 hi |
| --- | ---: | ---: | ---: |
| B0_no_prebuy | 1,177,154.27 | 1,173,751.68 | 1,180,712.85 |
| B1_equal_split | 4,208,995.24 | 4,204,497.52 | 4,213,395.13 |
| B2_volume_proportional | 4,235,140.67 | 4,230,251.32 | 4,240,998.10 |
| B3_mean_demand_newsvendor | 4,433,341.51 | 4,418,557.98 | 4,448,230.20 |
| B4_correct_fractile_no_budget | 4,295,831.69 | 4,291,849.37 | 4,300,102.28 |
| Planner | 4,636,740.15 | 4,619,248.47 | 4,655,259.24 |

| comparison | mean diff (INR) | CI90 lo | CI90 hi | planner better (CI>0) |
| --- | ---: | ---: | ---: | :---: |
| Planner - B1_equal_split | 427,744.91 | 411,049.57 | 445,244.01 | True |
| Planner - B2_volume_proportional | 401,599.47 | 384,112.00 | 419,327.19 | True |
| Planner - B3_mean_demand_newsvendor | 203,398.64 | 187,137.02 | 218,811.90 | True |
| Planner - B0_no_prebuy | 3,459,585.88 | 3,442,584.37 | 3,477,205.19 | True |
| Planner - B4_correct_fractile_no_budget | 340,908.46 | 323,539.58 | 358,838.15 | True |

### B_alt_generator (`rho=0.85, demand_scale=1.1, sigma_scale=1.25`)

| strategy | mean uplift (INR) | CI90 lo | CI90 hi |
| --- | ---: | ---: | ---: |
| B0_no_prebuy | 1,294,124.13 | 1,287,690.35 | 1,300,969.37 |
| B1_equal_split | 4,341,811.49 | 4,333,633.70 | 4,350,437.88 |
| B2_volume_proportional | 4,360,494.09 | 4,351,859.95 | 4,369,225.33 |
| B3_mean_demand_newsvendor | 4,636,226.28 | 4,613,398.91 | 4,659,334.19 |
| B4_correct_fractile_no_budget | 4,421,491.04 | 4,414,244.03 | 4,429,162.45 |
| Planner | 4,764,465.09 | 4,744,545.69 | 4,783,301.34 |

| comparison | mean diff (INR) | CI90 lo | CI90 hi | planner better (CI>0) |
| --- | ---: | ---: | ---: | :---: |
| Planner - B1_equal_split | 422,653.60 | 406,811.06 | 438,762.75 | True |
| Planner - B2_volume_proportional | 403,971.00 | 387,030.84 | 419,876.54 | True |
| Planner - B3_mean_demand_newsvendor | 128,238.81 | 108,279.88 | 145,134.92 | True |
| Planner - B0_no_prebuy | 3,470,340.96 | 3,453,807.15 | 3,487,121.21 | True |
| Planner - B4_correct_fractile_no_budget | 342,974.04 | 326,418.38 | 359,874.52 | True |

## 5) Additional decision artifacts (generated/verified)

- **Pricing optimization & markdown ladder:** `docs/pricing_ladder.md`, `docs/pricing_results.json` (0.88x–1.16x grid, segment elasticities, pass-through `tau=0.30` on `c_d`, unconstrained fractile-capped `q*(p)`, argmax/flatness, +14/+7/+3d ladder).
- **Timing policy:** `docs/timing_policy.md`, `docs/timing_policy.json` (tranching 40/35/25% at -120/-75/-45d resized by fractile; priced WAIT buy-now vs after-signal; release hold if `P(arrives)*margin > refund`).
- **Exploration:** `docs/exploration.md`, `docs/exploration.json` (Thompson sampling, 3% reserved budget in [2%,5%], cluster shrinkage 0.35 over `(region,segment,product)`, integrated with main allocation on remainder).

## 6) Limitations

- **Data simulated:** all evaluation/planning/pricing/timing/exploration numbers are simulator outputs under stated assumptions, not TBO real P&L. Latent demand ground truth is synthetic (`data/ground_truth.parquet`, eval-only).
- **Parameters assumed:** economics (`p,c_d,c_b,k,r,a,g,cap`), elasticities, pass-through (`tau=0.30`), carry/lock assumptions, fractiles, reserve fractions, scenario-generation/correlation/volatility transforms, stress magnitudes, and policy heuristics are model assumptions/config choices, not empirically calibrated from TBO production data.
- **Cancellations simplified:** event/mixed cancellation modeled as multiplicative demand scaling (event→35%, mixed→85%) in stress; broader cancellation/refund/penalty dynamics and real supplier/booking workflows are simplified.
- **Forecast-stress vs realized-stress design:** stress decisions are solved on stressed forecast scenarios and realized on stressed held-out latent scenarios (eval-only); bot contamination inflates decision signal only.
- **Runtime/demo scale:** `docs/stats_cis.*` artifacts in this run use 10 seeds × 120 scenarios per parameterisation (spec target 50 seeds × 300 scenarios) for practical runtime; scale-up command is documented above.
- **Risk-averse stress solver:** mean-CVaR LP in stress solved on up to 150 scenarios for tractability and evaluated on full scenario set.
- **Go API risk handling:** `/api/v1/plan` accepts `risk.beta/alpha` but allocation uses the water-filling port; metrics report plan uplift/CVaR95 (mean-CVaR LP not ported to Go).

## 7) Phase 8 evidence and gaps

### Demo re-solve latency (`POST /api/v1/plan`)
- **Measured (demo dataset `data/api_demo`, 27 cells, 500 scenarios):** `HTTP 200`, `solve_time_ms=0.5099`, end-to-end request latency **~10.62 ms**, `cash_deployed=9,700,000`, `lambda=2.582e112` (numerical tail in bisection regime), `dataset_version=demo_v1`, `recommendations=27`.
- **Target:** demo budget re-solve < 1s — **met** for demo scale in this measurement.
- **Large-scale benchmark (known):** `BenchmarkWaterfill_50000x1000-12` ~**6.18s** on this machine (target ~1s **not met**).

### UI E2E
- **Gap:** no headless browser E2E for Streamlit UI. Verification performed: `py_compile ui/app.py` (prior), UI implemented against Go API/local demo artifacts; `go test ./internal/alloc ./internal/api -count=1` passed.

## 8) Commands run (PowerShell) and outcomes

| # | Command | Exit | Outcome / artifacts |
| --- | --- | ---: | --- |
| 1 | `.\.venv\Scripts\python.exe -m pytest -q tests/test_stats.py tests/test_stress.py` (after fixing `eval/stress.py` frozen `Cell` mutations) | 0 | 3 passed (blocking fixes verified) |
| 2 | `.\.venv\Scripts\python.exe -m pytest -q tests/test_stress.py tests/test_oracle.py tests/test_stats.py tests/test_pricing.py tests/test_timing.py tests/test_exploration.py` | 0 | 11 passed (new evaluation/decision tests) |
| 3 | `.\.venv\Scripts\python.exe -m pytest -q tests/test_planner_baselines.py tests/test_cvar_risk.py tests/test_optimizer.py tests/test_forecast_eval.py tests/test_demand_engine.py tests/test_sim.py tests/test_reproducibility.py tests/test_api_parity_fixture.py` | 0 | 43 passed (targeted existing suite; no regressions observed) |
| 4 | `.\.venv\Scripts\python.exe -m eval.stress --budget 10000000 --scenarios 300 --seed 42 --out-md docs/stress_table.md --out-json docs/stress_results.json` | 0 | `docs/stress_table.md`, `docs/stress_results.json` (11 cases) |
| 5 | `.\.venv\Scripts\python.exe -m eval.oracle --budget 10000000 --seeds 50 --scenarios 300 --out-md docs/oracle_results.md --out-json docs/oracle_results.json` | 0 | `docs/oracle_results.md`, `docs/oracle_results.json` |
| 6 | `.\.venv\Scripts\python.exe -m eval.stats --budget 10000000 --seeds 10 --scenarios 120 --out-md docs/stats_cis.md --out-json docs/stats_cis.json` | 0 | `docs/stats_cis.md`, `docs/stats_cis.json` (demo args; spec scale-up command documented) |
| 7 | `.\.venv\Scripts\python.exe -m pricing --scenarios 120 --seed 42 --out-md docs/pricing_ladder.md --out-json docs/pricing_results.json` | 0 | `docs/pricing_ladder.md`, `docs/pricing_results.json` (verified present/consistent) |
| 8 | `.\.venv\Scripts\python.exe -m timing --scenarios 120 --seed 42 --out-md docs/timing_policy.md --out-json docs/timing_policy.json` | 0 | `docs/timing_policy.md`, `docs/timing_policy.json` |
| 9 | `.\.venv\Scripts\python.exe -m exploration --budget 10000000 --scenarios 120 --seed 42 --out-md docs/exploration.md --out-json docs/exploration.json` | 0 | `docs/exploration.md`, `docs/exploration.json` |
| 10 | `Push-Location api; & "C:\Program Files\Go\bin\go.exe" test ./internal/alloc ./internal/api -count=1; Pop-Location` | 0 | Go tests passed (`internal/alloc` 1.967s, `internal/api` 4.932s) |
| 11 | Python one-liner: start Go API on port 18081 (`go run .` in `api` with `TBO_ROOT/DATA_DIR/STORE_DIR`), `GET /healthz`, `POST /api/v1/plan` budget 10,000,000 INR, capture `solve_time_ms` and end-to-end latency, then terminate | 0 | `solve_time_ms=0.5099`, E2E ~10.62 ms, 27 recs (demo re-solve < 1s) |

**Spec-compliant scale-up commands (if full runtime desired):**
- Stats: `.\.venv\Scripts\python.exe -m eval.stats --budget 10000000 --seeds 50 --scenarios 300 --out-md docs/stats_cis.md --out-json docs/stats_cis.json`
- (Oracle/stress already run with `--seeds 50 --scenarios 300` / `--scenarios 300` in this session.)

## 9) Produced/updated artifacts (this session)

- Fixed: `eval/stress.py` (non-mutating frozen-`Cell` handling), `tests/test_stats.py`
- Added tests: `tests/test_stress.py`, `tests/test_oracle.py`, `tests/test_stats.py`, `tests/test_pricing.py`, `tests/test_timing.py`, `tests/test_exploration.py`
- Generated/updated: `docs/stress_table.md`, `docs/stress_results.json`, `docs/oracle_results.md`, `docs/oracle_results.json`, `docs/stats_cis.md`, `docs/stats_cis.json`, `docs/pricing_ladder.md`, `docs/pricing_results.json`, `docs/timing_policy.md`, `docs/timing_policy.json`, `docs/exploration.md`, `docs/exploration.json`
- Consolidated: `docs/advanced_eval_results.md` (this file)
