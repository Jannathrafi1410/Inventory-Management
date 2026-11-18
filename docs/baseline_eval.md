# Baseline & Planner Evaluation (held-out latent scenarios, eval-only)

- **Note:** All numbers are simulator outputs under stated assumptions, not TBO real P&L.
- **Setup:** Strategies decided on forecast scenarios from censored history (embargoed time split); realised incremental profit evaluated on latent-demand held-out scenarios (ground truth read only in /eval). Common random numbers per seed across all strategies.
- **Assumption (planner):** planner applies candidate funnel (screen delta_m>0, volume/sellout/discount/event; rank by delta_m/k with k=0 first, top 18) then budget-aware water-filling with reserve_fraction=0.03 to preserve capital discipline; numbers are simulator outputs under stated assumptions.
- **Budget:** Rs 10,000,000 | **Seeds:** 50 | **Scenarios/seed:** 300 | **Held-out start:** 2026-12-18 | **Forecast cutoff:** 2027-09-18 | **Embargo days:** 14
- **EXIT GATE:** PASS (never_spend_above_budget=True; planner_sometimes_unspent=True; beats_B1_equal_split=True; beats_B2_volume_proportional=True; beats_B3_mean_demand_newsvendor=True)

## Results (mean and 90% bootstrap CI over seeds)

| strategy | profit_mean | profit_CI90_lo | profit_CI90_hi | spend_mean | spend_max | spoilage_mean | fill_uplift_mean | roi_on_cash_mean | sometimes_unspent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| B0_no_prebuy | 1181751.63 | 1180664.36 | 1182784.15 | 0.00 | 0.00 | 0.00 | 0.000000 | 0.000000 | True |
| B1_equal_split | 4211741.52 | 4210347.83 | 4213186.45 | 10000000.00 | 10000000.00 | 20952.43 | 0.118203 | 0.421174 | False |
| B2_volume_proportional | 4235362.02 | 4233499.34 | 4237097.27 | 10000000.00 | 10000000.00 | 23799.56 | 0.112996 | 0.423536 | False |
| B3_mean_demand_newsvendor | 4452598.78 | 4448736.90 | 4456391.77 | 10000000.00 | 10000000.00 | 240157.93 | 0.106388 | 0.445260 | False |
| B4_correct_fractile_no_budget | 4300714.10 | 4299342.49 | 4302016.26 | 10000000.00 | 10000000.00 | 3708.39 | 0.113409 | 0.430071 | False |
| Planner | 4678493.18 | 4673574.36 | 4683133.64 | 9673849.20 | 9699842.25 | 64296.74 | 0.113675 | 0.483624 | True |

## Planner vs B1-B3 (profit CIs)

- B1_equal_split: planner_mean=4678493.18 CI90=4673574.36..4683133.64 vs baseline_mean=4211741.52 CI90=4210347.83..4213186.45 => favourable=True
- B2_volume_proportional: planner_mean=4678493.18 CI90=4673574.36..4683133.64 vs baseline_mean=4235362.02 CI90=4233499.34..4237097.27 => favourable=True
- B3_mean_demand_newsvendor: planner_mean=4678493.18 CI90=4673574.36..4683133.64 vs baseline_mean=4452598.78 CI90=4448736.90..4456391.77 => favourable=True
