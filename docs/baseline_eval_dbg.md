# Baseline & Planner Evaluation (held-out latent scenarios, eval-only)

- **Note:** All numbers are simulator outputs under stated assumptions, not TBO real P&L.
- **Setup:** Strategies decided on forecast scenarios from censored history (embargoed time split); realised incremental profit evaluated on latent-demand held-out scenarios (ground truth read only in /eval). Common random numbers per seed across all strategies.
- **Budget:** Rs 10,000,000 | **Seeds:** 10 | **Scenarios/seed:** 150 | **Held-out start:** 2026-12-18 | **Forecast cutoff:** 2027-09-18 | **Embargo days:** 14
- **EXIT GATE:** FAIL (never_spend_above_budget=True; planner_sometimes_unspent=False; beats_B1_equal_split=False; beats_B2_volume_proportional=False; beats_B3_mean_demand_newsvendor=False)

## Results (mean and 90% bootstrap CI over seeds)

| strategy | profit_mean | profit_CI90_lo | profit_CI90_hi | spend_mean | spend_max | spoilage_mean | fill_uplift_mean | roi_on_cash_mean | sometimes_unspent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| B0_no_prebuy | 1180360.31 | 1178081.78 | 1182429.06 | 0.00 | 0.00 | 0.00 | 0.000000 | 0.000000 | True |
| B1_equal_split | 4208749.01 | 4205551.31 | 4211475.97 | 10000000.00 | 10000000.00 | 21310.22 | 0.118102 | 0.420875 | False |
| B2_volume_proportional | 4233120.73 | 4229880.52 | 4236409.17 | 10000000.00 | 10000000.00 | 21992.52 | 0.112817 | 0.423312 | False |
| B3_mean_demand_newsvendor | 4458902.97 | 4443246.01 | 4475996.48 | 10000000.00 | 10000000.00 | 235359.49 | 0.106334 | 0.445890 | False |
| B4_correct_fractile_no_budget | 4299765.00 | 4296860.78 | 4303115.79 | 10000000.00 | 10000000.00 | 3215.09 | 0.113323 | 0.429976 | False |
| Planner | 4038659.29 | 4021603.65 | 4056721.45 | 10000000.00 | 10000000.00 | 72284.42 | 0.113844 | 0.403866 | False |

## Planner vs B1-B3 (profit CIs)

- B1_equal_split: planner_mean=4038659.29 CI90=4021603.65..4056721.45 vs baseline_mean=4208749.01 CI90=4205551.31..4211475.97 => favourable=False
- B2_volume_proportional: planner_mean=4038659.29 CI90=4021603.65..4056721.45 vs baseline_mean=4233120.73 CI90=4229880.52..4236409.17 => favourable=False
- B3_mean_demand_newsvendor: planner_mean=4038659.29 CI90=4021603.65..4056721.45 vs baseline_mean=4458902.97 CI90=4443246.01..4475996.48 => favourable=False
