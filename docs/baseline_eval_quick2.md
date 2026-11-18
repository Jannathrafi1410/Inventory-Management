# Baseline & Planner Evaluation (held-out latent scenarios, eval-only)

- **Note:** All numbers are simulator outputs under stated assumptions, not TBO real P&L.
- **Setup:** Strategies decided on forecast scenarios from censored history (embargoed time split); realised incremental profit evaluated on latent-demand held-out scenarios (ground truth read only in /eval). Common random numbers per seed across all strategies.
- **Budget:** Rs 10,000,000 | **Seeds:** 5 | **Scenarios/seed:** 120 | **Held-out start:** 2026-12-18 | **Forecast cutoff:** 2027-09-18 | **Embargo days:** 14
- **EXIT GATE:** PASS (never_spend_above_budget=True; planner_sometimes_unspent=True; beats_B1_equal_split=True; beats_B2_volume_proportional=True; beats_B3_mean_demand_newsvendor=True)

## Results (mean and 90% bootstrap CI over seeds)

| strategy | profit_mean | profit_CI90_lo | profit_CI90_hi | spend_mean | spend_max | spoilage_mean | fill_uplift_mean | roi_on_cash_mean | sometimes_unspent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| B0_no_prebuy | 1177153.05 | 1173108.63 | 1180934.62 | 0.00 | 0.00 | 0.00 | 0.000000 | 0.000000 | True |
| B1_equal_split | 4207497.66 | 4200692.37 | 4214295.54 | 10000000.00 | 10000000.00 | 21839.80 | 0.118074 | 0.420750 | False |
| B2_volume_proportional | 4232416.75 | 4224678.59 | 4238330.63 | 10000000.00 | 10000000.00 | 21911.34 | 0.113003 | 0.423242 | False |
| B3_mean_demand_newsvendor | 4462684.01 | 4440172.76 | 4483519.91 | 10000000.00 | 10000000.00 | 229905.57 | 0.106758 | 0.446268 | False |
| B4_correct_fractile_no_budget | 4297871.40 | 4290894.52 | 4303702.63 | 10000000.00 | 10000000.00 | 3183.55 | 0.113410 | 0.429787 | False |
| Planner | 4660535.57 | 4646998.01 | 4674033.13 | 9673584.39 | 9696393.11 | 57964.61 | 0.114405 | 0.481780 | True |

## Planner vs B1-B3 (profit CIs)

- B1_equal_split: planner_mean=4660535.57 CI90=4646998.01..4674033.13 vs baseline_mean=4207497.66 CI90=4200692.37..4214295.54 => favourable=True
- B2_volume_proportional: planner_mean=4660535.57 CI90=4646998.01..4674033.13 vs baseline_mean=4232416.75 CI90=4224678.59..4238330.63 => favourable=True
- B3_mean_demand_newsvendor: planner_mean=4660535.57 CI90=4646998.01..4674033.13 vs baseline_mean=4462684.01 CI90=4440172.76..4483519.91 => favourable=True
