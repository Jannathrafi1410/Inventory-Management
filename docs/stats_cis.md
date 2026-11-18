# Statistical Evaluation (Paired Bootstrap, CRN, 90% CIs)

- **Note:** Simulator outputs under stated assumptions, not TBO real P&L. Paired bootstrap on CRN, 90% percentile CIs (2000 resamples). Two generator parameterisations checked to reduce overfitting to simulator defaults.
- **Budget:** Rs 10,000,000 | **Seeds:** 10 | **Scenarios/seed:** 120
- **Robustness (Planner mean > B1-B3 in both parameterisations):** True

## Parameterisation: A_baseline (rho=0.6, demand_scale=1.0, sigma_scale=1.0)

| strategy | mean uplift (INR) | CI90 lo | CI90 hi |
| --- | ---: | ---: | ---: |
| B0_no_prebuy | 1177154.27 | 1173751.68 | 1180712.85 |
| B1_equal_split | 4208995.24 | 4204497.52 | 4213395.13 |
| B2_volume_proportional | 4235140.67 | 4230251.32 | 4240998.10 |
| B3_mean_demand_newsvendor | 4433341.51 | 4418557.98 | 4448230.20 |
| B4_correct_fractile_no_budget | 4295831.69 | 4291849.37 | 4300102.28 |
| Planner | 4636740.15 | 4619248.47 | 4655259.24 |

Paired differences (Planner - baseline), 90% CI:

| comparison | mean diff (INR) | CI90 lo | CI90 hi | planner better (CI>0) |
| --- | ---: | ---: | ---: | :---: |
| Planner - B1_equal_split | 427744.91 | 411049.57 | 445244.01 | True |
| Planner - B2_volume_proportional | 401599.47 | 384112.00 | 419327.19 | True |
| Planner - B3_mean_demand_newsvendor | 203398.64 | 187137.02 | 218811.90 | True |
| Planner - B0_no_prebuy | 3459585.88 | 3442584.37 | 3477205.19 | True |
| Planner - B4_correct_fractile_no_budget | 340908.46 | 323539.58 | 358838.15 | True |

## Parameterisation: B_alt_generator (rho=0.85, demand_scale=1.1, sigma_scale=1.25)

| strategy | mean uplift (INR) | CI90 lo | CI90 hi |
| --- | ---: | ---: | ---: |
| B0_no_prebuy | 1294124.13 | 1287690.35 | 1300969.37 |
| B1_equal_split | 4341811.49 | 4333633.70 | 4350437.88 |
| B2_volume_proportional | 4360494.09 | 4351859.95 | 4369225.33 |
| B3_mean_demand_newsvendor | 4636226.28 | 4613398.91 | 4659334.19 |
| B4_correct_fractile_no_budget | 4421491.04 | 4414244.03 | 4429162.45 |
| Planner | 4764465.09 | 4744545.69 | 4783301.34 |

Paired differences (Planner - baseline), 90% CI:

| comparison | mean diff (INR) | CI90 lo | CI90 hi | planner better (CI>0) |
| --- | ---: | ---: | ---: | :---: |
| Planner - B1_equal_split | 422653.60 | 406811.06 | 438762.75 | True |
| Planner - B2_volume_proportional | 403971.00 | 387030.84 | 419876.54 | True |
| Planner - B3_mean_demand_newsvendor | 128238.81 | 108279.88 | 145134.92 | True |
| Planner - B0_no_prebuy | 3470340.96 | 3453807.15 | 3487121.21 | True |
| Planner - B4_correct_fractile_no_budget | 342974.04 | 326418.38 | 359874.52 | True |

