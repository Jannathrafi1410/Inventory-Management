# Oracle (Perfect-Foresight) Evaluation

- **Note:** Simulator outputs under stated assumptions, not TBO real P&L. Oracle has perfect foresight of realized latent demand per scenario (eval-only).
- **Budget:** Rs 10,000,000 | **Seeds:** 50 | **Scenarios/seed:** 300
- **Oracle mean uplift (INR):** 5019656.06

| strategy | mean uplift (INR) | % oracle captured | mean regret (INR) | mean spoilage (INR) | mean spend (INR) |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0_no_prebuy | 1179362.82 | 23.49% | 3840293.24 | 0.00 | 0.00 |
| B1_equal_split | 4208720.94 | 83.84% | 810935.12 | 21335.49 | 10000000.00 |
| B2_volume_proportional | 4233535.75 | 84.34% | 786120.31 | 23236.45 | 10000000.00 |
| B3_mean_demand_newsvendor | 4449047.99 | 88.63% | 570608.07 | 240297.51 | 10000000.00 |
| B4_correct_fractile_no_budget | 4298693.04 | 85.64% | 720963.03 | 3486.28 | 10000000.00 |
| Planner | 4680099.21 | 93.24% | 339556.85 | 62174.74 | 9672937.17 |
