# Stress Test Results (risk-neutral vs risk-averse)

- **Note:** Simulator outputs under stated assumptions, not TBO real P&L. Decisions on stressed forecast scenarios; realized on stressed held-out latent scenarios (eval-only); bot contamination affects decision signal only.
- **Budget:** Rs 10,000,000 | **Scenarios:** 300 | **Seed:** 42 | **RN:** water-filling | **RA:** mean-CVaR LP (alpha=0.95, beta=0.5)
- **Cases:** demand -30%/+30%, volatility x1.6, correlation x2, sell-out a -0.20/+0.10, event cancellation, FX +10%, supplier default (largest commitment), competitor price war (p -8%), 20% bot contamination of search logs.

| scenario | RN expected uplift (INR) | RN CVaR95 (INR) | RN spoilage (INR) | RA expected uplift (INR) | RA CVaR95 (INR) | RA spoilage (INR) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| demand_-30pct | 3244008.53 | 2665497.66 | 264143.44 | 3392060.01 | 2643154.30 | 369238.59 |
| demand_+30pct | 4431086.05 | 4177381.26 | 1538.38 | 4484853.32 | 3709556.57 | 174511.66 |
| volatility_x1.6 | 3728961.15 | 2906254.58 | 203150.82 | 3855053.61 | 2684019.90 | 350671.57 |
| correlation_x2 | 4075481.54 | 3677461.99 | 44993.31 | 3918687.45 | 2864101.92 | 425077.43 |
| sellout_a_-0.20 | 3665151.94 | 3330409.34 | 67026.65 | 3497152.27 | 2559171.34 | 430807.52 |
| sellout_a_+0.10 | 4104362.00 | 3687777.57 | 67026.65 | 3943521.99 | 2936007.52 | 435991.30 |
| event_cancellation | 3727000.62 | 3323191.22 | 117969.29 | 3544328.88 | 2929007.82 | 377621.26 |
| fx_+10pct | 2516745.31 | 2190792.89 | 67026.65 | 2254859.61 | 1393246.05 | 432575.01 |
| supplier_default_largest | 3920162.74 | 3539228.00 | 67026.65 | 3740879.86 | 2745580.65 | 444470.62 |
| competitor_price_war_-8pct | 1495380.89 | 1227552.76 | 67026.65 | 1281274.14 | 511627.44 | 430167.05 |
| bot_contamination_20pct_search | 4030440.09 | 3705928.06 | 30073.29 | 3799851.05 | 2771259.83 | 449180.48 |
