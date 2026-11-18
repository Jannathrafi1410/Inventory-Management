# Exploration Allocation (Thompson Sampling on delta_m Posterior)

- **Note:** Simulator outputs under stated assumptions, not TBO real P&L. Thompson sampling over Normal posterior of delta_m for low-history cash-positive cells; similarity clusters (region,segment,product) with shrinkage 0.35 toward cluster mean; exploration budget 3% (2-5% allowed) reserved before main water-filling.
- **Total budget:** Rs 10,000,000 | **Exploration fraction:** 3.0% | **Exploration spend:** Rs 300,000.00 | **Eligible low-history cells:** 12
- **Main allocator spend:** Rs 9,409,000.00 | **Total spend:** Rs 9,709,000.00 | **lambda:** 25822498780869086436831993248681894182548833048615888584192974690338880141698009435618943865039225109799021576192.000000

| cell | cluster | eligible | posterior mean delta_m (INR) | posterior var | Thompson sample | explore units | explore cash (INR) |
| --- | --- | :---: | ---: | ---: | ---: | ---: | ---: |
| MLE-DEC | IO|leisure|hotel | True | 9123.00 | 10000.00 | 9066.84 | 1.59 | 50156.72 |
| SIN-F1 | SEA|event|hotel | True | 7885.00 | 10000.00 | 7985.52 | 2.45 | 44174.99 |
| LON-SUM | EU|leisure|hotel | True | 4257.28 | 10000.00 | 4186.69 | 1.60 | 23160.31 |
| PAR-SUM | EU|leisure|hotel | True | 4011.50 | 10000.00 | 4075.92 | 1.63 | 22547.50 |
| SIN-BIZ-Q2 | SEA|business|hotel | True | 3954.98 | 10000.00 | 4050.14 | 1.75 | 22404.93 |
| LON-SPR | EU|leisure|hotel | True | 3870.40 | 10000.00 | 3880.50 | 1.60 | 21466.46 |
| PAR-SPR | EU|leisure|hotel | True | 3792.52 | 10000.00 | 3781.38 | 1.61 | 20918.18 |
| ROM-SPR | EU|leisure|hotel | True | 3617.60 | 10000.00 | 3759.88 | 1.70 | 20799.23 |
| MAD-SUM | EU|leisure|hotel | True | 3562.00 | 10000.00 | 3626.96 | 1.70 | 20063.95 |
| ROM-SUM | EU|leisure|hotel | True | 3737.55 | 10000.00 | 3624.72 | 1.59 | 20051.55 |
| HKT-PEAK | SEA|leisure|hotel | True | 3102.64 | 10000.00 | 3242.73 | 1.99 | 17938.44 |
| AIR-DEL-SIN | SEA|mixed|air | True | 2926.00 | 10000.00 | 2949.76 | 0.67 | 16317.74 |
| DXB-NYE | GCC|event|hotel | False | 7274.75 | 10000.00 | 7321.01 | 0.00 | 0.00 |
| DXB-DEC | GCC|leisure|hotel | False | 4285.56 | 10000.00 | 4418.10 | 0.00 | 0.00 |
| AIR-BOMDXB | GCC|mixed|air | False | 3860.00 | 10000.00 | 3937.62 | 0.00 | 0.00 |
| BKK-SONGKRAN | SEA|event|hotel | False | 2236.60 | 10000.00 | 2178.53 | 0.00 | 0.00 |
| IST-MIX | TUR|mixed|hotel | False | 2364.05 | 10000.00 | 2455.02 | 0.00 | 0.00 |
| DXB-TRANS | GCC|mixed|transfer | False | 543.45 | 10000.00 | 491.71 | 0.00 | 0.00 |
| DXB-FREE-JAN | GCC|leisure|hotel | False | 1958.80 | 10000.00 | 2047.36 | 0.00 | 0.00 |
| BKK-FREE-MAR | SEA|leisure|hotel | False | 1421.16 | 10000.00 | 1512.23 | 0.00 | 0.00 |
| DXB-BIZ-Q1 | GCC|business|hotel | False | 3648.40 | 10000.00 | 3821.85 | 0.00 | 0.00 |
| IST-FREE-SPR | TUR|leisure|hotel | False | 1466.74 | 10000.00 | 1502.81 | 0.00 | 0.00 |
| HKT-FREE-MAY | SEA|leisure|hotel | False | 1488.60 | 10000.00 | 1451.58 | 0.00 | 0.00 |
| MLE-FREE-FEB | IO|leisure|hotel | False | 5455.60 | 10000.00 | 5435.72 | 0.00 | 0.00 |
| DXB-WKD | GCC|leisure|hotel | False | 2216.92 | 10000.00 | 2227.47 | 0.00 | 0.00 |
| BKK-WKD | SEA|leisure|hotel | False | 1537.17 | 10000.00 | 1640.12 | 0.00 | 0.00 |
| IST-WKD | TUR|business|hotel | False | 1886.90 | 10000.00 | 1774.41 | 0.00 | 0.00 |
