# Baseline Forecast Metrics

- **Data:** censored observed demand only from `data/history/booking.parquet` (latent evaluation table not used).
- **Split:** rolling-origin by stay date with embargo gap = 14 days (where folds available); no stay-date overlap.
- **Note:** All numbers are simulator outputs under stated assumptions, not TBO real P&L. These are the bar later models must beat.

| fold | model | n_test | WAPE | MASE_season7 | pinball_q10 | pinball_q50 | pinball_q90 | PICP_80 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | seasonal_naive_yoy | 3 | 0.391167 | nan | 695.600000 | 1432.000000 | 564.560000 | 0.333333 |
| 1 | seasonal_naive_yoy | 1 | 0.620551 | 0.684837 | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
| 2 | seasonal_naive_yoy | 2 | 0.983206 | 1.000064 | 438.390000 | 1217.750000 | 438.390000 | 0.000000 |

## PIT histogram counts (per fold, 10 bins over [0,1])

- fold 0 `seasonal_naive_yoy`: [0, 0, 0, 1, 0, 0, 1, 0, 0, 1]
- fold 1 `seasonal_naive_yoy`: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
- fold 2 `seasonal_naive_yoy`: [0, 0, 0, 0, 0, 1, 0, 0, 0, 1]

