# Phase 5 Demand Distribution Report

- **Training/model inputs:** censored history only; latent evaluation table never used in forecast/optimizer code.
- **Latent ground truth:** read only in `/eval` for unconstraining comparison and chart.
- **Note:** All numbers are simulator outputs under stated assumptions, not TBO real P&L.
- **Assumption (CQR/PICP):** 90% intervals are CQR-calibrated and PICP is evaluated on uncensored holdout rows (`is_sold_out == False`, so `observed_sales == realized demand`) from censored history on embargoed time blocks; sold-out/censored rows are excluded from PICP because true demand above capacity is unobserved in history and latent ground truth is eval-only. This targets observed-demand coverage under the censoring constraint with no latent leakage. split=contiguous_date_blocks_embargoed, embargo=14d, train/calib/test rows=1620/450/1350, uncensored calib/test=339/764.

## Model vs Baseline (Phase 4 bar)

| metric | value |
| --- | --- |
| model_WAPE_test | 0.580507 |
| model_MASE_season7_test | 1.018480 |
| model_pinball_q50_test | 17.284650 |
| model_pinball_q90_test | 6.602932 |
| model_PICP90_raw_test | 0.214660 |
| model_PICP90_after_CQR_test | 0.890052 |
| cqr_q_lower_adjust | 21.913338 |
| cqr_q_upper_adjust | 0.713288 |
| cqr_n_calib_uncensored | 339.000000 |
| n_train_rows | 1620.000000 |
| n_calib_rows | 450.000000 |
| n_test_rows | 1350.000000 |
| n_calib_uncensored_rows | 339.000000 |
| n_test_uncensored_rows | 764.000000 |
| baseline_mean_WAPE | 0.664975 |
| baseline_mean_pinball_q50 | 883.250000 |
| beats_baseline_wape_and_pinball | 1.000000 |
| uncon_mae_raw_vs_latent_soldout | 31.051918 |
| uncon_mae_em_vs_latent_soldout | 100.369036 |
| uncon_mae_pd_vs_latent_soldout | 24.343649 |
| uncon_mae_best_vs_latent_soldout | 24.343649 |
| uncon_best_method | projection_detruncation |
| uncon_best_beats_raw_soldout | 1.000000 |
| scenario_matrix_shape_Sxn | 500x27 |
| waterfill_spend_Rs_1Cr | 10000000.000000 |
| waterfill_lambda | 25822498780869086436831993248681894182548833048615888584192974690338880141698009435618943865039225109799021576192.000000 |

## Baseline comparison (honest)

Model beats Phase 4 seasonal-naive baseline on both mean WAPE and pinball(q50).

## 90% interval coverage (CQR, held-out time block)

- PICP90 raw: 0.2147
- PICP90 after CQR: 0.8901 (target 0.88–0.92)
- Split: {'split_mode': 'contiguous_date_blocks_embargoed', 'train_end': '2026-12-21', 'calib_start': '2027-01-17', 'calib_end': '2027-02-14', 'test_start': '2027-03-10', 'test_end': '2027-06-27', 'embargo_days': 14}
- Calibrated coverage is within target band on this held-out block.

## Unconstraining vs latent demand (sold-out dates, eval-only)

- Chart: `docs/unconstraining_before_after.png`
- MAE raw vs latent: 31.0519
- MAE EM-unconstrained vs latent: 100.3690
- MAE projection-detruncation vs latent: 24.3436
- Best unconstrained method: projection_detruncation (MAE 24.3436)
- Best unconstrained closer than raw: True

## Scenario matrix -> optimizer/waterfill integration

- demand matrix shape [S,n]: (500, 27) (cells ordered as sim/optimizer Phase 2 list)
- `waterfill(opt_cells, demand_s, budget=10_000_000)` spend=10000000.00, lambda=25822498780869086436831993248681894182548833048615888584192974690338880141698009435618943865039225109799021576192.00000000
