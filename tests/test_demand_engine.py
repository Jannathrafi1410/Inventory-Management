"""Tests for demand-distribution engine (Phase 5).

Covers point-in-time feature safety, unconstraining, conformal/CQR, cancellation, reconciliation,
scenario matrix -> waterfill integration, and gate checks (PICP, unconstraining vs latent in eval).
Assumption: CQR adjustment is validated on held-out test block in report; tests use deterministic
synthetic/calibrated cases and invoke eval-only report functions as specified. All simulator outputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from eval.metrics import picp
from forecast.cancellations import CancellationModel
from forecast.conformal import apply_cqr, cqr_calibrate
from forecast.features import build_point_in_time_features
from forecast.reconcile import compute_child_shares, reconcile_cluster_to_children
from forecast.scenarios import scenarios_for_cells
from forecast.splits import rolling_origin_splits
from forecast.unconstrain import em_unconstrain_lognormal, em_unconstrain_normal, projection_detruncation
from optimizer.demo_phase2 import sim_cell_to_optimizer
from optimizer.waterfill import waterfill
from sim.cells import get_candidate_cells


def test_splits_embargo_no_leakage() -> None:
    dates = pd.date_range("2026-10-01", periods=120, freq="D")
    folds = rolling_origin_splits(dates, forecast_horizon_days=7, test_window_days=14, step_days=14, min_train_days=20)
    assert folds
    for f in folds:
        assert (f.test_start_stay - f.train_end_stay).days > f.embargo_days


def test_features_point_in_time_and_required_fields() -> None:
    feats = build_point_in_time_features(decision_ts=None, history_dir="data/history")
    assert not feats.empty
    cutoff = pd.to_datetime(feats["decision_ts"]).median()
    feats_cut = build_point_in_time_features(decision_ts=cutoff, history_dir="data/history")
    assert (pd.to_datetime(feats_cut["decision_ts"]) <= cutoff).all()
    for col in ["stay_month", "stay_week", "lead_time_days", "pace_vs_plan", "price_rank", "sellout_rate_lead", "event_decay", "fx_factor", "yoy_lag"]:
        assert col in feats.columns
    for w in [4, 6, 8, 10, 12]:
        assert f"search_lag_{w}w" in feats.columns


def test_em_unconstrain_normal_and_lognormal_closer_than_censored() -> None:
    rng = np.random.default_rng(0)
    d = rng.normal(100.0, 20.0, size=1500)
    cap = 110.0
    cens = d >= cap
    y = np.minimum(d, cap)
    res = em_unconstrain_normal(y, cens, s=np.full_like(y, cap), max_iter=150)
    assert abs(np.mean(res.unconstrained) - np.mean(d)) < abs(np.mean(y) - np.mean(d))
    d2 = rng.lognormal(mean=np.log(80), sigma=0.3, size=1200)
    cap2 = np.quantile(d2, 0.7)
    cens2 = d2 >= cap2
    y2 = np.minimum(d2, cap2)
    res2 = em_unconstrain_lognormal(y2, cens2, s=np.full_like(y2, cap2), max_iter=120)
    pd_est = projection_detruncation(y2, cens2, s=np.full_like(y2, cap2))
    assert np.mean(res2.unconstrained[cens2]) >= cap2 - 1e-6
    assert np.mean(pd_est[cens2]) >= cap2 - 1e-6


def test_cqr_and_picp_computation() -> None:
    rng = np.random.default_rng(1)
    y = rng.normal(50, 10, size=400)
    # intentionally mis-scaled intervals to exercise CQR monotonicity and calibration effect
    lo = y - 20 + rng.normal(0, 1, size=400)
    hi = y + 2 + rng.normal(0, 1, size=400)
    calib = cqr_calibrate(y[:200], lo[:200], hi[:200], alpha=0.1)
    lo2, hi2 = apply_cqr(lo[200:], hi[200:], calib)
    assert np.all(hi2 >= lo2)
    assert np.all(lo2 >= 0.0)
    cov = picp(y[200:], lo2, hi2)
    assert 0.0 <= cov <= 1.0
    # symmetric exchangeable case: calibrated coverage should be close to nominal
    base = y + rng.normal(0, 2, size=400)
    lo_s, hi_s = base - 3.0, base + 3.0
    calib_s = cqr_calibrate(y[:200], lo_s[:200], hi_s[:200], alpha=0.1)
    lo_s2, hi_s2 = apply_cqr(lo_s[200:], hi_s[200:], calib_s)
    cov_s = picp(y[200:], lo_s2, hi_s2)
    assert 0.85 <= cov_s <= 0.97



def test_cancellation_reconcile_and_scenarios_feed_waterfill() -> None:
    m = CancellationModel().fit(history_dir="data/history")
    assert 0.0 < m.mean_cancel_prob("leisure", 45) < 1.0
    hist = pd.DataFrame({"cluster_id": ["A", "A", "B", "B"], "cell_id": ["a1", "a2", "b1", "b2"], "y_censored": [10, 30, 50, 50]})
    shares = compute_child_shares(hist, cluster_col="cluster_id", child_col="cell_id", y_col="y_censored", shrinkage=0.5)
    rec = reconcile_cluster_to_children(pd.DataFrame({"cluster_id": ["A", "B"], "forecast": [100.0, 80.0]}), shares)
    sums = rec.groupby("cluster_id")["forecast_child"].sum()
    assert np.isclose(sums.loc["A"], 100.0) and np.isclose(sums.loc["B"], 80.0)

    sim_cells = get_candidate_cells()
    opt_cells = [sim_cell_to_optimizer(c) for c in sim_cells]
    qs = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    qf = pd.DataFrame([{f"q_{q}": max(c.base_demand_mean, 1.0) * (0.5 + q) for q in qs} for c in sim_cells])
    demand, _ = scenarios_for_cells(qf, [c.id for c in opt_cells], [c.group for c in opt_cells], s=200, quantiles=qs, intra_group_rho=0.6, seed=42)
    assert demand.shape == (200, len(opt_cells))
    res = waterfill(opt_cells, demand, budget=10_000_000.0)
    assert res.spend <= 10_000_000.0 + 1e-6


def test_unconstraining_beats_raw_on_soldout_latent_eval_only() -> None:
    from eval.unconstraining_report import evaluate_unconstraining

    metrics = evaluate_unconstraining(history_dir="data/history", ground_truth_path="data/ground_truth.parquet", out_chart="docs/unconstraining_before_after.png")
    assert metrics["n_sold_out"] > 0
    assert metrics["unconstrained_beats_raw"] == 1.0
    assert Path(metrics["chart_path"]).exists()


def test_phase5_report_generated_and_integration() -> None:
    from eval.phase5_report import run_phase5

    res = run_phase5(history_dir="data/history", ground_truth_path="data/ground_truth.parquet", out_report="docs/phase5_report.md")
    assert Path("docs/phase5_report.md").exists()
    assert Path("docs/unconstraining_before_after.png").exists()
    assert res["demand_shape"][1] == len(get_candidate_cells())
    assert res["waterfill_spend"] <= 10_000_000.0 + 1e-6
    text = Path("docs/phase5_report.md").read_text(encoding="utf-8")
    assert "Baseline comparison" in text
    # PICP reported on held-out test block after CQR calibration (target 0.88-0.92)
    assert "PICP90 after CQR" in text or "PICP90_after_CQR" in text
    assert 0.88 <= float(res["picp_90_cal"]) <= 0.92
    assert res.get("n_test", 0) > 0 and res.get("n_calib", 0) > 0
    assert bool(res["beats_baseline"]) is True
    assert bool(res["uncon_metrics"]["unconstrained_beats_raw"]) is True

