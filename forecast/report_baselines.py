"""Generate baseline forecast metrics report to docs/baseline_metrics.md.

Uses censored history only. Simulator outputs under stated assumptions, not TBO real P&L.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from eval.metrics import mase, picp, pinball_loss, pit_histogram_data, wape
from forecast.baseline import SeasonalNaiveModel, build_censored_demand_panel
from forecast.splits import apply_split, rolling_origin_splits


def _to_markdown_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    if not rows:
        return "_No rows_\n"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for r in rows:
        vals = []
        for c in columns:
            v = r.get(c, "")
            if isinstance(v, float):
                vals.append(f"{v:.6f}" if np.isfinite(v) else "nan")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def main(history_dir: str = "data/history", out_path: str = "docs/baseline_metrics.md") -> Path:
    panel = build_censored_demand_panel(history_dir)
    if panel.empty:
        raise SystemExit(f"No censored demand panel built from {history_dir}/booking.parquet")
    panel["stay_date"] = pd.to_datetime(panel["stay_date"]).dt.normalize()
    folds = rolling_origin_splits(
        panel["stay_date"], forecast_horizon_days=14, test_window_days=21, step_days=21, min_train_days=30
    )
    if folds:
        fold_rows = [apply_split(panel, f, stay_col="stay_date") for f in folds[:3]]
    else:
        dates = sorted(panel["stay_date"].unique().tolist())
        if len(dates) < 3:
            train, test = panel, panel.iloc[0:0].copy()
        else:
            cut = dates[max(1, int(len(dates) * 0.7))]
            train = panel[panel["stay_date"] <= cut]
            test = panel[panel["stay_date"] > cut]
        fold_rows = [(train, test)]

    rows: List[Dict[str, Any]] = []
    for i, (train, test) in enumerate(fold_rows):
        if train.empty or test.empty:
            continue
        model = SeasonalNaiveModel.fit(train, y_col="y_censored")
        yhat = model.predict(test)
        y = test["y_censored"].to_numpy(dtype=float)
        yt = train["y_censored"].to_numpy(dtype=float)
        resid = y - yhat
        q10 = yhat + np.quantile(resid, 0.10)
        q50 = yhat + np.quantile(resid, 0.50)
        q90 = yhat + np.quantile(resid, 0.90)
        # PIT proxy via empirical residual CDF rank
        ranks = np.array([np.mean(resid <= (yy - yh)) for yy, yh in zip(y, yhat)], dtype=float)
        counts, _ = pit_histogram_data(y, ranks, bins=10)
        rows.append(
            {
                "fold": i,
                "model": "seasonal_naive_yoy",
                "n_test": int(len(y)),
                "WAPE": float(wape(y, yhat)),
                "MASE_season7": float(mase(y, yhat, yt, seasonality=7)),
                "pinball_q10": float(pinball_loss(y, q10, 0.10)),
                "pinball_q50": float(pinball_loss(y, q50, 0.50)),
                "pinball_q90": float(pinball_loss(y, q90, 0.90)),
                "PICP_80": float(picp(y, q10, q90)),
                "PIT_counts": counts.tolist(),
            }
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["fold", "model", "n_test", "WAPE", "MASE_season7", "pinball_q10", "pinball_q50", "pinball_q90", "PICP_80"]
    with out.open("w", encoding="utf-8") as f:
        f.write("# Baseline Forecast Metrics\n\n")
        f.write("- **Data:** censored observed demand only from `data/history/booking.parquet` (latent evaluation table not used).\n")
        f.write("- **Split:** rolling-origin by stay date with embargo gap = 14 days (where folds available); no stay-date overlap.\n")
        f.write("- **Note:** All numbers are simulator outputs under stated assumptions, not TBO real P&L. These are the bar later models must beat.\n\n")
        if not rows:
            f.write("No evaluation rows produced (insufficient history/splits).\n")
        else:
            f.write(_to_markdown_table(rows, cols))
            f.write("\n## PIT histogram counts (per fold, 10 bins over [0,1])\n\n")
            for r in rows:
                f.write(f"- fold {int(r['fold'])} `{r['model']}`: {r['PIT_counts']}\n")
            f.write("\n")
    print(f"Wrote baseline metrics to {out}")
    return out


if __name__ == "__main__":
    main()
