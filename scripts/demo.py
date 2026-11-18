"""One-command demo smoke test: regenerate data, train/fit, plan, and print headline results.

All numbers are simulator outputs under stated assumptions, not TBO real P&L.
Assumption: "train" step runs Phase 5 demand-distribution pipeline (quantile fit + CQR/unconstraining
report on censored history) and planner fits its forecast model for allocation; this is a demo-time
training/calibration pass, not a production training service.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], cwd: Path | None = None) -> int:
    print("\n$ " + " ".join(cmd))
    p = subprocess.run(cmd, cwd=str(cwd or ROOT))
    return int(p.returncode)


def main() -> int:
    ap = argparse.ArgumentParser(description="TBO Inventory Optimization demo smoke test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--budget", type=float, default=10_000_000.0)
    ap.add_argument("--quick-eval", action="store_true", help="use smaller seeds/scenarios for faster run")
    ap.add_argument("--skip-train", action="store_true", help="skip Phase 5 train/report step")
    args = ap.parse_args()

    py = sys.executable
    rc = 0

    # 1) Regenerate data
    rc |= _run([py, "-m", "sim", "--seed", str(args.seed), "--out", "data"])
    rc |= _run([py, "scripts/export_api_demo.py"])

    # 2) Train/calibrate (Phase 5 report uses censored history only; latent read only in /eval)
    if not args.skip_train:
        rc |= _run([py, "-m", "eval.phase5_report"])

    # 3) Plan (fits forecast on censored history and runs candidate funnel + water-filling)
    rc |= _run([py, "-m", "optimizer.planner", "--budget", str(args.budget), "--scenarios", "300", "--seed", str(args.seed)])

    # 4) Headline baselines/planner evaluation table
    seeds = 5 if args.quick_eval else 10
    scenarios = 120 if args.quick_eval else 200
    out_md = "docs/baseline_eval_demo.md" if args.quick_eval else "docs/baseline_eval.md"
    rc |= _run([py, "-m", "eval.run_baselines", "--budget", str(args.budget), "--seeds", str(seeds), "--scenarios", str(scenarios), "--out", out_md])

    print("\n=== DEMO COMPLETE ===")
    print(f"exit_code={rc} (0 = all steps succeeded)")
    print("Artifacts: data/history/*.parquet, data/ground_truth.parquet, data/api_demo/*, docs/phase5_report.md (if trained), planner stdout table, " + out_md)
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
