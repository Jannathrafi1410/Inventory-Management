"""CLI entrypoint: `python -m sim` generates synthetic tables from a seed."""

from __future__ import annotations

import argparse
from pathlib import Path

from sim.generator import generate_and_write


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic TBO inventory data.")
    parser.add_argument("--seed", type=int, default=42, help="Master seed for reproducibility.")
    parser.add_argument("--out", type=str, default="data", help="Output directory for parquet tables.")
    args = parser.parse_args()
    out = generate_and_write(master_seed=args.seed, out_dir=Path(args.out))
    print(f"Wrote synthetic tables to {out} (history/*.parquet, ground_truth.parquet)")


if __name__ == "__main__":
    main()
