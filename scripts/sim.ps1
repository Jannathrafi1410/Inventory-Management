$ErrorActionPreference = "Stop"
python -m sim --seed 42 --out data
Write-Host "Synthetic tables written to data/ (history/*.parquet, ground_truth.parquet)."
