.PHONY: setup test sim export-api train plan ui demo demo-quick api

PY ?= python

setup:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e .
	$(PY) -m pip install streamlit plotly requests matplotlib fastparquet

test:
	$(PY) -m pytest -q

sim:
	$(PY) -m sim --seed 42 --out data

export-api:
	$(PY) scripts/export_api_demo.py

train:
	$(PY) -m eval.phase5_report

plan:
	$(PY) -m optimizer.planner --budget 10000000 --scenarios 300 --seed 42 --reserve-fraction 0.03

ui:
	$(PY) -m streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0

api:
	cd api && go run .

demo:
	$(PY) scripts/demo.py --seed 42 --budget 10000000

demo-quick:
	$(PY) scripts/demo.py --seed 42 --budget 10000000 --quick-eval

