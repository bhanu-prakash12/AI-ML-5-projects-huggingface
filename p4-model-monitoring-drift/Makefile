PY=.venv/bin/python
export PYTHONPATH=src
.PHONY: setup data train baseline simulate monitor ui eval serve mlflow-ui test lint smoke verify stop clean retrain demo
setup:
	uv venv --python 3.11 .venv && uv pip install --python $(PY) -r requirements-dev.txt
data:
	$(PY) scripts/fetch_data.py
train:
	$(PY) scripts/train_baseline.py
simulate:
	$(PY) scripts/simulate.py
monitor:
	$(PY) scripts/monitor.py
retrain:
	$(PY) scripts/retrain.py
eval:
	@echo "not used in this project"
serve:
	$(PY) -m uvicorn fraud_monitor.api:app --port $$($(PY) -c "from fraud_monitor import config;print(config.load()['ports']['api'])") --workers 1
ui:
	$(PY) scripts/ui.py
mlflow-ui:
	$(PY) run.py mlflow-ui
test:
	$(PY) -m pytest -q
lint:
	$(PY) -m ruff check src tests scripts run.py
smoke:
	$(PY) scripts/smoke.py
verify: lint test smoke
demo:
	$(PY) run.py demo
stop:
	$(PY) run.py stop
clean:
	$(PY) run.py clean
