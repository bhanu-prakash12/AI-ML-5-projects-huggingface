# macOS/Linux convenience; Windows users: `python run.py <target>` (same targets).
PY=.venv/bin/python
export PYTHONPATH := src
.PHONY: package setup data train eval serve ui mlflow-ui test lint smoke verify demo stop clean
package:
	$(PY) scripts/build_lambda.py
setup:
	uv venv --python 3.11 .venv && uv pip install --python $(PY) -r requirements-dev.txt
data:
	$(PY) scripts/make_data.py && $(PY) scripts/eda.py
train:
	$(PY) scripts/train.py
eval:
	$(PY) scripts/evaluate.py
serve:
	$(PY) -m uvicorn dropout_risk.api:app --port 8000
ui:
	$(PY) scripts/ui.py
mlflow-ui:
	$(PY) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
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
