PY=.venv/bin/python
export PYTHONPATH=src
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
.PHONY: merge setup data baseline train eval serve ui mlflow-ui test lint smoke verify stop clean demo
setup:
	uv venv --python 3.11 .venv && uv pip install --python $(PY) -r requirements-dev.txt
data:
	$(PY) scripts/make_data.py
baseline:
	$(PY) scripts/run_baseline.py
train:
	$(PY) scripts/train.py
eval:
	$(PY) scripts/run_compare.py
serve:
	$(PY) -m uvicorn med_assistant.api:app --port 8003 --workers 1
ui:
	$(PY) scripts/ui.py
mlflow-ui:
	$(PY) -m mlflow ui --backend-store-uri sqlite:///$(CURDIR)/mlflow.db --port 5003
test:
	$(PY) -m pytest -q
lint:
	$(PY) -m ruff check src tests scripts run.py
smoke:
	$(PY) scripts/smoke.py
demo:
	$(PY) scripts/smoke.py --demo
verify: lint test smoke
stop:
	@for f in .run/*.pid; do [ -f "$$f" ] && kill $$(cat $$f) 2>/dev/null; rm -f $$f; done; true
clean:
	rm -rf .run .pytest_cache .ruff_cache
merge:
	$(PY) scripts/merge.py
