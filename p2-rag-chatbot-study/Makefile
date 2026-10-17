PY=.venv/bin/python
export PYTHONPATH=src
.PHONY: ingest demo index setup data train eval serve ui mlflow-ui test lint smoke verify stop clean
setup:
	uv venv --python 3.11 .venv && uv pip install --python $(PY) -r requirements-dev.txt
data:
	$(PY) scripts/fetch_corpus.py
train:
	@echo "not used in this project"
eval:
	$(PY) scripts/run_eval.py
serve:
	$(PY) -m uvicorn study_assistant.api:app --port $(or $(P2_API_PORT),8000) --workers 1
ui:
	$(PY) scripts/ui.py
mlflow-ui:
	@echo "not used in this project"
test:
	$(PY) -m pytest -q
lint:
	$(PY) -m ruff check src tests scripts run.py
smoke:
	$(PY) scripts/smoke.py
ingest:
	$(PY) scripts/ingest.py
index: ingest
	$(PY) scripts/probe.py
demo:
	$(PY) scripts/rag_demo.py
verify: lint test smoke
stop:
	@for f in .run/*.pid; do [ -f "$$f" ] && kill $$(cat $$f) 2>/dev/null; rm -f $$f; done; true
clean:
	rm -rf .run chroma data/corpus reports/*.json
