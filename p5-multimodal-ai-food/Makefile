# macOS/Linux convenience wrapper; the cross-platform entry point is `python run.py <cmd>`.
# run.py finds .venv itself, so any python3 works here.
.PHONY: demo setup models data train eval serve ui mlflow-ui test lint smoke verify stop clean
setup models data train eval serve ui mlflow-ui test lint smoke verify demo stop clean:
	python3 run.py $@
