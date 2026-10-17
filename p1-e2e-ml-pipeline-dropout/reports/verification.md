# P1 verification (2026-10-11)

- Fresh build in throwaway copy (new venv): make setup, data, train OK. Wall: setup ~2 s (uv cache), data 39.6 s, train 18.1 s, peak RSS 510 MB.
- Reproducibility: fresh metrics.json equals real p1 exactly except MLflow run_id.
- make verify (real p1): lint OK, 23 tests passed, smoke OK; latency p50 1.29 ms, p95 1.83 ms.
- Serve flow: /health 200; /predict 200 (3 top features); /predict/batch 200 (3 rows); out-of-range 422; empty body 422; batch of 102 -> 422. make stop leaves :8000 free.
- Gradio/ui: skipped by decision; make ui prints "not used".
- Architect review fixed directly: resources.md added; PLAN.md gradio/boto3 rows corrected; make clean no longer deletes committed reports.
- Open items needing user decision: (A) build or drop Gradio (and remove gradio pin from requirements.txt); (B) Lambda libgomp unverified, zip 52.7 MB > 50 MB direct limit; (C) no checksum/snapshot of UCI data, license to README; (D) fairness gap (flag rate male 0.594 vs female 0.35; <20 age precision 0.46) - acknowledge in README; (E) .gitignore lacks mlflow.db, build/, .pytest_cache, .ruff_cache.
- Small open items: API RAM unmeasured; train_time.txt has log noise; make eval only prints metrics; README pending (/readme).
