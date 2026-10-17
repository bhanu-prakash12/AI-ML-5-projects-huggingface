# P4 Decisions
| Date | Decision | Reason (alternative) |
|---|---|---|
| 2026-10-11 | `small` profile (8 GB) | User rule |
| 2026-10-11 | Evidently 0.4.40 legacy API (Report/TestSuite/presets) | Stable, spec pin; alt: 0.7+ new API (not needed for demo) |
| 2026-10-11 | Pins as CLAUDE.md global + lightgbm==4.5.0 evidently==0.4.40 pyarrow==18.1.0; dry-run resolved cleanly. After install, `sqlalchemy==2.0.36` added (unpinned resolve gave 2.1.4, which breaks mlflow 2.19.0 import) | Nearest compatible pin |
| 2026-10-11 | Ports API 8004 / UI 7864 / MLflow 5004 (configs/config.yaml `ports`) | Parallel agents share the machine |
| 2026-10-11 | Data: OpenML 1597 cached to parquet. This OpenML copy has NO `Time` column; rows are already in original time order, so row order = time order and split is by row index (60/40) | Verified by inspecting columns |
| 2026-10-11 | Features = V1–V28 + Amount (29). No Time | Not available |
| 2026-10-11 | Stream sample (small): keep ALL 132 stream frauds + 22,000 random non-fraud (seed 42), time order kept → 22,132 rows, 0.6% fraud, 10 "days" of ~2,213 rows. Documented as an enrichment: stream base rate is higher than the real 0.12% so per-window PR-AUC is measurable | Full stream (114k rows, ~13 frauds/day at 1 window = 10) too noisy and slow; alt: 3 windows of full data |
| 2026-10-11 | Reference for drift = random 20k rows of the reference part; model trained on first 80% of reference, last 20% (time order) = validation → baseline PR-AUC | Keeps Evidently fast; honest baseline |
| 2026-10-11 | Threshold 0.5 for precision/recall; PR-AUC is the primary metric and drives the alert rule | Simple; threshold tuning is an exercise |
| 2026-10-11 | Alert rule exactly as spec (drift share > 0.3 OR PR-AUC drop > 10% vs baseline) plus guard: PR-AUC rule only applies when ≥10 labelled frauds in the window (alert-fatigue guard; logged as `perf_skipped`) | Windows have ~5-23 frauds; PR-AUC on 5 frauds is noise |
| 2026-10-11 | Delayed labels: simulator sends /feedback for day d-1 after day d predictions (lag 1 day). Window w performance is available when processed with as-of day w+1 | Models real label delay |
| 2026-10-11 | Model registry via MLflow SQLite (artifacts/mlflow.db, mlruns under artifacts/); API loads alias `champion`, `/reload` swaps it | Spec: MLflow tracking/registry |
| 2026-10-11 | Scenario default for the demo = B starting day 6 (days 1–5 clean) | Spec acceptance |
| 2026-10-11 | Pin plotly==5.24.1 | Evidently 0.4.40 report HTML needs plotly.figure_factory.create_distplot, removed in plotly 7 (alt: skip ClassificationPreset) |
| 2026-10-11 | Drift reference = stream days 1–2 (ref_days: 2; windows 3–10 monitored). Measured: vs the 20k training-period reference 83% of features drift on CLEAN days (max normalised Wasserstein 0.86σ), but day-vs-day inside the stream: 0% | Natural covariate shift between training period and stream would alert every window = alert fatigue. Training-period reference kept via `ref_days: 0`. Alt: raise thresholds (hides real drift) |
| 2026-10-11 | Demo-readiness: `run.py` (stdlib runner, Windows/macOS), `scripts/smoke.py` + `run.py demo` replace the bash `smoke.sh`/`demo.sh`; Makefile delegates smoke/demo/stop/clean | Windows VS Code terminal support; Windows logic unit-tested via monkeypatch, no Windows run |
| 2026-10-11 | `/predict` OpenAPI examples = one real legit + one real fraud row from data/reference.parquet, keys from reports/baseline.json (29 features: V1-V28 + Amount, not 30), built lazily; zero-valued fallback if files are absent | No invented values |
| 2026-10-11 | Demo writes to `reports/demo/` (gitignored) and `.run/*.db`, never over the saved `reports/` runs; dashboard dropdown lists demo + saved runs | Existing outputs stay intact |
| 2026-10-11 | Demo/dashboard retrain is a dry run by default (`--promote` / checkbox moves the champion alias); demo first resets the alias to the baseline version (`--keep-champion` to skip) | Repeatable story; v2 (trained on scenario B) would mute scenario B |
| 2026-10-11 | Dashboard buttons run the API in-process (Starlette TestClient) instead of needing a separate server | One command (`run.py ui`) is enough for a click-driven demo |
| 2026-10-11 | Repaired absolute `p4/mlruns` paths in the local, gitignored `artifacts/mlflow.db` after the folder rename (tables experiments, runs, model_versions.source/storage_location) | API could not load any model version; data-only fix |
| 2026-10-11 | `rich==13.9.4` added (console narration, plain-print fallback, honours NO_COLOR) | Only new dependency |
