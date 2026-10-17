# P4 Fraud Model Monitoring — Plan
Goal: serve a LightGBM fraud model, log every prediction, inject a new fraud pattern, detect drift + decay, alert, retrain champion/challenger. Profile `small` (8 GB), no Docker.

```mermaid
flowchart LR
  D[(OpenML creditcard parquet)] --> T[train baseline LightGBM]
  T --> R[(MLflow registry: champion)]
  R --> A[FastAPI /predict /feedback /health]
  S[simulate.py + scenarios] --> A
  A --> L[(SQLite prediction log)]
  L --> M[monitor: Evidently per window + alert rule]
  M --> W[reports/windows/*.html, status.json, alerts.jsonl]
  W --> U[Gradio dashboard]
  L --> C[retrain: challenger vs champion]
  C --> R
  classDef data fill:#E3F2FD,stroke:#1565C0,color:#0D47A1
  classDef model fill:#E8F5E9,stroke:#2E7D32,color:#1B5E20
  classDef serve fill:#FFF3E0,stroke:#EF6C00,color:#E65100
  classDef monitor fill:#F3E5F5,stroke:#6A1B9A,color:#4A148C
  classDef store fill:#ECEFF1,stroke:#455A64,color:#263238
  class D data
  class T,C model
  class A,S,U serve
  class M,W monitor
  class R,L store
```
Layout: `src/fraud_monitor/{config,data,model,store,api,scenarios,monitor,retrain}.py`, `scripts/*.py`, `configs/config.yaml`, `tests/`.
Phases (spec): 0 plan/scaffold; 1 baseline; 2 serving+log; 3 simulator; 4 monitoring; 5 dashboard; 6 retrain; 7 verify+readme.
Risks: PR-AUC noise at ~10 frauds/window (guard, see DECISIONS); natural drift on clean days could false-alert (tune on real run); Evidently 0.4.40 speed (20k ref sample).
Acceptance per phase = spec lines; proof in PROGRESS.md.
