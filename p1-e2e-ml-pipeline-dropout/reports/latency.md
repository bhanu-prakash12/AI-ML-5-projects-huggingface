# Latency (single /predict, local uvicorn, 200 sequential requests)

| p50 ms | p95 ms | max ms |
|---|---|---|
| 2.17 | 4.96 | 14.28 |

Model: dropout-risk-v3. Measured by scripts/smoke_client.py over loopback HTTP.
