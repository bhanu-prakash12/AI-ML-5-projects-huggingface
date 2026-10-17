# Resources (small profile, macOS arm64)

| Step | Wall time | Max RSS | Peak footprint | Source |
|---|---|---|---|---|
| make data | 39.6 s (fresh copy, incl. UCI fetch) | not measured | not measured | /usr/bin/time shell |
| make train (real p1) | 11.05 s | 500,252,672 B | 424,494,688 B | reports/train_time.txt |
| make train (fresh copy) | 18.11 s | 510,361,600 B | not recorded | fresh-build run 2026-10-11 |
| API serve RAM | not measured | - | - | open item |
| run.py train (Python-measured, 2026-10-11) | 8.55 s | 480,788,480 B | not recorded | reports/train_time.txt |
