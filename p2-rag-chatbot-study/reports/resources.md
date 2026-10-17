# p2 resources (measured 2026-10-11, MacBook Apple Silicon, MPS fp16, offline)

| step | wall time | peak RAM |
|---|---|---|
| make index (bge-small, 2745 chunks) | ~30 s | 966 MB (phase 2 run) |
| API startup to /health ok (embeddings + Chroma + Qwen2.5-1.5B) | ~10 s | - |
| one /ask (in-corpus, streamed, with sources) | 6.1 s | API RSS 807 MB sampled via ps after the ask (sampled, not true peak) |
| make verify (lint + 20 tests + smoke with 2 real /ask) | 21.6 s | - |
| fresh copy: make setup (uv cache) / lint / test | 1.9 s / 0.5 s / 8.5 s | - |
