# AI Engineering Projects — Build Rules (macOS Apple Silicon, laptop-first)

## Mission
Build 5 self-contained, production-shaped, beginner-friendly AI/ML projects (p1–p5) that run
natively on a MacBook with Apple Silicon (baseline: 8 GB unified memory), with NO Docker,
the smallest practical models, and the fewest possible tools. Repo is local git now and is
pushed to GitHub later, so keep it portable and small. Audience: engineers new to AI
engineering. You act as senior AI/ML engineer AND technical architect.

## Allowed toolchain (do not add system software without asking)
- Xcode Command Line Tools (git), Homebrew (ONLY for `libomp`), uv 0.5.11, Claude Code.
  No Ollama or other model servers, no Docker, no Node/npm, no database servers, no Prometheus/Grafana, no cloud
  accounts required (AWS is optional in P1 only).
- Python 3.11 via `uv python install 3.11`; one venv per project: `uv venv --python 3.11 .venv`;
  install with `uv pip install -r requirements.txt`.
- Storage: files + SQLite only. Vector DB: Chroma embedded. Tracking: MLflow (SQLite).
- UI: Gradio only (one UI framework across all projects). API: FastAPI + uvicorn.

## Local model runtime (all LLM/VLM use)
- Models run IN-PROCESS with Hugging Face `transformers` on MPS. No model server, no API keys.
- Approved small models (smallest that does the job; ask before using anything else):
  text: Qwen/Qwen2.5-1.5B-Instruct (default), Qwen/Qwen2.5-0.5B-Instruct (fallback)
  vision: Qwen/Qwen2-VL-2B-Instruct (default), HuggingFaceTB/SmolVLM-500M-Instruct (fallback)
  embeddings: BAAI/bge-small-en-v1.5; image-text: CLIP ViT-B-32 (open_clip)
- Each project has `src/<pkg>/llm.py` with the same shape: `load()` (lazy singleton, fp16 on
  MPS for inference, fp32 on CPU), `generate(messages, max_new_tokens)` using
  `tokenizer.apply_chat_template`, and `stream(...)` using `TextIteratorStreamer`.
  Wrap calls in `torch.inference_mode()`. FastAPI loads models once in the lifespan hook.
- `scripts/download_models.py` uses `huggingface_hub.snapshot_download` with a pinned
  `revision` (commit hash recorded in DECISIONS.md). After download, run with
  HF_HUB_OFFLINE=1 so the app is fully local.
- Structured output: prompt for JSON only → extract first JSON object → validate with
  pydantic → one retry with the validation error appended → else return a typed error.

## Apple Silicon rules
- `src/<pkg>/device.py: get_device()` → "mps" if torch.backends.mps.is_available() else "cpu".
- Set in .env.example: PYTORCH_ENABLE_MPS_FALLBACK=1, TOKENIZERS_PARALLELISM=false,
  HF_HUB_OFFLINE=0 (set to 1 after models are downloaded).
- Train in fp32 on MPS (fp16/bf16 training on MPS is unreliable).
- Never hold two large models in memory at once unless the spec says so.
- Ports: API 8000 (p3 8003, p4 8004, p5 8005), Gradio 7860 (p1 7861, p3 7863, p4 7864, p5 7865), MLflow UI 5001 (5000 is taken by macOS AirPlay Receiver).
- Every project also ships a stdlib `run.py` (Windows-friendly; same targets as the Makefile).

## Context & token budget (strict)
- Per session read ONLY: this file, PROGRESS.md, docs/specs/<pN>.md, <pN>-*/PLAN.md, <pN>-*/DECISIONS.md.
- Never read data/, artifacts/, mlruns/, chroma/, ~/.cache/huggingface, reports/*.html, .venv/, large files.
  Inspect with `head -n 5`, `wc -l`, `du -sh`, `python -c` one-liners.
- To understand existing code use `grep -n "def \|class " -r src/` before opening files.
- Never echo file contents back. After writing files, summarize in ≤5 bullets.
- Test/log output: `pytest -q`, `tail -n 40`; show failures only.
- ONE phase per session. End of phase: update PROGRESS.md, commit, tell user "run /clear".

## Decision rule
- Ambiguous, conflicting, unavailable, or a pin fails → STOP, ask with numbered options +
  your recommendation. Log answers in <pN>-*/DECISIONS.md (date, decision, reason).
- Never invent data values (nutrition, medical facts, metrics, benchmarks).
  Every number in a README must come from a real run saved under reports/.

## Global pins (Python 3.11, macOS arm64 wheels from PyPI — no extra index needed)
numpy==1.26.4 pandas==2.2.3 scikit-learn==1.5.2 pyyaml==6.0.2 pydantic==2.10.3
pydantic-settings==2.7.0 fastapi==0.115.6 uvicorn==0.32.1 httpx==0.28.1 requests==2.32.3
python-dotenv==1.0.1 mlflow==2.19.0 matplotlib==3.9.4 python-multipart==0.0.20
torch==2.5.1 transformers==4.47.1 sentence-transformers==3.3.1 datasets==3.2.0
accelerate==1.2.1 huggingface-hub==0.27.0 gradio==5.9.1
dev: pytest==8.3.4 ruff==0.8.4
Only add a package when the spec lists it. Phase 0 verifies with
`uv pip install --dry-run -r requirements.txt`. If a pin fails, propose the nearest
compatible version, ask, and log it in DECISIONS.md.

## Project layout (identical for all)
pN-<name>/ README.md PLAN.md DECISIONS.md Makefile requirements.txt requirements-dev.txt
.env.example configs/config.yaml src/<pkg>/ tests/ scripts/ data/ artifacts/ reports/
(data/ and artifacts/ gitignored, keep .gitkeep)

## Code standards
- Typed Python 3.11, small modules, config from YAML + pydantic-settings, no hardcoded paths.
- `logging` not print; seed 42 everywhere; deterministic splits.
- Profiles in config: `small` (8 GB, DEFAULT) and `standard` (16 GB+, opt-in).
- Unit tests use tiny fixtures, no network, total < 60 s. Integration tests marked `@pytest.mark.slow`.
- Same Makefile targets everywhere: setup data train eval serve ui mlflow-ui test lint
  smoke verify stop clean (N/A targets print "not used in this project").
- `make smoke` starts the API in the background (PID in .run/api.pid), waits for /health,
  runs 1–3 real requests, then stops it. `make stop` kills anything in .run/*.pid.
- `make verify` = lint + test + smoke.
- Record peak RAM (`/usr/bin/time -l`) and wall time of heavy steps in reports/resources.md.

## Definition of Done (every phase)
[ ] code + tests  [ ] `make verify` green  [ ] spec acceptance check passed, ≤10 lines of
proof in PROGRESS.md  [ ] DECISIONS.md updated  [ ] commit "pN: phase X – <title>"

## README standard (written by /readme, identical structure for all 5)
1 Title, shields.io badges, one-line pitch   2 What you'll learn (concept table)
3 Architecture (mermaid) + layer table: layer | tool | exact version | responsibility
4 Tech stack, why chosen vs alternatives (incl. why no Docker here)
5 Prerequisites: macOS Apple Silicon, RAM/disk/time
6 Quickstart (≤10 commands)
7 Build from scratch: one step per phase → goal, concept intro, commands, files, expected output, checkpoint
8 Key workflows (sequenceDiagram for training, inference, project-specific flow)
9 Configuration reference   10 Testing & verification   11 Results (real numbers only)
12 Limitations & gotchas    13 Scaling 100 → 10K → 1M users (table + target mermaid + cost
   drivers; containers/cloud belong HERE as the scaling path, not in the local build)
14 Troubleshooting (macOS-specific first)   15 Exercises / next steps
16 References + dataset/model licenses
Style: senior engineer mentoring juniors. At first use of each new concept/tool add a
GitHub alert box: "> [!TIP] **<Concept>** — What / Why / How we use it here."
Use GitHub alerts (NOTE/TIP/WARNING), tables, <details> for long code, ≤1 emoji per heading.
Mermaid: only flowchart and sequenceDiagram; quote labels with special chars; colors via:
classDef data fill:#E3F2FD,stroke:#1565C0,color:#0D47A1
classDef model fill:#E8F5E9,stroke:#2E7D32,color:#1B5E20
classDef serve fill:#FFF3E0,stroke:#EF6C00,color:#E65100
classDef monitor fill:#F3E5F5,stroke:#6A1B9A,color:#4A148C
classDef store fill:#ECEFF1,stroke:#455A64,color:#263238
