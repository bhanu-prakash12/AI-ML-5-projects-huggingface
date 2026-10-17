# P5 Decisions
| Date | Decision | Reason |
|---|---|---|
| 2026-10-11 | `small` profile (8 GB laptop) | User rule |
| 2026-10-11 | VLM = HuggingFaceTB/SmolVLM-500M-Instruct (rev a7da5b98...), not Qwen2-VL-2B (not in HF cache; ~4.5 GB download, tight on 8 GB). Alt: Qwen2-VL-2B in `standard` profile | Demo, token/speed/memory |
| 2026-10-11 | CLIP = laion/CLIP-ViT-B-32-laion2B-s34B-b79K rev 1a25a446..., loaded by open_clip from the pinned local snapshot (open_clip_model.safetensors only) | Pinned revision; avoids ~1.1 GB duplicate weights |
| 2026-10-11 | Pins: spec pins + torchvision==0.20.1 (open_clip dep, matches torch 2.5.1) + CLAUDE.md global pins incl. transformers==4.47.1; dry-run resolved cleanly, no conflicts | uv dry-run in temp venv |
| 2026-10-11 | FDC key: user has none known -> DEMO_KEY fallback with throttle/backoff; override with FDC_API_KEY. Alt: ask user | Demo |
| 2026-10-11 | Ports API 8005 / UI 7865 / MLflow 5005 (parallel agents) | Orchestrator instruction |
| 2026-10-11 | Add pin pyarrow==18.1.0 (unpinned resolve picked a pyarrow needing NumPy 2; breaks numpy==1.26.4 import) | Runtime import failure |
| 2026-10-11 | Nutrition source: USDA bulk CSVs (SR Legacy + FNDDS) by default; FDC API only when FDC_API_KEY set (replaces DEMO_KEY fallback: ~10 calls/h cannot cover 101 classes) | User decision |
| 2026-10-11 | data/ files never opened; build_nutrition.py prints aggregate counts only and writes reports/mapping_review.md (review list + summary) | User decision |
| 2026-10-11 | VLM structured output: SmolVLM-500M cannot emit the 5-field JSON in one go (0/6 valid; copies templates, truncates). Now one short free-text question per field, JSON assembled in code, then pydantic validate + one retry + typed error kept. Validator rejects kcal/protein claims in VLM text. 6/6 valid in reports/vlm_demo.md | Real runs |
| 2026-10-11 | Demo defaults: top-3 CLIP candidates, VLM may only pick among them; nutrition always from CSV; 15 classes without a USDA match show null (no invented values) | Grounding |
| 2026-10-11 | Demo-ready pass: `run.py` (stdlib, Windows-aware) replaces bash smoke; Makefile delegates to it; `scripts/smoke.py` uses urllib; `console.py` on `rich==13.9.4` with plain fallback; device order cuda>mps>cpu; `/analyze` has `response_model`; 3 sample JPEGs (384 px, <100 KB) copied from the cached Food-101 eval sample into `assets/` | Windows demo, narrated CLI |
| 2026-10-11 | Windows support verified by monkeypatched unit tests only (no Windows machine); `/usr/bin/time -l` not used by `run.py`; `eval_clip.py` peak RSS uses `resource` only when available | Honesty about verification |
| 2026-10-11 | Re-ran `demo` for this pass: vlm_demo.* now reflect a busy machine (median 13.3 s vs 6.2 s earlier); both are cited in README | Real numbers only |
