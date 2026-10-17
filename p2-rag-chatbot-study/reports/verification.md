# p2 verification (2026-10-11)

## Fresh environment (throwaway copy in scratchpad, real p2 untouched)
- make setup (uv cache) 1.9 s; make lint clean; make test 20 passed (8.5 s).
- Models/corpus/Chroma not rebuilt in the copy (user rule); full pipeline verified on real p2.

## Real p2
- make verify green: ruff clean, 20 tests, smoke (real uvicorn, 1 in-corpus answered with sources, 1 out-of-corpus refused), 21.6 s.
- Flow: API healthy in ~10 s; /ask SSE 200 in 6.1 s with sources and refused=false; Gradio on :7860 returns 200 and exposes its chat endpoints. make stop left ports 8000/7860 free.
- Resources: see resources.md (API RSS ~807 MB after one ask, sampled).

## Architect review (PLAN vs code vs config vs reports)
Fixed: duplicate DECISIONS rows, stale PLAN notes (pin approval, eval review wording), .env.example offline hint, resources.md added.

## Open findings
1. Model-emitted citation rate 0.083; answers rely on appended "Sources:" fallback. Disclose in README; optional few-shot or larger model (needs approval).
2. Eval is indicative: 31 auto-drafted questions, gate tuned on same set so 100% refusal accuracy is in-sample.
3. Exam chunk fallback rate 0.571; most exam content unstructured. Disclose.
4. Possible junk short chunks (a 3-char minimum seen in ingest.md): add min-length filter in ingest.py (not done, needs re-index).
5. Confirm unit tests cover the gate/refusal path without an LLM call.
6. README pending (phase 6 /readme).
