# P5 verification (short)
- `make verify`: lint clean, 12 tests pass, smoke ok (real uvicorn :8005, 3 images analyzed, 5 MB limit returns 413, Gradio UI HTTP 200 on :7865, `ui.analyze()` end-to-end).
- Spec acceptance: clip_eval.json top-1 0.83 / top-5 0.97 (300 imgs); nutrition.csv has 101 rows, 86 matched, 15 null (not invented); VLM demo 6/6 valid JSON; /analyze returns timings; UI launches.
- Data privacy rule honoured: p5-multimodal-ai-food/data/ never opened; build script prints counts only; review list in reports/mapping_review.md.
## Open findings
1. VLM quality is limited (500M): sentences are generic, `portion_estimate.approx_grams` is a weak guess. The JSON is assembled in code from per-field answers (deviation from "single JSON-only reply", logged in DECISIONS.md) because a single-reply schema gave 0/6 valid.
2. No ground truth for the VLM dish choice (6 demo images); only CLIP accuracy is measured (300 images). The VLM may only pick among CLIP top-3.
3. 21 of 101 classes flagged for review (15 no USDA match, 6 partial/weak match, e.g. prime_rib maps to a NZ raw beef ribs row; french_fries match is generic potato). Mapping is generic USDA food, not the exact dish.
4. USDA bulk files: SR Legacy 2018-04 + FNDDS 2024-10-31; FNDDS rows use nutrient numbers (mapped in code).
5. Single-process, one inference at a time (semaphore 1); no auth, no rate limiting.
6. Qwen2-VL-2B (`standard` profile) not downloaded or tested.
