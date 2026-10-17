# P5 Food Lens — Plan
Photo -> CLIP zero-shot top-3 (Food-101 classes) -> USDA nutrition rows (CSV, never VLM) -> SmolVLM-500M JSON (dish, why, ingredients, portion estimate, 2 recipes) -> FastAPI /analyze -> Gradio UI.
Profile `small` (8 GB): SmolVLM-500M (~1 GB fp16) + CLIP (~0.6 GB) both stay resident (~2-3 GB). The "free CLIP before VLM" rule only applies with Qwen2-VL (standard profile); `ClipClassifier.close()` implements it.
Ports API 8005, UI 7865, MLflow N/A (5005 reserved).
Modules: config, device, clip_zs (prompt ensemble, cache), nutrition (csv load), vlm (load/generate, JSON+pydantic+retry), pipeline, api, scripts/{download_models,eval_clip,build_nutrition,demo,ui,smoke.sh}.
Phases: 0 scaffold+models | 1 CLIP zero-shot + eval | 2 nutrition DB | 3 VLM reasoning | 4 API | 5 UI | 6 verify + README.
Acceptance: clip_eval.json (top1/top5 on 300 imgs); nutrition.csv 101 rows, nulls not invented; VLM demo JSON valid for sample images; /analyze returns timings; UI launches; make verify green.
Risks: small VLM JSON reliability (retry + typed error); USDA DEMO_KEY rate limit (sleep/backoff, resume); SmolVLM on MPS fp16 may be unstable (fallback fp32/cpu).
