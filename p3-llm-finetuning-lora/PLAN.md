# P3 Plan: LoRA fine-tuning for medical Q&A
Goal: fine-tune Qwen2.5-0.5B-Instruct on MedQuAD (300/50/50 small profile), measure vs base honestly, serve base+tuned from one model.
Layout: p3-llm-finetuning-lora/{configs,src/med_assistant,scripts,tests,data,artifacts,reports} per CLAUDE.md; ports 8003/7863/5003.

| Phase | Deliverable | Acceptance |
|---|---|---|
| 0 | scaffold, pins dry-run | uv dry-run resolves |
| 1 Data | make data: clean, dedupe, <=384 tok (chat template), deterministic splits, reports/data.md | splits 300/50/50, no overlap, lengths <=384, rerun identical |
| 2 Baseline | base greedy on test, ROUGE-L, avg len, 10 prompts -> reports/baseline.json | json written, 50 test rows |
| 3 Train | SFTTrainer+LoRA r8/a16, 1 epoch, bs1 ga8, fp32, grad ckpt, resume, MLflow, ETA first | loss logged and falling, artifacts/adapter |
| 4 Evaluate | base vs tuned, forgetting on 10 general prompts -> reports/compare.md | honest table |
| 5 Serve | one model, disable_adapter for base; /generate mode; Gradio side-by-side; safety; merged copy | smoke ok both modes + emergency refusal |
| 6 Verify + README | make verify, reports/verification.md, README | green |

Risks: MPS fp32 speed (ETA gate 60 min); 0.5B capacity; ROUGE != correctness; dataset bias; tiny n makes deltas noisy.
Eval: ROUGE-L (evaluate/rouge-score) + length + qualitative; forgetting = same 10 general prompts, compare outputs and length, flag changes.
