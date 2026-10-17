# P3 verification (2026-10-11)
`make verify` green: ruff clean, 10 unit tests pass (~2 s), smoke (real uvicorn on 8003: tuned and base answers differ, emergency question refused, invalid mode 422). `make stop` run; nothing left on 8003/7863.

## Resources (this 8 GB MacBook, MPS fp32, `small` profile)
See resources.md. Baseline 73 s / 3.28 GB peak; training 37 steps 5.2 min / 3.43 GB; compare 128 s / 3.44 GB; adapter 19 MB, merged fp32 copy 1.9 GB (gitignored).

## Results in one line
ROUGE-L base 0.157 vs tuned 0.154 (no gain, CI spans 0); tuned answers shorter (87 vs 140 words); mild drift on general prompts. See compare.md.

## Open findings
1. No quantitative gain at 300 examples / 37 steps; n=50 test is too small to detect small effects (CI [-0.014, +0.011]).
2. ROUGE-L measures overlap with one reference, not medical correctness. No clinician review; qualitative notes are anecdotal.
3. Checkpoint resume code path was not exercised in this run.
4. Dataset license not declared on the HF card (upstream MedQuAD CC BY 4.0 per its repo); dataset is never redistributed here. Sampling keeps only <=384-token rows, biasing toward shorter answers.
5. Emergency refusal is keyword-based (misses paraphrases, can over-trigger); not a clinical safety system.
6. compare.md "Interpretation" section is hand-written after reading compare.json; rerunning `make eval` regenerates numbers but drops that section.
7. Inline review only (no separate architect/verify agent, no fresh-copy rebuild) per run instructions.
