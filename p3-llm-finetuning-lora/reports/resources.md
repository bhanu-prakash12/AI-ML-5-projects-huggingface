# Resources
| step | wall | peak RAM |
|---|---|---|
| baseline (50 test + 10 prompts, MPS fp32) | 72.7 s generation | 3.28 GB |
| 5 | LoRA train (37 steps, MPS fp32) | 5.2 min | 3.43 GB |
| 7 | compare (base+tuned, 50 test + 20 prompts each) | 128 s | 3.44 GB |
