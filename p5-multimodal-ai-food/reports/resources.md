# Resources (real runs, MacBook Apple Silicon, 8 GB profile `small`)
| step | wall | peak RSS | source |
|---|---|---|---|
| CLIP zero-shot eval, 300 images | 22 ms/img (MPS) | ~2.0 GB | reports/clip_eval.json |
| `make demo` (CLIP + SmolVLM, 6 images) | load 5.3 s; median total 6.2 s/img | 2.47 GB (`/usr/bin/time -l`, earlier prompt version) | reports/vlm_demo.md |
| `make smoke` API requests | total 6.6-7.5 s/img (clip 29-203 ms, vlm 6.5-7.3 s) | n/a | smoke output |
| `python run.py demo` (6 images, run while other jobs shared the machine) | load 8.3 s; median total 13.3 s/img | n/a | reports/vlm_demo.md |
