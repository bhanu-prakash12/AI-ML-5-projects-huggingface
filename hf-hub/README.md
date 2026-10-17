---
title: AI Engineering Projects
emoji: 🧪
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 5.9.1
python_version: "3.11"
app_file: app.py
pinned: true
license: mit
short_description: ML, RAG, LoRA, drift monitoring and multimodal AI demos
---

# AI Engineering Projects

One Space, five projects: **P1** student dropout risk (LightGBM + SHAP), **P2** RAG study assistant, **P3** LoRA medical Q&A
(base vs tuned), **P4** fraud-model drift monitoring (replay of a real run) and **P5** Food Lens (CLIP + USDA + SmolVLM).

Open the **Home** tab, pick a card, follow the "How to use" at the top of each tab. Educational demos, not medical or dietary advice.

## Dependency dedup
One `requirements.txt` (one version per package) and one Qwen2.5-0.5B-Instruct shared by P2 and P3. P5 also serves `POST /analyze`.
