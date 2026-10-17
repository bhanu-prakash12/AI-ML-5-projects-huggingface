---
title: Food Lens
emoji: 🍽️
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 5.9.1
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: Food photo to dish, USDA nutrition and recipe ideas
---

# Food Lens

Upload a food photo. CLIP (zero-shot, Food-101 classes) names the dish, nutrition comes from USDA FoodData Central (US public domain), and SmolVLM-500M suggests ingredients and two recipes. Models run on a ZeroGPU GPU slice; the first request after a cold start is slower while models load. Nutrition values are looked up, never generated. Not dietary or medical advice.

The same app serves a REST API: `POST /analyze` (Swagger at `/docs`).
