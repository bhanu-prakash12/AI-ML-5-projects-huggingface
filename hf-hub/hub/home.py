"""Landing page: hero, how-to-use steps, five project cards with a button that opens the project's tab."""
from __future__ import annotations

import gradio as gr

CARDS = [
    ("p1", "#1565C0", "Student dropout risk", "Tabular ML", "LightGBM, SHAP, FastAPI",
     "Type a student's numbers, get a risk score and the 3 features behind it.", "Test ROC-AUC 0.928, PR-AUC 0.903", "CPU, ms"),
    ("p2", "#2E7D32", "Study assistant", "RAG chatbot", "bge-small, Chroma, Qwen 0.5B",
     "Ask about machine learning; answers cite notes and exams, off-topic questions are refused.",
     "recall@4 0.792, refusal accuracy 7/7", "GPU slice"),
    ("p3", "#6A1B9A", "Medical Q&amp;A: base vs LoRA", "Fine-tuning", "Qwen 0.5B + LoRA (peft)",
     "Same question, two answers: adapter off vs on. An honest evaluation: style changed, accuracy did not.",
     "ROUGE-L 0.157 vs 0.154 (no gain)", "GPU slice"),
    ("p4", "#EF6C00", "Fraud model monitoring", "MLOps replay", "Evidently, PSI, LightGBM",
     "Replay a 10-day run: drift appears on day 6, the alert fires, the retrain gate decides.",
     "baseline PR-AUC 0.801; alert on window 6", "Replay"),
    ("p5", "#C62828", "Food Lens", "Multimodal", "CLIP, USDA, SmolVLM-500M",
     "Upload a food photo: dish, USDA nutrition and recipe ideas.", "CLIP top-1 0.83, top-5 0.97", "GPU slice"),
]

HERO = """<div class="hub-hero"><h1>AI Engineering Projects</h1>
<p>Five small, production-shaped AI/ML projects in one place: tabular ML, RAG, LoRA fine-tuning, model monitoring and
multimodal AI. Every number on these pages comes from a real run saved in each project's <code>reports/</code>.</p>
<p><b>New here?</b> Pick a card below, click <b>Open</b>, and follow the short "How to use" at the top of the tab.</p></div>"""

STEPS = """### Start in 3 steps
<ol class="hub-steps">
<li><b>Choose a project</b> with the cards below, or the tabs at the top of the page.</li>
<li><b>Click an example</b> (every tab has ready samples) and press the main button. First answer after a cold start can take a minute while models load.</li>
<li><b>Read the "Pros / cons / at scale"</b> accordion at the bottom of each tab: it explains what you just saw and what changes in production.</li>
</ol>
<small>Educational demos. p3 is not medical advice and p5 is not dietary advice. Models are small on purpose; answers can be wrong.</small>"""

FOOTER = """---
**What is shared:** one Python environment, one Qwen2.5-0.5B-Instruct in memory for p2 and p3, one Gradio theme.
**Licenses:** code MIT; models Apache-2.0 (Qwen, SmolVLM) / MIT (CLIP weights laion2B); USDA FoodData Central is US public domain;
Wikipedia text CC BY-SA; MIT OCW exams CC BY-NC-SA 4.0; MedQuAD CC BY 4.0."""


def render(tabs: gr.Tabs) -> None:
    gr.HTML(HERO)
    gr.Markdown(STEPS)
    for row in (CARDS[:3], CARDS[3:]):
        with gr.Row(equal_height=True):
            for p, color, title, kind, stack, blurb, result, cost in row:
                with gr.Column():
                    gr.HTML(
                        f'<div class="hub-card" style="border-top:6px solid {color}">'
                        f'<span class="hub-pill" style="background:{color}">{p.upper()}</span> '
                        f'<span class="hub-pill hub-kind">{kind}</span>'
                        f'<h3>{title}</h3><div>{blurb}</div>'
                        f'<p><small><b>Result:</b> {result}<br><b>Stack:</b> {stack}<br><b>Runs on:</b> {cost}</small></p></div>')
                    gr.Button(f"Open {p.upper()}", variant="primary").click(
                        lambda p=p: gr.Tabs(selected=p), None, tabs)
    gr.Markdown(FOOTER)
