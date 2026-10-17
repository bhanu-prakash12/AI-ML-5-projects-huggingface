"""Shared tab header: coloured banner, one-line pitch, links to the source README."""
import gradio as gr

SOURCE = "https://huggingface.co/spaces/{space}/tree/main/projects/{p}"

COLORS = {"p1": "#1565C0", "p2": "#2E7D32", "p3": "#6A1B9A", "p4": "#EF6C00", "p5": "#C62828"}


def header(p: str, title: str, pitch: str, folder: str, kind: str = "Live demo") -> None:
    gr.HTML(
        f'<div class="hub-banner" style="border-left:8px solid {COLORS[p]}">'
        f'<span class="hub-pill" style="background:{COLORS[p]}">{p.upper()}</span> '
        f'<b style="font-size:1.25em">{title}</b> <span class="hub-pill hub-kind">{kind}</span>'
        f'<div class="hub-pitch">{pitch}</div>'
        f'<div class="hub-src">Source folder in the repo: <code>{folder}/</code></div></div>')
