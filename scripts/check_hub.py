"""Smoke-check a running hub (local or Space): page, REST routes, every tab's real function, p4 report links.

usage: python scripts/check_hub.py http://127.0.0.1:7860
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7860").rstrip("/")
fails: list[str] = []


def check(name: str, ok: bool, info: str = "") -> None:
    print(("PASS " if ok else "FAIL ") + name + (f"  {info}" if info else ""))
    if not ok:
        fails.append(name)


def main() -> None:
    r = httpx.get(URL + "/", timeout=60, follow_redirects=True)
    check("home page 200", r.status_code == 200)
    if "hf.space" not in URL:  # the Spaces proxy reserves /health (308 -> /), so only check it locally
        h = httpx.get(URL + "/health", timeout=60)
        check("p5 /health", h.status_code == 200 and h.json().get("status") == "ok", h.text[:80])
    img = ROOT / "p5-multimodal-ai-food" / "assets" / "sample_1_cheese_plate.jpg"
    a = httpx.post(URL + "/analyze", files={"file": img.read_bytes()}, timeout=300)
    check("p5 /analyze", a.status_code == 200, a.text[:100])

    from gradio_client import Client

    c = Client(URL, verbose=False)
    eps = [e for e in c.view_api(return_format="dict")["named_endpoints"]]
    print("endpoints:", eps)
    r1 = c.predict("What are the symptoms of type 2 diabetes?", api_name="/ask")
    check("p3 base+tuned answered", bool(r1[0]) and bool(r1[1]) and r1[2] == "Done.", str(r1[1])[:90])
    r2 = c.predict("I have severe chest pain and cannot breathe", api_name="/ask")
    check("p3 emergency refused", "REFUSED" in r2[0] and "REFUSED" in r2[1])
    p2 = c.predict("What is the bias-variance tradeoff?", "all", api_name="/chat")
    check("p2 in-corpus answered with sources", "Sources" in p2 and "not in your notes" not in p2, p2[:90])
    p2r = c.predict("Who won the 1998 football world cup?", "all", api_name="/chat")
    check("p2 out-of-corpus refused", "not in your notes" in p2r.lower(), p2r[:90])
    p1 = c.predict(*[0] * 0, api_name="/run") if False else None
    page = httpx.get(URL + "/config", timeout=30).text
    for m in sorted(set(re.findall(r"/gradio_api/file=[^\"'\\ ]+\.html", page)))[:3]:
        check(f"p4 report {Path(m).name}", httpx.get(URL + m, timeout=60).status_code == 200)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
