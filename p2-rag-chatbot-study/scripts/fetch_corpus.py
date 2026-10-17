"""Build data/corpus/: Wikipedia notes (markdown + provenance) and, if approved, exam PDFs."""
import logging
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import requests  # noqa: E402

from study_assistant.config import load_config, resolve  # noqa: E402

log = logging.getLogger("fetch_corpus")
API = "https://en.wikipedia.org/w/api.php"


def slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


def extract_to_markdown(text: str) -> str:
    """Turn MediaWiki plain-text '== Heading ==' markers into markdown headings."""

    def repl(m: re.Match) -> str:
        return "\n" + "#" * len(m.group(1)) + " " + m.group(2).strip() + "\n"

    return re.sub(r"^(={2,6})\s*(.+?)\s*\1\s*$", repl, text, flags=re.M)


def fetch_article(session: requests.Session, title: str) -> dict:
    params = {
        "action": "query", "format": "json", "formatversion": 2, "redirects": 1,
        "prop": "extracts|revisions|info", "explaintext": 1, "exsectionformat": "wiki",
        "rvprop": "ids|timestamp", "inprop": "url", "titles": title,
    }
    r = session.get(API, params=params, timeout=30)
    r.raise_for_status()
    page = r.json()["query"]["pages"][0]
    if page.get("missing"):
        raise ValueError(f"missing page: {title}")
    rev = page["revisions"][0]
    return {
        "title": page["title"], "url": page["fullurl"], "revid": rev["revid"],
        "timestamp": rev["timestamp"], "text": extract_to_markdown(page["extract"]),
    }


def write_note(out: Path, art: dict) -> None:
    front = (
        f"---\ntitle: {art['title']}\nurl: {art['url']}\nrevision: {art['revid']}\n"
        f"revision_timestamp: {art['timestamp']}\nlicense: CC BY-SA 4.0\ndoc_type: notes\n---\n"
    )
    out.write_text(front + f"# {art['title']}\n\n" + art["text"].strip() + "\n")


def main() -> None:
    import argparse

    from study_assistant import console

    argparse.ArgumentParser(description="Build data/corpus/: 20 Wikipedia notes (markdown + provenance) and 2 MIT OCW exam PDFs.\nInputs: titles/URLs in configs/config.yaml. Outputs: data/corpus/*. Sample: python scripts/fetch_corpus.py",
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    console.banner("Fetch corpus", "Download notes and exams into data/corpus.", "20 notes + 2 exam PDFs (needs network)")
    console.setup_logging()
    cfg = load_config()
    notes_dir = resolve(cfg, "corpus") / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = cfg["corpus"]["user_agent"]
    for title in cfg["corpus"]["wikipedia_titles"]:
        out = notes_dir / f"{slug(title)}.md"
        if out.exists():
            log.info("skip (exists) %s", title)
            continue
        art = fetch_article(s, title)
        write_note(out, art)
        log.info("fetched %s rev %s (%d chars)", art["title"], art["revid"], len(art["text"]))
        time.sleep(1.0)  # polite
    if not cfg["corpus"]["exam_urls"]:
        log.warning("no exam_urls approved; skipping exams (see DECISIONS.md)")
        return
    exam_dir = resolve(cfg, "corpus") / "exams"
    exam_dir.mkdir(parents=True, exist_ok=True)
    for item in cfg["corpus"]["exam_urls"]:
        out = exam_dir / item["file"]
        if out.exists():
            continue
        r = s.get(item["url"], timeout=60)
        r.raise_for_status()
        out.write_bytes(r.content)
        log.info("downloaded %s", item["url"])
        time.sleep(1.0)


if __name__ == "__main__":
    main()
