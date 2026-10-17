import json

from fastapi.testclient import TestClient

from study_assistant import api


def fake_events(store, req, cfg):
    if "football" in req.question:
        yield {"sources": []}
        yield {"token": "Not in your notes."}
        yield {"done": {"refused": True}}
    else:
        yield {"sources": [{"label": "a.md:1", "score": 0.8, "snippet": "x"}]}
        yield {"token": "Hello "}
        yield {"token": "[a.md:1]"}
        yield {"done": {"refused": False, "citations": ["a.md:1"]}}


def client(monkeypatch):
    monkeypatch.setattr(api, "load_resources", lambda cfg: {"store": object()})
    monkeypatch.setattr(api, "stream_events", fake_events)
    return TestClient(api.app)


def parse(text):
    out, ev = [], None
    for line in text.splitlines():
        if line.startswith("event: "):
            ev = line[7:]
        elif line.startswith("data: "):
            out.append((ev, json.loads(line[6:])))
    return out


def test_health_and_ask(monkeypatch):
    with client(monkeypatch) as c:
        assert c.get("/health").json()["status"] == "ok"
        ev = parse(c.post("/ask", json={"question": "What is SVM?"}).text)
        assert [e for e, _ in ev] == ["sources", "token", "token", "done"]
        assert ev[-1][1]["refused"] is False
        ev = parse(c.post("/ask", json={"question": "football?"}).text)
        assert ev[-1][1]["refused"] is True


def test_ask_validation(monkeypatch):
    with client(monkeypatch) as c:
        assert c.post("/ask", json={"question": ""}).status_code == 422
        assert c.post("/ask", json={"question": "x", "doc_type": "bogus"}).status_code == 422


def test_ingest_lock_and_result(monkeypatch):
    monkeypatch.setattr(api, "run_ingest", lambda cfg, store: {"chunks": 3, "added": 0})
    with client(monkeypatch) as c:
        assert c.post("/ingest").json() == {"chunks": 3, "added": 0}
        api.INGEST_LOCK.acquire()
        try:
            assert c.post("/ingest").status_code == 409
        finally:
            api.INGEST_LOCK.release()


def test_ask_simple_answered_and_refused(monkeypatch):
    with client(monkeypatch) as c:
        r = c.post("/ask/simple", json={"question": "What is SVM?"}).json()
        assert r["refused"] is False and r["answer"] == "Hello [a.md:1]"
        assert r["citations"] == ["a.md:1"] and r["sources"][0]["label"] == "a.md:1"
        r = c.post("/ask/simple", json={"question": "football?"}).json()
        assert r["refused"] is True


def test_openapi_examples_valid():
    from study_assistant.api import AskRequest, AskResponse

    spec = api.app.openapi()
    assert "/ask/simple" in spec["paths"] and spec["info"]["description"]
    for ex in AskRequest.model_config["json_schema_extra"]["examples"]:
        AskRequest(**ex)
    AskResponse(**AskResponse.model_config["json_schema_extra"]["examples"][0])
