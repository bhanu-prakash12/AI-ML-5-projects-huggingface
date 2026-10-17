from fastapi.testclient import TestClient

from med_assistant import api
from med_assistant.safety import is_emergency


def test_emergency_keywords():
    assert is_emergency("I think I'm having a Heart Attack")
    assert is_emergency("thinking about suicide")
    assert not is_emergency("What are the symptoms of type 2 diabetes?")


def test_api_validation_and_refusal():
    api._S.update(cfg={"disclaimer": "d", "system_prompt": "s"}, model=object(), adapter=True)
    c = TestClient(api.app)  # no lifespan context: state injected above
    assert c.post("/generate", json={"question": "hi there", "mode": "x"}).status_code == 422
    assert c.post("/generate", json={"question": "x"}).status_code == 422
    r = c.post("/generate", json={"question": "severe chest pain now", "mode": "tuned"}).json()
    assert r["refused"] and "emergency" in r["answer"].lower()
    api._S.clear()
