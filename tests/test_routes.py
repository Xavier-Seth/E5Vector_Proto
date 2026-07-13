import json
import pytest
from app import app, WORD_COUNT_MIN

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_index_route(client):
    res = client.get("/")
    data = res.get_json()

    assert res.status_code == 200
    assert data["ok"] is True

def test_classify_empty_text(client):
    res = client.post("/classify", json={"text": ""})
    assert res.status_code == 400

def test_classify_short_text_guard(client):
    short_text = "This is too short"
    res = client.post("/classify", json={"text": short_text})
    data = res.get_json()

    assert res.status_code == 200
    assert data["too_short"] is True
    assert data["subcategory"] == "Uncategorized"
    assert data["word_count"] < WORD_COUNT_MIN

def test_rebuild_prototypes_route(client):
    res = client.post("/rebuild-prototypes")
    data = res.get_json()

    assert res.status_code == 200
    assert "classes" in data
