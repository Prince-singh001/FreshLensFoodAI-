def test_chatbot_greeting(client):
    response = client.post("/api/v1/chat", json={"message": "hello"})
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert "FreshLens" in data["answer"]


def test_chatbot_storage_tip_rag(client):
    response = client.post("/api/v1/chat", json={
        "message": "how should I store apples?",
        "language": "en"
    })
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert "apple" in data["answer"].lower()
    assert "refrigerator" in data["answer"].lower() or "crisper" in data["answer"].lower()
    assert "tool_used" in data
    assert len(data.get("sources", [])) > 0


def test_chatbot_hindi_response(client):
    response = client.post("/api/v1/chat", json={
        "message": "सेब को कैसे स्टोर करना चाहिए?",
        "language": "hi"
    })
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["language"] == "hi"
    # Verify Devanagari Hindi output
    assert any('\u0900' <= char <= '\u097f' for char in data["answer"])


def test_chatbot_active_scan_context(client):
    response = client.post("/api/v1/chat", json={
        "message": "Is this safe to eat?",
        "language": "en",
        "scan_context": {
            "item": "Banana",
            "category": "Fruit",
            "freshness": "Spoiled",
            "confidence": 0.92
        }
    })
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert "banana" in data["answer"].lower()
    assert "spoiled" in data["answer"].lower() or "not recommended" in data["answer"].lower() or "discard" in data["answer"].lower()


def test_food_metadata_endpoint(client):
    response = client.get("/api/v1/food/apple")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    food = data["data"]
    assert food["name"] == "Apple"
    assert "nutrition" in food
    assert food["nutrition"]["calories_per_100g"] > 0


def test_food_metadata_not_found(client):
    response = client.get("/api/v1/food/space_alien_soup")
    assert response.status_code == 404
    data = response.get_json()
    assert data["success"] is False


def test_chatbot_empty_message(client):
    response = client.post("/api/v1/chat", json={"message": ""})
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
