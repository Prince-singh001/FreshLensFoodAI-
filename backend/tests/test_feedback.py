def test_feedback_submission(client):
    payload = {
        "image_url": "/static/uploads/sample.jpg",
        "predicted_class": "freshapples",
        "correct_class": "spoileapples",
        "notes": "Color is brown on the side."
    }
    response = client.post("/api/v1/feedback", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert "feedback_id" in data
    assert data["status"] == "queued_for_review"


def test_feedback_missing_fields(client):
    response = client.post("/api/v1/feedback", json={"image_url": ""})
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert data["error"]["code"] == "MISSING_IMAGE_URL"
