def test_get_history_empty_or_list(client):
    response = client.get("/api/v1/history")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert isinstance(data["data"], list)


def test_clear_and_delete_history(client):
    # Test clear
    response = client.delete("/api/v1/history")
    assert response.status_code == 200
    assert response.get_json()["success"] is True

    # Check empty
    get_res = client.get("/api/v1/history")
    assert get_res.get_json()["total"] == 0
