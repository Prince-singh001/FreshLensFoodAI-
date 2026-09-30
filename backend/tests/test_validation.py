import io

def test_missing_file_upload(client):
    response = client.post("/api/v1/predict", data={})
    assert response.status_code in (400, 422)
    data = response.get_json()
    assert data["success"] is False
    assert data["error"]["code"] == "MISSING_FILE"


def test_empty_file_upload(client):
    data = {
        "file": (io.BytesIO(b""), "empty.jpg")
    }
    response = client.post("/api/v1/predict", data=data, content_type="multipart/form-data")
    assert response.status_code in (400, 422)
    res = response.get_json()
    assert res["success"] is False


def test_invalid_extension_upload(client):
    data = {
        "file": (io.BytesIO(b"malicious script or text"), "document.pdf")
    }
    response = client.post("/api/v1/predict", data=data, content_type="multipart/form-data")
    assert response.status_code in (400, 422)
    res = response.get_json()
    assert res["success"] is False
    assert "Unsupported file extension" in res["error"]["message"]


def test_corrupted_image_upload(client):
    # A fake JPEG header with junk bytes
    corrupted_bytes = b"\xFF\xD8\xFF\xE0" + b"\x00" * 50
    data = {
        "file": (io.BytesIO(corrupted_bytes), "corrupt.jpg")
    }
    response = client.post("/api/v1/predict", data=data, content_type="multipart/form-data")
    assert response.status_code in (400, 422)
    res = response.get_json()
    assert res["success"] is False
    assert res["error"]["code"] == "INVALID_IMAGE"
