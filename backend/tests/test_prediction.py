import io
import os
import cv2
import numpy as np
from PIL import Image

def test_prediction_synthetic_image(client):
    # Generate clean 224x224 RGB test image with red apple-like shape
    arr = np.ones((224, 224, 3), dtype=np.uint8) * 240
    cv2.circle(arr, (112, 112), 60, (40, 40, 220), -1)  # Red circle
    cv2.circle(arr, (110, 50), 10, (30, 180, 50), -1)   # Green leaf

    buf = io.BytesIO()
    img = Image.fromarray(arr)
    img.save(buf, format="JPEG")
    buf.seek(0)

    data = {
        "file": (buf, "test_produce.jpg"),
        "selected_category": "Fruit"
    }

    response = client.post("/api/v1/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    res = response.get_json()

    assert res["success"] is True
    assert "objects" in res
    assert isinstance(res["objects"], list)
    assert len(res["objects"]) >= 1

    first_obj = res["objects"][0]
    assert "item" in first_obj
    assert "category" in first_obj
    assert "detection_confidence" in first_obj
    assert "bbox" in first_obj
    assert "x1" in first_obj["bbox"]
    assert "summary" in res
    assert "total_objects" in res["summary"]


def test_prediction_existing_sample_image(client):
    # Check if a real image exists in dataset/Train/freshapples
    sample_dir = os.path.join(os.path.dirname(__file__), "..", "dataset", "Train", "freshapples")
    if os.path.exists(sample_dir):
        files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.png'))]
        if files:
            sample_path = os.path.join(sample_dir, files[0])
            with open(sample_path, "rb") as f:
                content = f.read()

            data = {
                "file": (io.BytesIO(content), "real_apple.png"),
                "selected_category": "Fruit"
            }
            response = client.post("/api/v1/predict", data=data, content_type="multipart/form-data")
            assert response.status_code == 200
            res = response.get_json()
            assert res["success"] is True
            assert res["food_name"] in ("Apple", "Produce Item")
            assert res["condition"] in ("Fresh", "Spoiled", "Not Available")
