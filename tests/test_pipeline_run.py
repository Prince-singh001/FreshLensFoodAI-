import os
import sys
import json

root_dir = os.path.abspath(".")
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
backend_dir = os.path.abspath("backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from werkzeug.datastructures import FileStorage
from backend.services.prediction_service import prediction_service

test_cases = [
    ("Test 1 (1 Banana + 3 Apples)", "frontend/static/uploads/20261004185927_Screenshot_2026-09-30_005141.png"),
    ("Test 2 (Single Apple)", "frontend/static/image/apple.jpg"),
    ("Test 3 (Single Banana)", "frontend/static/image/banana.jpg"),
    ("Test 4 (Single Tomato)", "frontend/static/image/tomato.jpg"),
    ("Test 5 (Non-Food Laptop)", "frontend/static/uploads/20260930235625_test_laptop.jpg"),
    ("Test 6 (Blurred Image)", "frontend/static/uploads/20260930235626_test_blurry.jpg"),
]

for label, path in test_cases:
    print(f"\n==========================================")
    print(f"RUNNING: {label}")
    print(f"File: {path}")
    if not os.path.exists(path):
        print(f"Skipping: file not found at {path}")
        continue

    with open(path, "rb") as f:
        ext = os.path.splitext(path)[1].lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        fs = FileStorage(stream=f, filename=os.path.basename(path), content_type=mime)
        res = prediction_service.process_image_upload(fs, selected_category="All")

    print(f"Success: {res.get('success')} | Status: {res.get('status')}")
    if not res.get("success"):
        err = res.get("error", {})
        print(f"Message: {res.get('message') or err.get('message')}")
    else:
        summary = res.get("summary", {})
        print(f"Summary: total={summary.get('total_objects')} | fruits={summary.get('fruits')} | veg={summary.get('vegetables')} | fresh={summary.get('fresh')} | spoiled={summary.get('spoiled')}")
        for o in res.get("objects", []):
            print(f"  -> [{o['item']}] Category={o['category']} | Freshness={o['freshness_status']} ({o['confidence']*100:.1f}%) | BBox={o['bbox']}")
