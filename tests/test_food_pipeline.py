import os
import sys
import cv2
import numpy as np
import io
from pathlib import Path
from werkzeug.datastructures import FileStorage

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "backend"))

from config import UPLOAD_FOLDER
from services.prediction_service import prediction_service

def make_file_storage(image_bgr: np.ndarray, filename: str) -> FileStorage:
    _, encoded = cv2.imencode(".jpg", image_bgr)
    bio = io.BytesIO(encoded.tobytes())
    return FileStorage(stream=bio, filename=filename, content_type="image/jpeg")

def run_tests():
    print("==================================================")
    print("Testing FoodLens-AI Food Pipeline & Rejection")
    print("==================================================")
    all_passed = True

    # Test 1: Person / Room (synthetic person silhouette & room background)
    room_person = np.ones((480, 640, 3), dtype=np.uint8) * 180  # Room wall
    cv2.rectangle(room_person, (0, 360), (640, 480), (120, 100, 80), -1)  # Floor
    cv2.circle(room_person, (320, 160), 60, (190, 170, 150), -1)  # Face
    cv2.rectangle(room_person, (240, 220), (400, 440), (80, 50, 40), -1)  # Torso/clothes
    res1 = prediction_service.process_image_upload(make_file_storage(room_person, "test_person_room.jpg"))
    passed1 = res1.get("food_detected") is False and res1.get("status") in ("no_food", "no_supported_food_detected")
    print(f"Test 1 - Person/Room: {'PASS' if passed1 else 'FAIL'} -> Status: {res1.get('status')}, Food Detected: {res1.get('food_detected')}, Food Name: {res1.get('food_name')}")
    if not passed1: all_passed = False

    # Test 2: Apple
    apple_path = ROOT_DIR / "frontend" / "static" / "image" / "apple.jpg"
    if apple_path.exists():
        apple_img = cv2.imread(str(apple_path))
        res2 = prediction_service.process_image_upload(make_file_storage(apple_img, "test_apple.jpg"))
        passed2 = res2.get("food_detected") is True and "apple" in res2.get("food_name", "").lower()
        print(f"Test 2 - Apple: {'PASS' if passed2 else 'FAIL'} -> Food: {res2.get('food_name')}, Conf: {res2.get('confidence')}%, Condition: {res2.get('condition')}")
        if not passed2: all_passed = False

    # Test 3: Banana
    banana_path = ROOT_DIR / "frontend" / "static" / "image" / "banana.jpg"
    if banana_path.exists():
        banana_img = cv2.imread(str(banana_path))
        res3 = prediction_service.process_image_upload(make_file_storage(banana_img, "test_banana.jpg"))
        passed3 = res3.get("food_detected") is True and "banana" in res3.get("food_name", "").lower()
        print(f"Test 3 - Banana: {'PASS' if passed3 else 'FAIL'} -> Food: {res3.get('food_name')}, Conf: {res3.get('confidence')}%, Condition: {res3.get('condition')}")
        if not passed3: all_passed = False

    # Test 4: Tomato
    tomato_path = ROOT_DIR / "frontend" / "static" / "image" / "tomato.jpg"
    if tomato_path.exists():
        tomato_img = cv2.imread(str(tomato_path))
        res4 = prediction_service.process_image_upload(make_file_storage(tomato_img, "test_tomato.jpg"))
        passed4 = res4.get("food_detected") is True and "tomato" in res4.get("food_name", "").lower()
        print(f"Test 4 - Tomato: {'PASS' if passed4 else 'FAIL'} -> Food: {res4.get('food_name')}, Conf: {res4.get('confidence')}%, Condition: {res4.get('condition')}")
        if not passed4: all_passed = False

    # Test 5: Pizza
    pizza_path = ROOT_DIR / "frontend" / "static" / "image" / "spoiled_pizza.jpg"
    if pizza_path.exists():
        pizza_img = cv2.imread(str(pizza_path))
        res5 = prediction_service.process_image_upload(make_file_storage(pizza_img, "test_pizza.jpg"))
        passed5 = res5.get("food_detected") is True or res5.get("status") in ("success", "low_confidence")
        print(f"Test 5 - Pizza: {'PASS' if passed5 else 'FAIL'} -> Food: {res5.get('food_name')}, Status: {res5.get('status')}")

    # Test 6: Random Object (Laptop / Chair / Blank table)
    laptop_img = np.ones((480, 640, 3), dtype=np.uint8) * 220
    cv2.rectangle(laptop_img, (150, 100), (490, 320), (30, 30, 30), -1)  # Laptop screen
    cv2.rectangle(laptop_img, (120, 320), (520, 370), (140, 140, 140), -1)  # Base keyboard
    res6 = prediction_service.process_image_upload(make_file_storage(laptop_img, "test_laptop.jpg"))
    passed6 = res6.get("food_detected") is False and res6.get("status") in ("no_food", "no_supported_food_detected")
    print(f"Test 6 - Random Object: {'PASS' if passed6 else 'FAIL'} -> Status: {res6.get('status')}, Food Detected: {res6.get('food_detected')}")
    if not passed6: all_passed = False

    # Test 7: Blurry Image
    blurry_img = cv2.GaussianBlur(apple_img if apple_path.exists() else np.ones((224, 224, 3), dtype=np.uint8)*150, (55, 55), 0)
    res7 = prediction_service.process_image_upload(make_file_storage(blurry_img, "test_blurry.jpg"))
    passed7 = res7.get("status") == "poor_image_quality" and "blur" in res7.get("error", {}).get("message", "").lower()
    print(f"Test 7 - Blurry Image: {'PASS' if passed7 else 'FAIL'} -> Status: {res7.get('status')}, Error: {res7.get('error', {}).get('message')}")
    if not passed7: all_passed = False

    # Test 8: Dark Image
    dark_img = np.ones((224, 224, 3), dtype=np.uint8) * 8
    res8 = prediction_service.process_image_upload(make_file_storage(dark_img, "test_dark.jpg"))
    passed8 = res8.get("status") == "poor_image_quality" and "dark" in res8.get("error", {}).get("message", "").lower()
    print(f"Test 8 - Dark Image: {'PASS' if passed8 else 'FAIL'} -> Status: {res8.get('status')}, Error: {res8.get('error', {}).get('message')}")
    if not passed8: all_passed = False

    print("==================================================")
    print(f"Overall Result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("==================================================")
    return all_passed

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
