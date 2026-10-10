# FoodLens-AI - API Documentation (v2.0)

FoodLens-AI provides a production-ready, versioned REST API (`/api/v1/`) designed for web applications, mobile apps (React Native, Flutter, native Android/iOS), and PWA clients.

---

## Base URLs
- **Local Development:** `http://localhost:5000/api/v1`
- **Production (Render):** `https://foodlens-ai.onrender.com/api/v1`

---

## Authentication & Headers
- Currently public endpoints with standard multipart and JSON payloads.
- All JSON responses follow the standard API response structure:
```json
{
  "success": true,
  "status": "success",
  ...
}
```
Error responses:
```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable explanation"
  }
}
```

---

## 1. System Health Check
`GET /api/v1/health` (also accessible at `GET /health`)

### Response (200 OK)
```json
{
  "success": true,
  "status": "healthy",
  "service": "FoodLens-AI API",
  "version": "2.0.0",
  "environment": "production",
  "timestamp": "2026-09-30T01:30:00.000000"
}
```

---

## 2. Multi-Object Food Prediction
`POST /api/v1/predict` (also accessible at legacy `POST /predict`)

Performs image validation, quality inspection, object detection, bounding box extraction, and MobileNetV2 freshness analysis.

### Request
- **Content-Type:** `multipart/form-data`
- **Parameters:**
  - `file` *(binary image)*: JPG, JPEG, PNG, or WEBP (Max 16MB).
  - `selected_category` *(optional string)*: "All", "Fruit", "Vegetable", or "Food".

### Example cURL
```bash
curl -X POST "http://localhost:5000/api/v1/predict" \
  -F "file=@produce_basket.jpg" \
  -F "selected_category=All"
```

### Success Response (200 OK)
```json
{
  "success": true,
  "status": "success",
  "image_url": "/static/uploads/20260930013000_produce_basket.jpg",
  "annotated_image_url": "/static/uploads/annotated_20260930013000_produce_basket.jpg",
  "inference_time_ms": 112.45,
  "objects": [
    {
      "id": 1,
      "item": "Apple",
      "item_key": "apple",
      "category": "Fruit",
      "detection_confidence": 0.9412,
      "freshness": "Fresh",
      "freshness_status": "Fresh",
      "freshness_confidence": 0.9150,
      "bbox": {
        "x1": 84,
        "y1": 105,
        "x2": 290,
        "y2": 320
      },
      "stability_warning": "",
      "storage_tip": "Store apples in the refrigerator crisper drawer.",
      "safety_guideline": "Avoid consuming apples with deep internal core rot."
    },
    {
      "id": 2,
      "item": "Banana",
      "item_key": "banana",
      "category": "Fruit",
      "detection_confidence": 0.9230,
      "freshness": "Fresh",
      "freshness_status": "Fresh",
      "freshness_confidence": 0.8840,
      "bbox": {
        "x1": 310,
        "y1": 120,
        "x2": 520,
        "y2": 360
      },
      "stability_warning": "",
      "storage_tip": "Store bananas at room temperature.",
      "safety_guideline": "Discard bananas showing liquid leakage or mold."
    }
  ],
  "summary": {
    "total_objects": 2,
    "fruits": 2,
    "vegetables": 0,
    "food": 0
  }
}
```

### Error Responses
- **400 Bad Request:** Missing file or invalid MIME type.
- **422 Unprocessable Entity:** Poor image quality (blurry, extreme darkness) or non-food rejection:
```json
{
  "success": false,
  "status": "poor_image_quality",
  "error": {
    "code": "POOR_IMAGE_QUALITY",
    "message": "Image is too blurry. Please upload a sharper, steady image of your food."
  },
  "quality_metrics": {
    "width": 1024,
    "height": 768,
    "brightness": 120.5,
    "blur_score": 14.2
  }
}
```

---

## 3. Submit Prediction Feedback
`POST /api/v1/feedback`

Directs user feedback and corrections into a quality-review queue for batch retraining without directly mutating production model weights.

### Request Body (JSON)
```json
{
  "image_url": "/static/uploads/sample.jpg",
  "predicted_class": "freshapples",
  "correct_class": "spoileapples",
  "notes": "Small brown soft rot on lower quadrant."
}
```

### Response (200 OK)
```json
{
  "success": true,
  "feedback_id": "a9f82c4e10b1",
  "status": "queued_for_review",
  "message": "Thank you! Your feedback has been securely submitted to our quality queue for review and future batch retraining."
}
```

---

## 4. Retrieve Scan History
`GET /api/v1/history?limit=20&offset=0`

### Response (200 OK)
```json
{
  "success": true,
  "total": 45,
  "limit": 20,
  "offset": 0,
  "data": [
    {
      "id": "e4c19a82",
      "timestamp": "2026-09-30 01:25:12",
      "food_name": "Apple",
      "condition": "Fresh",
      "confidence": 94.2,
      "category": "Fruit",
      "image_url": "/static/uploads/20260930012512_apple.jpg",
      "annotated_image_url": "/static/uploads/annotated_20260930012512_apple.jpg",
      "total_objects": 1
    }
  ]
}
```

---

## 5. Delete Scan History
`DELETE /api/v1/history?id=e4c19a82`
- Pass `?id=...` to delete a specific scan record.
- Omit `id` to clear all scan history.

---

## 6. AI Food Safety Chatbot
`POST /api/v1/chat`

### Request Body (JSON)
```json
{
  "message": "How should I store fresh tomatoes?"
}
```

### Response (200 OK)
```json
{
  "success": true,
  "response": "**Tomato Storage & Safety Profile:**\n\n- **Storage Tip:** Store whole tomatoes stem-side down at room temperature away from direct sunlight...\n- **Safety Guideline:** Inspect for surface breaks, skin collapse, or foul fermented odor.\n\n*Important Safety Note: AI visual assessments are informational...*"
}
```

---

## 7. Supported Classes Metadata
`GET /api/v1/classes`

Returns centralized dictionary of supported items, categories, freshness support flags, storage tips, and safety guidelines.
