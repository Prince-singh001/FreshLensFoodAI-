# FreshLens AI - System Architecture

## 1. Architectural Philosophy
FreshLens AI is engineered around an **API-first, modular, two-stage computer vision architecture** designed for high dependability, low latency, and honest probabilistic evaluations.

Unlike simplistic prototype models that conflate object identity with freshness condition and hardcode artificial prediction shortcuts, FreshLens AI strictly separates:
1. **Object Detection & Localization:** *What food items are present in the image and where are they located?*
2. **Freshness Assessment:** *For recognized produce items, is the cropped physical surface fresh or spoiled?*
3. **Out-of-Distribution (OOD) & Image Quality Rejection:** *Is the input image actually food, and is the resolution, illumination, and focus sufficient for dependable inference?*

```
                             [ USER IMAGE ]
                                   │
                                   ▼
                       ┌───────────────────────┐
                       │   IMAGE VALIDATION    │
                       │ (MIME, Size, Corrupt) │
                       └───────────┬───────────┘
                                   │ Pass
                                   ▼
                       ┌───────────────────────┐
                       │ QUALITY INSPECTION    │
                       │ (Blur, Darkness, Lux) │
                       └───────────┬───────────┘
                                   │ Pass
                                   ▼
                       ┌───────────────────────┐
                       │  FOOD / OBJECT LOCATOR│
                       │(YOLO / Salient Bounds)│
                       └───────────┬───────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              │ Non-food / Unknown                      │ Food Objects Found
              ▼                                         ▼
   ┌───────────────────────┐                ┌───────────────────────┐
   │ REJECTION RESPONSE    │                │  CROP OBJECT REGIONS  │
   │ (Human-readable tips) │                └───────────┬───────────┘
   └───────────────────────┘                            │
                                                        ▼
                                            ┌───────────────────────┐
                                            │ CATEGORY & METADATA   │
                                            │  classes.json Mapping │
                                            └───────────┬───────────┘
                                                        │
                         ┌──────────────────────────────┴──────────────────────────────┐
                         │ Produce (Freshness supported)                               │ Prepared Food
                         ▼                                                             ▼
             ┌───────────────────────┐                                     ┌───────────────────────┐
             │ MOBILENETV2 CLASSIFIER│                                     │ CATEGORY RECOGNITION  │
             │   (Fresh vs Spoiled)  │                                     │ (Freshness: N/A)      │
             └───────────┬───────────┘                                     └───────────┬───────────┘
                         │                                                             │
                         └──────────────────────────────┬──────────────────────────────┘
                                                        │
                                                        ▼
                                            ┌───────────────────────┐
                                            │ ANNOTATION GENERATOR  │
                                            │ (Bounding Boxes/Pills)│
                                            └───────────┬───────────┘
                                                        │
                                                        ▼
                                            ┌───────────────────────┐
                                            │ STRUCTURED JSON API   │
                                            │ & RESPONSIVE FRONTEND │
                                            └───────────────────────┘
```

---

## 2. Component Breakdown

### A. Image Quality & OOD Inspection (`backend/ml/quality.py`)
- **Dimension Check:** Validates input is between 64x64px and 4096x4096px. Overly large images are scaled with aspect-ratio preservation.
- **Laplacian Variance Focus Metric:** Detects out-of-focus or motion-blurred captures (`var < 30.0`).
- **Mean Pixel Luminance:** Identifies severe underexposure (`< 20.0`) and bleached overexposure (`> 245.0`).
- **Non-Food Rejection:** Detects COCO non-food classes (cars, chairs, animals, persons) and flags them with explicit guidance.

### B. Multi-Object Detection (`backend/ml/detector.py`)
- Executes multi-object bounding box extraction using Ultralytics YOLO with fallback to morphological salient object segmentation.
- Extracts discrete bounding boxes `[x1, y1, x2, y2]` for each distinct item.
- Eliminates the previous anti-pattern where top-3 softmax probabilities of a single classifier were misrepresented as multiple objects.

### C. Freshness Evaluation (`backend/ml/freshness.py`)
- MobileNetV2 architecture with GlobalAveragePooling2D, Dense(256), Dropout, Dense(128), and Softmax.
- Operates on cropped object crops, comparing fresh and spoiled class pairs for each produce type.
- Returns honest probabilistic softmax confidences. No artificial caps (such as 97.5% or 98.0%).
- Emits borderline freshness alerts when prediction margin between conditions is narrow.

### D. Centralized Class Metadata (`backend/data/classes.json`)
- Single source of truth for categories (Fruit, Vegetable, Food).
- Eliminates fragile string pattern matching in `app.py`.
- Encapsulates display names, detector aliases, storage tips, and safety protocols.

### E. Feedback & Review Dataset Queue (`backend/services/feedback_service.py`)
- Replaces unsafe on-the-fly model mutation.
- Stores user corrections and original images in `backend/data/feedback_uploads/` with metadata in `feedback.json`.
- Feeds offline, validated batch retraining workflows.

### F. Dual Runtime Deployment (Render & Local)
- **Local Development:** Full TensorFlow 2.x / Keras 3.x and Ultralytics YOLO with multithreaded preprocessing.
- **Render Cloud:** Optimized TFLite runtime with single Gunicorn worker (`workers = 1`, `threads = 2`, `timeout = 180`) to guarantee low memory usage (< 512MB RAM).
