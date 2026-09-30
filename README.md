# FreshLens AI 🥗
> **AI-powered food recognition, freshness & safety analysis.**

FreshLens AI is an AI-powered food analysis platform that recognizes food items from images and analyzes freshness and safety-related information.

---

## 1. Overview
FreshLens AI is an enterprise-grade, API-first computer vision system engineered to analyze produce condition, detect multiple food objects in mixed baskets, and provide structured food safety guidance. Designed with strict scientific rigor, it replaces brittle heuristics and artificial confidence values with honest, probabilistic machine learning.

---

## 2. Features
- **Mobile-App-First UX:** Tailored for modern mobile viewports (320px–430px) with fixed bottom navigation bar, native camera frame overlay ("Place your food inside the frame"), quick action chips, and desktop expansion.
- **Agentic AI Assistant (LangChain + RAG):** Intelligent food safety copilot driven by LangChain agent architecture, modular tools, verified USDA FSIS / FDA documents, and active conversation memory.
- **Bilingual Reasoning (English & हिन्दी):** Complete multilingual assistant answering food safety, spoilage, and storage queries in English and native Hindi.
- **Scan Context Awareness:** Seamless handoff from produce scans to the AI Assistant ("Is this safe to eat?", "How should I store this?").
- **True Multi-Object Detection:** Localizes multiple discrete food items within a single image using YOLOv8 bounding boxes, rendering individual result cards.
- **Two-Stage Decoupled Analysis:** Distinctly separates object recognition (what item is it?) from freshness classification (is it fresh or spoiled?).
- **Honest Probabilistic Scoring:** Zero fake confidence caps or filename heuristics. Confidences strictly reflect true model outputs.
- **Out-of-Distribution (OOD) & Quality Checks:** Detects and gracefully rejects non-food items (cars, electronics, animals) as well as blurry or poorly-illuminated photos.
- **Verified Reference Nutrition Facts:** Enriched produce cards with USDA reference metrics (calories, carbs, protein, fiber, fat, vitamins).
- **Progressive Web App (PWA):** Standalone installation, ServiceWorker pre-caching, app icons, and offline fallbacks.

---

## 3. Agentic AI & RAG Architecture

```
User Query + Scan Context + Selected Language (EN / HI)
                        │
                        ▼
            [ FreshLens AI Orchestrator ]
                        │
       ┌────────────────┴────────────────┐
       ▼                                 ▼
[ Intent Routing ]               [ Agent Tools ]
       │                         ├── 1. Food Knowledge RAG (USDA / FDA / WHO)
       │                         ├── 2. Storage Recommendation Engine
       │                         ├── 3. Food Safety Protocol Tool
       │                         ├── 4. Nutrition Facts Lookup
       │                         ├── 5. Current Scan Context Inspector
       │                         ├── 6. Scan History Retriever
       │                         └── 7. External Nutrition API (Optional)
       │                                 │
       └────────────────┬────────────────┘
                        ▼
       [ LLM Synthesis (LangChain / Local Fallback) ]
                        │
                        ▼
       [ Multilingual Answer (EN / हिन्दी) + Verified Sources ]
```

---

## 3. How It Works
1. **File Validation & Quality Gate:** Scans image MIME type, file size (<= 16MB), Laplacian blur variance, and average illumination.
2. **Food Localization (Stage 1):** Ultralytics YOLOv8 detector localizes discrete produce and food items, extracting normalized bounding boxes and rejecting non-food clutter.
3. **Freshness Assessment (Stage 2):** Crops produce regions and evaluates physical surface characteristics using a MobileNetV2 convolutional neural network.
4. **Metadata & Advice Aggregation:** Category definitions, storage tips, and safety protocols are retrieved from centralized metadata.
5. **Client Presentation:** The backend generates color-coded annotated bounding box imagery and structured JSON results for cards and mobile apps.

---

## 4. Technology Stack
- **Deep Learning / Vision:** TensorFlow 2.x / Keras 3.x, TFLite Float32, Ultralytics YOLOv8, OpenCV, Pillow.
- **Backend API:** Python 3.11, Flask 3.x, Gunicorn, Scikit-learn, NumPy.
- **Frontend & PWA:** Semantic HTML5, Vanilla CSS Design System, Modern JavaScript (ES6+), WebRTC Camera Stream, Service Workers.
- **DevOps & Deployment:** Render.yaml (Blueprint), Git, Pytest.

---

## 5. Project Architecture

```
                                 [ USER IMAGE ]
                                        │
                                        ▼
                         [ 1. FILE & QUALITY GATEWAY ]
                 (MIME validation, size <= 16MB, magic bytes,
                  blur variance > 30, darkness > 20, brightness < 245)
                                        │
                                        ▼
                         [ 2. YOLOv8 OBJECT DETECTOR ]
                    (Locates discrete food objects & crops)
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
        [ Saliency / Produce Crops ]             [ Prepared / Non-Produce ]
                    │                                       │
                    ▼                                       ▼
     [ 3. MobileNetV2 Freshness ]              [ Category Metadata Lookup ]
       (Fresh vs. Spoiled Softmax)             (Item: Pizza/Rice/Burger)
                    │                          (Freshness: "Not Available")
                    └───────────────────┬───────────────────┘
                                        │
                                        ▼
                     [ 4. STRUCTURED JSON AGGREGATOR ]
              (Bounding boxes, annotated image, category tags,
               safety tips, storage advice, inference timing)
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
     [ Web UI / PWA Frontend ]                 [ Mobile Clients (iOS/Android) ]
```

---

## 6. ML Model
- **Freshness Classifier:** MobileNetV2 pre-trained on ImageNet with customized GlobalAveragePooling, Dense(256), and Dense(18, softmax) heads.
- **Detector:** YOLOv8 Nano (`yolov8n.pt`) fine-tuned for food categories and salient bounding proposals.
- **Model Storage:**
  - Production Keras: `models/production/freshlens_mobilenetv2_model.h5`
  - Production TFLite: `models/production/freshlens_mobilenetv2_model.tflite`
  - YOLO Weights: `models/production/yolov8n.pt`
  - Backwards-compatible aliases exist for legacy deployments.

---

## 7. Dataset
- **Raw Directory:** `dataset/raw/` for incoming public dataset archives.
- **Processed Directory:** `dataset/processed/` with strict, non-overlapping splits:
  - `Train/`: 70% of verified samples.
  - `Validation/`: 15% for EarlyStopping & checkpoint monitoring.
  - `Test/`: 15% dedicated exclusively for holdout evaluation.
- **Integrity Tools:** `tools/prepare_dataset.py` ensures deduplication via MD5 hashing and eliminates corrupted files.

---

## 8. Backend
- **Framework:** Flask 3.x with modular blueprints.
- **Singleton ML Loader:** Thread-safe model caching in memory with background warmup.
- **Production Logging:** Structured millisecond latency tracking via `backend/utils/logger.py`.
- **Security:** Strict file validation, path traversal sanitization, and environment-driven secrets.

---

## 9. Frontend
- **Design System:** Custom CSS tokens, accessible contrast, smooth transitions, and dark/light mode toggle stored in `localStorage`.
- **Pages:**
  - `/`: 12-section landing showcase with live scanner preview and FAQs.
  - `/scan`: Interactive scanner supporting drag-and-drop, camera streaming, and dynamic multi-cards.
  - `/history`: Scan history inspection with clear and single-delete controls.
  - `/chatbot`: AI Food Safety Assistant with storage tips and non-medical disclaimers.
  - `/features`, `/about`, `/contact`: Comprehensive documentation and branding.

---

## 10. API
Versioned endpoints under `/api/v1/`:
- `GET  /api/v1/health` - Uptime and model load verification.
- `POST /api/v1/predict` - Multipart image upload; returns detected objects, crops, freshness, reference nutrition, and bounding boxes.
- `POST /api/v1/feedback` - Queues misclassifications into the offline review dataset.
- `GET  /api/v1/history` - Returns paginated scan history.
- `DELETE /api/v1/history` - Deletes specific scan or wipes history.
- `POST /api/v1/chat` - Agentic multilingual food assistant (`conversation_id`, `language`, `scan_context`).
- `GET  /api/v1/food/<food_name>` - Returns verified produce metadata, storage tips, and USDA reference nutrition facts.
- `GET  /api/v1/classes` - Returns supported class and category metadata.

See [docs/API.md](docs/API.md) for schemas and cURL examples.

---

## 11. Installation

```bash
# 1. Clone repository
git clone https://github.com/Prince-singh001/SafeBite-AI.git
cd SafeBite-AI

# 2. Create virtual environment
python -m venv backend/venv

# 3. Activate virtual environment
# Windows:
.\backend\venv\Scripts\activate
# Linux/macOS:
source backend/venv/bin/activate

# 4. Install dependencies
pip install -r backend/requirements.txt
```

---

## 12. Local Development

```bash
# Run root entrypoint
python main.py

# Access application at: http://localhost:5000
```

---

## 13. Model Training
```bash
# 1. Clean, deduplicate, and split raw data:
python tools/prepare_dataset.py --source backend/dataset/Train --output dataset/processed

# 2. Train Freshness Model (monitors val_loss):
python training/train_freshness.py --epochs 15

# 3. Export to optimized TFLite:
python training/export_model.py
```

---

## 14. Model Evaluation
```bash
python training/evaluate.py --model models/production/freshlens_mobilenetv2_model.h5 --test-dir backend/dataset/Test
```

### Verified Test Results (6,780 Holdout Test Images):
- **Accuracy:** 93.29%
- **Weighted F1 Score:** 0.9323
- **Precision:** 0.9445
- **Recall:** 0.9329
- *Reports saved to `reports/metrics.json` and `reports/confusion_matrix.png`.*

---

## 15. Deployment
Configured for automated zero-downtime deployment on Render via [render.yaml](render.yaml):
- **Worker Configuration:** 1 worker, 4 threads (optimized for 512MB RAM constraints).
- **Runtime Optimization:** `USE_TFLITE=1` for low-memory CPU inference.
- **Port:** Dynamic binding via `$PORT`.

---

## 16. Future Improvements
- **Expanded Indian & Packaged Food Recognition:** Ingest additional datasets for regional cuisines (dal, roti, dosa, paneer dishes).
- **Mobile SDK Wrapper:** React Native / Flutter client consuming the `/api/v1/` endpoints.
- **Nutritional & Calorie Estimation:** Integration with USDA or Edamam nutrition APIs.
- **Edge Deployment:** On-device camera inference using TFLite and CoreML.

---

## 17. Limitations
- **Visual Surface vs. Internal Spoilage:** Visual computer vision assesses exterior surface degradation (mold, soft rot, browning). It cannot detect internal anaerobic pathogens, bacterial toxins, or chemical spoilage.
- **Prepared Foods Freshness:** Cooked foods without distinct visual decomposition are categorized with `freshness: "Not Available"`.
- **Dataset Class Coverage:** Certain produce classes (e.g. onion, carrot) currently lack sufficient training samples in the starter dataset and require additional data collection before enabling freshness classification.

---

## 📄 License & Safety Disclaimer
Licensed under the MIT License.

**Food Safety Disclaimer:** AI visual assessments are informational estimates based on physical surface features. They cannot detect microscopic bacterial toxins or internal pathogens. Always verify food texture, smell, and temperature before consumption.
