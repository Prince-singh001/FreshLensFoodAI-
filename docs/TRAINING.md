# FreshLens AI - Training & Evaluation Pipeline

## 1. Overview
FreshLens AI implements standard machine learning workflows with independent validation and test sets, strictly preventing data leakage and eliminating fake performance claims.

---

## 2. Freshness Classifier Architecture
- **Backbone:** MobileNetV2 pre-trained on ImageNet
- **Top Layers:**
  - `GlobalAveragePooling2D()`
  - `Dense(256, activation='relu')`
  - `Dropout(0.3)`
  - `Dense(128, activation='relu')`
  - `Dropout(0.2)`
  - `Dense(NUM_CLASSES, activation='softmax')`
- **Loss Function:** `categorical_crossentropy`
- **Optimizer:** `Adam(learning_rate=1e-4)`

---

## 3. Training Script (`training/train_freshness.py`)
Addresses all training integrity requirements:
- Monitors `val_loss` rather than training loss.
- `EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)`
- `ModelCheckpoint('models/candidates/...', monitor='val_loss', save_best_only=True, mode='min')`
- `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)`
- Saves training loss and accuracy curves to `graphs/training_curves.png`.

### Command:
```bash
python training/train_freshness.py --data backend/dataset/Train --epochs 15
```

---

## 4. Model Evaluation (`training/evaluate.py`)
Evaluates candidate or production models exclusively on the holdout test set (`backend/dataset/Test`).
- Computes weighted Precision, Recall, F1 score, and Confusion Matrix.
- Outputs honest reports:
  - `reports/metrics.json`
  - `reports/classification_report.txt`
  - `reports/confusion_matrix.png`

### Command:
```bash
python training/evaluate.py --model models/production/freshlens_mobilenetv2_model.h5 --test-dir backend/dataset/Test
```

---

## 5. TFLite Export & Verification (`training/export_model.py`)
Converts trained Keras `.h5` models to optimized `.tflite` format and verifies output prediction consistency (< 0.05 absolute divergence):
```bash
python training/export_model.py
```

---

## 6. Detector Fine-Tuning (`training/train_detector.py`)
Trains Ultralytics YOLOv8 on custom produce datasets using `dataset/produce_data.yaml`:
```bash
python training/train_detector.py --data dataset/produce_data.yaml --epochs 30
```
