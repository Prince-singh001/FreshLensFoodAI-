# FreshLens AI - Dataset Architecture & Management

## 1. Overview
FreshLens AI utilizes a structured dataset ingestion and preparation pipeline supporting fresh and spoiled produce classifications as well as multi-class food recognition.

---

## 2. Directory Hierarchy
```
dataset/
├── raw/                 # Downloaded raw zip files / archives from verified sources
├── processed/           # Normalized, validated, deduplicated images
│   ├── Train/           # 70% Training split
│   ├── Validation/      # 15% Validation split
│   └── Test/            # 15% Holdout evaluation split (Never used in training)
└── feedback/            # User-submitted corrections queued for review
```

---

## 3. Supported Classes & Categories
The platform maps recognized produce and food items using `backend/data/classes.json`:

| Canonical Key | Display Name | Category | Freshness Model Supported |
|---|---|---|---|
| `apple` | Apple | Fruit | Yes |
| `banana` | Banana | Fruit | Yes |
| `orange` | Orange | Fruit | Yes |
| `tomato` | Tomato | Vegetable | Yes |
| `potato` | Potato | Vegetable | Yes |
| `cucumber` | Cucumber | Vegetable | Yes |
| `bitter_gourd` | Bitter Gourd | Vegetable | Yes |
| `onion` | Onion | Vegetable | Yes (Extended) |
| `carrot` | Carrot | Vegetable | Yes (Extended) |
| `broccoli` | Broccoli | Vegetable | Planned |
| `sandwich` | Sandwich | Food | Guidance Only |
| `pizza` | Pizza | Food | Guidance Only |
| `hot_dog` | Hot Dog | Food | Guidance Only |
| `donut` | Donut | Food | Guidance Only |
| `cake` | Cake | Food | Guidance Only |
| `rice` | Rice | Food | Guidance Only |
| `bread` | Bread | Food | Guidance Only |
| `mango` | Mango | Fruit | Planned |
| `grapes` | Grapes | Fruit | Planned |
| `strawberry` | Strawberry | Fruit | Planned |

---

## 4. Legitimate Public Dataset Sources
- **Fruits Fresh and Rotten Dataset:** Available on Kaggle under Creative Commons CC BY 4.0 license.
- **Produce-101 / Food-101:** Open academic benchmark for food image recognition.
- **License compliance:** No unauthorized web scraping. All training data requires documented attribution and permissive licensing.

---

## 5. Dataset Tools

### Automated Download
```bash
python tools/download_dataset.py --dest dataset/raw
```
Supports Kaggle API credentials via `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables.

### Normalization and Split
```bash
python tools/prepare_dataset.py
```
- Corrects folder naming inconsistencies (`bittergroud` -> `bitter_gourd`, `spoile` -> `spoiled`).
- Computes MD5 checksums to eliminate duplicate images across classes.
- Validates file integrity via PIL, pruning corrupt or 0-byte files.
- Generates 70% Train, 15% Validation, and 15% Test splits.

### Integrity Validation Audit
```bash
python tools/validate_dataset.py --dir backend/dataset
```
Scans all dataset directories and outputs file counts, corrupted file reports, and class balance statistics.
