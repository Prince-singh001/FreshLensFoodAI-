"""
FoodLens-AI - Evaluation Script (backend/src/evaluate.py)
Evaluates the end-to-end Food Detection and Rejection Pipeline.
Specifically measures and reports:
  - Accuracy, Precision, Recall, F1-score
  - Confusion Matrix
  - Food -> Food (Correct Food)
  - Food -> Non-Food (Missed Food)
  - Non-Food -> Food (CRITICAL FALSE POSITIVE RATE)
  - Non-Food -> Non-Food (Correct Rejection)
"""
import os
import sys
import json
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
sys.path.insert(0, str(BASE_DIR.parent))

from src.predict import predict_image
from config import ROOT_DIR, REPORTS_DIR

def evaluate_pipeline():
    print("=" * 60)
    print("FoodLens-AI - Model Evaluation & Pipeline Audit")
    print("=" * 60)

    test_data_dir = ROOT_DIR / "backend" / "data" / "food_vs_nonfood" / "test"
    if not test_data_dir.exists():
        print(f"Test directory not found at: {test_data_dir}")
        print("Please run python training/prepare_food_vs_nonfood_dataset.py first.")
        return

    food_dir = test_data_dir / "food"
    non_food_dir = test_data_dir / "non_food"

    food_files = sorted(list(food_dir.glob("*.jpg")) + list(food_dir.glob("*.png")))
    non_food_files = sorted(list(non_food_dir.glob("*.jpg")) + list(non_food_dir.glob("*.png")))

    print(f"Found {len(food_files)} food test samples and {len(non_food_files)} non-food test samples.")
    print("Running inference across holdout test set...")

    # Evaluation counters
    tp = 0  # Food -> Food
    fn = 0  # Food -> Non-Food
    fp = 0  # Non-Food -> Food (CRITICAL)
    tn = 0  # Non-Food -> Non-Food

    # Evaluate Food samples
    print(f"\nEvaluating {len(food_files)} Food test samples...", flush=True)
    for idx, fpath in enumerate(food_files):
        res = predict_image(str(fpath))
        if res.get("food_detected") is True:
            tp += 1
        else:
            fn += 1
        if (idx + 1) % 50 == 0 or (idx + 1) == len(food_files):
            print(f"  Processed {idx + 1}/{len(food_files)} food samples...", flush=True)

    # Evaluate Non-Food samples
    print(f"\nEvaluating {len(non_food_files)} Non-Food test samples...", flush=True)
    for idx, fpath in enumerate(non_food_files):
        res = predict_image(str(fpath))
        if res.get("food_detected") is True:
            fp += 1
        else:
            tn += 1
        if (idx + 1) % 50 == 0 or (idx + 1) == len(non_food_files):
            print(f"  Processed {idx + 1}/{len(non_food_files)} non-food samples...", flush=True)

    total_samples = tp + fn + fp + tn
    accuracy = (tp + tn) / total_samples if total_samples > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    non_food_fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print("\n" + "=" * 60)
    print("PIPELINE EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total Test Samples:             {total_samples}")
    print(f"Overall Accuracy:               {accuracy * 100:.2f}%")
    print(f"Food Precision:                 {precision * 100:.2f}%")
    print(f"Food Recall:                    {recall * 100:.2f}%")
    print(f"F1-Score:                       {f1:.4f}")
    print("-" * 60)
    print("CONFUSION MATRIX:")
    print(f"  Actual Food     -> Predicted Food (TP):     {tp}  (Correct Food)")
    print(f"  Actual Food     -> Predicted Non-Food (FN): {fn}  (Missed Food)")
    print(f"  Actual Non-Food -> Predicted Food (FP):     {fp}  <-- CRITICAL FALSE POSITIVE RATE: {non_food_fp_rate*100:.2f}%")
    print(f"  Actual Non-Food -> Predicted Non-Food (TN): {tn}  (Correctly Rejected)")
    print("=" * 60)

    # Save to reports
    os.makedirs(str(REPORTS_DIR), exist_ok=True)
    report_file = REPORTS_DIR / "evaluation_summary.json"
    metrics_data = {
        "model": "FoodLens-AI Production Pipeline",
        "total_samples": total_samples,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": {
            "food_to_food": tp,
            "food_to_nonfood": fn,
            "nonfood_to_food": fp,
            "nonfood_to_nonfood": tn
        },
        "critical_metrics": {
            "non_food_to_food_false_positives": fp,
            "non_food_false_positive_rate": round(non_food_fp_rate, 4),
            "non_food_rejection_accuracy": round(tn / (fp + tn) if (fp + tn) > 0 else 1.0, 4)
        }
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    print(f"\nDetailed evaluation metrics saved to: {report_file}")
    return metrics_data

if __name__ == "__main__":
    evaluate_pipeline()