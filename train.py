"""
AKASH AttendIQ — End-to-End Model Training, Benchmarking & Serialization CLI
AI-Powered College Attendance Prediction, Risk Intelligence & Analytics
Usage: python train.py
"""

import sys
import os
import time
import pandas as pd
import numpy as np

# Ensure UTF-8 output encoding if supported on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import (
    RAW_DATA_FILE,
    CLEANED_DATA_FILE,
    MODEL_FILE,
    PRODUCT_NAME,
    PRODUCT_TAGLINE
)
from src.data_processor import process_and_save_data
from src.model_trainer import train_complete_system
from src.predictor import AttendancePredictor


def main():
    print("=" * 75)
    print(f" 🚀 {PRODUCT_NAME} — MODEL PIPELINE TRAINING")
    print(f"    {PRODUCT_TAGLINE}")
    print("=" * 75)
    start_time = time.time()

    # Step 1: Data Ingestion & Leakage-Free Feature Engineering
    print("\n[1/3] Ingesting Raw Attendance Dataset & Engineering Temporal Features...")
    if not os.path.exists(RAW_DATA_FILE):
        print(f"[!] Error: Raw data file '{RAW_DATA_FILE}' not found!")
        sys.exit(1)

    df_featured = process_and_save_data(RAW_DATA_FILE, CLEANED_DATA_FILE)
    print(f"  ✓ Processed {len(df_featured):,} sessions across {df_featured.shape[1]} features.")
    print(f"  ✓ Validated zero-leakage lag & rolling calculations.")
    print(f"  ✓ Cleaned dataset saved to: {CLEANED_DATA_FILE}")

    # Step 2: Multi-Model Benchmarking & Evaluation
    print("\n[2/3] Training & Benchmarking Regression and Classification Models...")
    package, df_reg, df_cls = train_complete_system(df_featured, MODEL_FILE)

    print("\n" + "-" * 75)
    print(" 📈 REGRESSION MODEL LEADERBOARD (Target: Attendance_Pct)")
    print("-" * 75)
    print(df_reg.to_string(index=False))

    print("\n" + "-" * 75)
    print(" 🎯 CLASSIFICATION MODEL LEADERBOARD (Target: Attendance_Risk [CRITICAL, WARNING, SAFE])")
    print("-" * 75)
    print(df_cls.to_string(index=False))

    meta = package["metadata"]
    print(f"\n  ⭐ Champion Regression Model     : {meta['best_regression_model']}")
    print(f"  ⭐ Champion Classification Model : {meta['best_classification_model']}")
    print(f"  ⭐ Silhouette Score (K-Means)    : {meta['silhouette_score']:.4f}")
    print(f"  ✓ Deployment pipeline saved to  : {MODEL_FILE}")

    # Step 3: Inference & Explainability Self-Test
    print("\n[3/3] Running Inference & Explainability Self-Test...")
    predictor = AttendancePredictor(MODEL_FILE)
    test_sample = {
        "Lecture_Number": 1,
        "Total_Enrolled": 180,
        "Test_Week": 0,
        "Assignment_Due": 0,
        "Holiday_Near": 0,
        "Prev_Lecture_Pct": 85.0,
        "Rolling_3_Avg": 82.5,
        "Rolling_7_Avg": 81.0,
        "Gap_Hours": 24.0,
        "Day_of_Semester": 35,
        "Week_Number": 5,
        "Consecutive_Lecture_Count": 1,
        "Monthly_Avg_Attendance": 80.0,
        "Day": "Monday",
        "Subject": "Python Programming",
        "Faculty_ID": "FAC_01",
        "Branch": "MCA",
        "Session_Type": "Theory",
        "Time_of_Day": "Morning",
        "Classroom": "Room 201"
    }

    pred_res = predictor.predict_single(test_sample)
    print(f"  ✓ Sample Inference Output:")
    print(f"    - Predicted Attendance : {pred_res['predicted_attendance']:.2f}% (Range: {pred_res['estimated_range']})")
    print(f"    - Risk Classification  : {pred_res['risk_band']} (Confidence: {pred_res['confidence_pct']}%)")
    print(f"    - Probabilities        : {pred_res['probabilities']}")
    print(f"    - Expected Headcount   : {pred_res['expected_students']} present / {pred_res['total_enrolled']} enrolled")
    print(f"    - Top Positive Factors : {[f['factor'] for f in pred_res['top_positive_factors']]}")
    print(f"    - Top Negative Factors : {[f['factor'] for f in pred_res['top_negative_factors']]}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 75)
    print(f" [SUCCESS] PIPELINE TRAINING & VALIDATION COMPLETED IN {elapsed:.2f}s!")
    print("=" * 75)


if __name__ == "__main__":
    main()
