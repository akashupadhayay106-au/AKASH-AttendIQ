"""
Unit tests for model pipeline artifacts and evaluation metrics in AKASH AttendIQ.
"""

import pytest
import os
import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_FILE, ROOT_MODEL_FILE, ALL_MODEL_FEATURES, ATTENDANCE_BANDS
from src.evaluation import calculate_regression_metrics, calculate_classification_metrics
from src.predictor import AttendancePredictor


def test_model_pipeline_artifact_exists():
    """Verify that serialized model package exists and loads correctly."""
    target_path = MODEL_FILE if os.path.exists(MODEL_FILE) else ROOT_MODEL_FILE
    assert os.path.exists(target_path), f"Pipeline artifact missing: {target_path}"

    package = joblib.load(target_path)
    assert "regression_model" in package
    assert "classification_model" in package
    assert "metadata" in package
    assert "band_labels" in package


def test_regression_metrics_calculation():
    """Verify regression metric calculations."""
    y_true = np.array([80.0, 75.0, 90.0, 60.0])
    y_pred = np.array([82.0, 70.0, 88.0, 65.0])
    metrics = calculate_regression_metrics(y_true, y_pred)

    assert "MAE (%)" in metrics
    assert "RMSE (%)" in metrics
    assert "R2 Score" in metrics
    assert metrics["MAE (%)"] > 0


def test_classification_metrics_calculation():
    """Verify classification metric calculations and probability handling."""
    y_true = np.array([0, 1, 2, 1, 2])
    y_pred = np.array([0, 1, 2, 2, 2])
    metrics = calculate_classification_metrics(y_true, y_pred, labels=ATTENDANCE_BANDS)

    assert "Accuracy (%)" in metrics
    assert "F1-Score" in metrics
    assert "Confusion_Matrix" in metrics
    assert metrics["Accuracy (%)"] > 0


def test_probability_calibration_sum():
    """Verify that output probabilities always sum perfectly to 1.0."""
    predictor = AttendancePredictor()
    dummy_input = {f: 50.0 for f in ALL_MODEL_FEATURES}
    res = predictor.predict_single(dummy_input)
    
    probs = res["probabilities"]
    # Check that SAFE + WARNING + CRITICAL sum to approximately 100.0%
    total_prob = sum(probs.values())
    assert abs(total_prob - 100.0) < 0.2, f"Probabilities do not sum to 100%, instead got {total_prob}%"

def test_leakage_protection_no_future_features():
    """Verify that no feature names contain 'Target', 'Future', or 'Actual' hinting at data leakage."""
    for feat in ALL_MODEL_FEATURES:
        feat_lower = feat.lower()
        assert "target" not in feat_lower, f"LEAKAGE DETECTED: {feat}"
        assert "future" not in feat_lower, f"LEAKAGE DETECTED: {feat}"
        assert "actual" not in feat_lower, f"LEAKAGE DETECTED: {feat}"
