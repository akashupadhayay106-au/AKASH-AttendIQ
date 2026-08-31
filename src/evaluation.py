"""
AKASH AttendIQ — Evaluation & Validation Engine
Calculates robust regression, classification, and time-aware evaluation metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit, cross_val_score
from src.config import ATTENDANCE_BANDS


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes comprehensive regression evaluation metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.clip(np.asarray(y_pred), 0.0, 100.0)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100.0)
    r2 = float(r2_score(y_true, y_pred))
    residuals = y_true - y_pred
    res_mean = float(np.mean(residuals))
    res_std = float(np.std(residuals))

    return {
        "MAE (%)": round(mae, 3),
        "RMSE (%)": round(rmse, 3),
        "MAPE (%)": round(mape, 3),
        "R2 Score": round(r2, 4),
        "Residual_Mean": round(res_mean, 3),
        "Residual_Std": round(res_std, 3)
    }


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    labels: List[str] = ATTENDANCE_BANDS
) -> Dict[str, Any]:
    """Computes multi-class classification metrics prioritizing WARNING and CRITICAL recall."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float(accuracy_score(y_true, y_pred) * 100.0)
    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Per-class recall (Critical to ensure at-risk students are detected)
    per_class_rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    rec_dict = {f"Recall_{labels[i]}": round(float(per_class_rec[i]), 4) for i in range(len(labels)) if i < len(per_class_rec)}

    roc_auc = 0.0
    if y_prob is not None:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"))
        except Exception:
            roc_auc = 0.0

    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(
        y_true, y_pred, target_names=labels, zero_division=0, output_dict=True
    )

    metrics = {
        "Accuracy (%)": round(acc, 2),
        "Precision": round(prec_weighted, 4),
        "Recall": round(rec_weighted, 4),
        "F1-Score": round(f1_weighted, 4),
        "F1-Macro": round(f1_macro, 4),
        "ROC-AUC": round(roc_auc, 4),
        "Confusion_Matrix": cm,
        "Classification_Report": report
    }
    metrics.update(rec_dict)
    return metrics


def perform_cross_validation(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv_strategy: str = "stratified",
    n_splits: int = 5,
    scoring: str = "neg_root_mean_squared_error"
) -> Dict[str, float]:
    """Executes cross-validation and returns performance summary."""
    if cv_strategy == "stratified":
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    elif cv_strategy == "temporal":
        cv = TimeSeriesSplit(n_splits=n_splits)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return {
        "cv_mean": round(float(np.mean(scores)), 4),
        "cv_std": round(float(np.std(scores)), 4),
        "scores": [round(float(s), 4) for s in scores]
    }
