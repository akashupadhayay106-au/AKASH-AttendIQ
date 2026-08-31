"""
Unit tests for predictor engine, explainability, and What-If simulator in AKASH AttendIQ.
"""

import pytest
import os
import pandas as pd
from src.config import MODEL_FILE
from src.predictor import AttendancePredictor


@pytest.fixture
def predictor():
    return AttendancePredictor(MODEL_FILE)


@pytest.fixture
def valid_sample():
    return {
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


def test_predict_single_output_structure(predictor, valid_sample):
    """Verify single prediction output schema and range."""
    res = predictor.predict_single(valid_sample)

    assert "predicted_attendance" in res
    assert 0.0 <= res["predicted_attendance"] <= 100.0
    assert res["risk_band"] in ["CRITICAL", "WARNING", "SAFE"]
    assert "probabilities" in res
    assert "CRITICAL" in res["probabilities"]
    assert "estimated_range" in res
    assert "expected_students" in res
    assert res["expected_students"] <= res["total_enrolled"]
    assert "top_positive_factors" in res
    assert "top_negative_factors" in res
    assert "advisory" in res


def test_simulate_what_if(predictor, valid_sample):
    """Verify What-If scenario simulation."""
    override = {"Test_Week": 1, "Holiday_Near": 1}
    sim = predictor.simulate_what_if(valid_sample, override)

    assert "baseline" in sim
    assert "scenario" in sim
    assert "diff_attendance_pct" in sim
    assert "changed_factors" in sim
    assert len(sim["changed_factors"]) == 2
