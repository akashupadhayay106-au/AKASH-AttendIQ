"""
Unit tests for batch prediction and edge case handling in AKASH AttendIQ.
"""

import pytest
import pandas as pd
import numpy as np
from src.config import MODEL_FILE
from src.predictor import AttendancePredictor


@pytest.fixture
def predictor():
    return AttendancePredictor(MODEL_FILE)


def test_predict_batch_valid(predictor):
    """Verify batch CSV prediction outputs correct columns and row counts."""
    sample_df = pd.DataFrame([
        {
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
        },
        {
            "Lecture_Number": 6,
            "Total_Enrolled": 120,
            "Test_Week": 1,
            "Assignment_Due": 1,
            "Holiday_Near": 1,
            "Prev_Lecture_Pct": 55.0,
            "Rolling_3_Avg": 60.0,
            "Rolling_7_Avg": 65.0,
            "Gap_Hours": 48.0,
            "Day_of_Semester": 80,
            "Week_Number": 12,
            "Consecutive_Lecture_Count": 6,
            "Monthly_Avg_Attendance": 68.0,
            "Day": "Friday",
            "Subject": "Cloud Computing",
            "Faculty_ID": "FAC_03",
            "Branch": "MBA",
            "Session_Type": "Theory",
            "Time_of_Day": "Afternoon",
            "Classroom": "Room 202"
        }
    ])

    df_res = predictor.predict_batch(sample_df)
    assert len(df_res) == 2
    assert "Predicted_Attendance_Pct" in df_res.columns
    assert "Predicted_Risk_Band" in df_res.columns
    assert "Confidence_Pct" in df_res.columns
    assert "Estimated_Range" in df_res.columns
    assert "Expected_Students_Present" in df_res.columns


def test_predict_batch_missing_optional_columns(predictor):
    """Verify batch inference handles missing non-core features gracefully with imputation."""
    partial_df = pd.DataFrame([
        {
            "Lecture_Number": 3,
            "Total_Enrolled": 150,
            "Subject": "Advanced DBMS",
            "Branch": "MCA"
        }
    ])

    df_res = predictor.predict_batch(partial_df)
    assert len(df_res) == 1
    assert "Predicted_Attendance_Pct" in df_res.columns
    assert not np.isnan(df_res["Predicted_Attendance_Pct"].iloc[0])
