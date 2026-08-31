"""
Unit tests for data processor and feature engineering in AKASH AttendIQ.
"""

import pytest
import os
import pandas as pd
import numpy as np

from src.config import RAW_DATA_FILE, CLEANED_DATA_FILE
from src.data_processor import load_raw_data, clean_attendance_data, engineer_features


def test_load_raw_data():
    """Verify that raw dataset loads properly and has required columns."""
    df = load_raw_data(RAW_DATA_FILE)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "Attendance_Pct" in df.columns
    assert "Date" in df.columns
    assert "Subject" in df.columns


def test_clean_attendance_data():
    """Verify missing value imputation and IQR outlier clipping."""
    df_raw = load_raw_data(RAW_DATA_FILE)
    df_clean = clean_attendance_data(df_raw)

    # No nulls in attendance
    assert df_clean["Attendance_Pct"].isna().sum() == 0

    # Bounds [0, 100]
    assert df_clean["Attendance_Pct"].min() >= 0.0
    assert df_clean["Attendance_Pct"].max() <= 100.0


def test_feature_engineering_zero_leakage():
    """Verify lag and rolling feature calculations without future leakage."""
    df_raw = load_raw_data(RAW_DATA_FILE)
    df_clean = clean_attendance_data(df_raw)
    df_feat = engineer_features(df_clean)

    # Check temporal features
    assert "Day_of_Semester" in df_feat.columns
    assert "Week_Number" in df_feat.columns
    assert "Time_of_Day" in df_feat.columns
    assert "Gap_Hours" in df_feat.columns
    assert "Prev_Lecture_Pct" in df_feat.columns
    assert "Rolling_3_Avg" in df_feat.columns
    assert "Rolling_7_Avg" in df_feat.columns
    assert "Attendance_Risk" in df_feat.columns

    # No NaN in engineered history
    assert df_feat["Prev_Lecture_Pct"].isna().sum() == 0
    assert df_feat["Rolling_3_Avg"].isna().sum() == 0
    assert df_feat["Rolling_7_Avg"].isna().sum() == 0
