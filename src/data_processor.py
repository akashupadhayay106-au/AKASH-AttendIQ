"""
AKASH AttendIQ — Data Processor & Feature Engineering Engine
Handles data ingestion, validation, cleaning, IQR outlier bounds, and strict zero-leakage feature engineering.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from src.config import (
    RAW_DATA_FILE,
    CLEANED_DATA_FILE,
    ATTENDANCE_BANDS,
    BAND_BINS,
    TARGET_REGRESSION,
    TARGET_CLASSIFICATION,
    TARGET_CLASSIFICATION_CODE
)


def load_raw_data(filepath: str = RAW_DATA_FILE) -> pd.DataFrame:
    """Load raw attendance dataset from CSV and validate basic schema."""
    if not os.path.exists(filepath):
        # Fallback to current directory if not found in data/
        base_fallback = os.path.join(os.path.dirname(os.path.dirname(__file__)), "imcc_raw_attendance.csv")
        if os.path.exists(base_fallback):
            filepath = base_fallback
        else:
            raise FileNotFoundError(f"Raw data file not found at: {filepath}")

    df = pd.read_csv(filepath)
    required_cols = [
        "Date", "Day", "Lecture_Number", "Start_Time", "Subject",
        "Faculty_ID", "Semester", "Branch", "Section", "Classroom",
        "Total_Enrolled", "Attendance_Pct"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required core columns: {missing}")

    return df


def clean_attendance_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw attendance data:
    - Parses dates safely
    - Converts numerical types
    - Imputes missing attendance via subject & session-type medians
    - Caps IQR outliers within valid physical limits [0, 100]%
    """
    df = df_raw.copy()

    # 1. Parse Date safely
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="coerce")
    if df["Date"].isna().any():
        df["Date"] = df["Date"].fillna(pd.to_datetime(df_raw["Date"], errors="coerce"))

    # 2. Parse Attendance_Pct
    df["Attendance_Pct"] = pd.to_numeric(df["Attendance_Pct"], errors="coerce")

    # 3. Missing Attendance Imputation
    df["Attendance_Pct"] = (
        df.groupby(["Subject", "Session_Type"])["Attendance_Pct"]
        .transform(lambda x: x.fillna(x.median()))
    )
    df["Attendance_Pct"] = df["Attendance_Pct"].fillna(df["Attendance_Pct"].median())

    # 4. Outlier & Physical Boundary Treatment [0%, 100%]
    q1 = df["Attendance_Pct"].quantile(0.25)
    q3 = df["Attendance_Pct"].quantile(0.75)
    iqr = q3 - q1
    lower_limit = max(0.0, float(q1 - 1.5 * iqr))
    upper_limit = min(100.0, float(q3 + 1.5 * iqr))

    df["Attendance_Pct"] = df["Attendance_Pct"].clip(lower_limit, upper_limit)

    # 5. Synchronize Students_Present (only for tracking, NOT as a predictor feature)
    if "Total_Enrolled" in df.columns:
        df["Students_Present"] = (
            df["Total_Enrolled"] * df["Attendance_Pct"] / 100.0
        ).round().astype(int)

    return df


def engineer_features(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Generates temporal, historical lag, rolling average, and gap features.
    STRICT ZERO-LEAKAGE: All lag and rolling calculations strictly use shifted (t-1) historical records.
    """
    df = df_clean.sort_values(
        ["Branch", "Section", "Date", "Lecture_Number"]
    ).reset_index(drop=True)

    # Temporal features
    semester_start = df["Date"].min()
    df["Day_of_Semester"] = (df["Date"] - semester_start).dt.days + 1
    df["Week_Number"] = ((df["Day_of_Semester"] - 1) // 7) + 1
    df["Month"] = df["Date"].dt.month
    df["Day_of_Month"] = df["Date"].dt.day
    df["Is_Weekend"] = df["Day"].isin(["Saturday", "Sunday"]).astype(int)

    # Session Timing (Morning vs Afternoon)
    df["Time_of_Day"] = np.where(
        df["Start_Time"].str.contains("AM", case=False, na=False),
        "Morning",
        "Afternoon"
    )

    # Lecture Datetime for precise inter-lecture gap tracking
    df["Lecture_DateTime"] = pd.to_datetime(
        df["Date"].dt.strftime("%Y-%m-%d") + " " + df["Start_Time"],
        errors="coerce"
    )

    # Grouping by Branch and Section for isolated cohort histories
    group = df.groupby(["Branch", "Section"])

    # Strict (t-1) historical shifted lags (NO current session leakage)
    df["Prev_Lecture_Pct"] = group["Attendance_Pct"].shift(1)
    df["Rolling_3_Avg"] = group["Attendance_Pct"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df["Rolling_7_Avg"] = group["Attendance_Pct"].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).mean()
    )

    # Monthly expanding average (shifted)
    df["Monthly_Avg_Attendance"] = df.groupby(
        df["Date"].dt.to_period("M")
    )["Attendance_Pct"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )

    # Consecutive lectures within the same day
    df["Consecutive_Lecture_Count"] = df.groupby(
        ["Branch", "Section", "Date"]
    ).cumcount() + 1

    # Inter-lecture time gap in hours
    df["Gap_Hours"] = group["Lecture_DateTime"].diff().dt.total_seconds() / 3600.0

    # Fill initial boundary nulls with historical overall mean
    overall_mean = float(df["Attendance_Pct"].mean())
    for col in ["Prev_Lecture_Pct", "Rolling_3_Avg", "Rolling_7_Avg", "Monthly_Avg_Attendance"]:
        df[col] = df[col].fillna(overall_mean)

    df["Gap_Hours"] = df["Gap_Hours"].fillna(24.0)
    df.loc[df["Gap_Hours"] <= 0, "Gap_Hours"] = 24.0

    # Target Classification Labels: CRITICAL (<50%), WARNING (50-75%), SAFE (>75%)
    df[TARGET_CLASSIFICATION] = pd.cut(
        df[TARGET_REGRESSION],
        bins=BAND_BINS,
        labels=ATTENDANCE_BANDS
    )
    df[TARGET_CLASSIFICATION_CODE] = df[TARGET_CLASSIFICATION].cat.codes

    return df


def process_and_save_data(
    raw_path: str = RAW_DATA_FILE,
    cleaned_path: str = CLEANED_DATA_FILE
) -> pd.DataFrame:
    """Orchestrates complete data ingestion, cleaning, and feature engineering."""
    df_raw = load_raw_data(raw_path)
    df_clean = clean_attendance_data(df_raw)
    df_featured = engineer_features(df_clean)
    
    # Save to primary and duplicate to root if needed
    df_featured.to_csv(cleaned_path, index=False)
    root_cleaned = os.path.join(os.path.dirname(os.path.dirname(__file__)), "imcc_cleaned_attendance.csv")
    df_featured.to_csv(root_cleaned, index=False)
    
    return df_featured
