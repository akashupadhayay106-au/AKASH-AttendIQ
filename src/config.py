"""
AKASH AttendIQ — System Configuration & Constants
AI-Powered College Attendance Prediction, Risk Intelligence & Analytics
"""

import os
from pathlib import Path

# Product Branding
PRODUCT_NAME = "AKASH AttendIQ"
PRODUCT_TAGLINE = "AI-Powered College Attendance Prediction, Risk Intelligence & Analytics"
PRODUCT_VERSION = "2.5.0"

# Base Directory Resolution
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Dataset & Model Paths (Supports both data/ and root locations)
RAW_DATA_FILE = os.path.join(DATA_DIR, "attendance.csv")
if not os.path.exists(RAW_DATA_FILE):
    RAW_DATA_FILE = os.path.join(BASE_DIR, "imcc_raw_attendance.csv")

CLEANED_DATA_FILE = os.path.join(DATA_DIR, "cleaned_attendance.csv")
if not os.path.exists(CLEANED_DATA_FILE):
    CLEANED_DATA_FILE = os.path.join(BASE_DIR, "imcc_cleaned_attendance.csv")

MODEL_FILE = os.path.join(MODELS_DIR, "imcc_attendance_ml_pipeline.pkl")
ROOT_MODEL_FILE = os.path.join(BASE_DIR, "imcc_attendance_ml_pipeline.pkl")

# Feature Column Definitions
CATEGORICAL_FEATURES = [
    "Day",
    "Subject",
    "Faculty_ID",
    "Branch",
    "Session_Type",
    "Time_of_Day",
    "Classroom"
]

NUMERICAL_FEATURES = [
    "Lecture_Number",
    "Total_Enrolled",
    "Test_Week",
    "Assignment_Due",
    "Holiday_Near",
    "Prev_Lecture_Pct",
    "Rolling_3_Avg",
    "Rolling_7_Avg",
    "Gap_Hours",
    "Day_of_Semester",
    "Week_Number",
    "Consecutive_Lecture_Count",
    "Monthly_Avg_Attendance"
]

ALL_MODEL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Target Variables
TARGET_REGRESSION = "Attendance_Pct"
TARGET_CLASSIFICATION = "Attendance_Risk"
TARGET_CLASSIFICATION_CODE = "Attendance_Risk_Code"

# Attendance Risk Bands
# Ordered from 0 (CRITICAL) to 1 (WARNING) to 2 (SAFE)
ATTENDANCE_BANDS = ["CRITICAL", "WARNING", "SAFE"]
BAND_BINS = [-float("inf"), 50.0, 75.0, float("inf")]

# Human-Readable Feature Mapping for Explainability
HUMAN_FEATURE_NAMES = {
    "Prev_Lecture_Pct": "Previous Lecture Attendance",
    "Rolling_3_Avg": "Recent 3-Lecture Moving Avg",
    "Rolling_7_Avg": "Recent 7-Lecture Moving Avg",
    "Gap_Hours": "Inter-Lecture Time Gap (Hours)",
    "Lecture_Number": "Lecture Slot (Timing Fatigue)",
    "Total_Enrolled": "Classroom Enrollment Strength",
    "Test_Week": "Internal Test Week Pressure",
    "Assignment_Due": "Assignment Submission Due",
    "Holiday_Near": "Holiday Proximity Indicator",
    "Day_of_Semester": "Semester Progression (Days)",
    "Week_Number": "Academic Week of Semester",
    "Consecutive_Lecture_Count": "Consecutive Lectures Today",
    "Monthly_Avg_Attendance": "Monthly Historical Benchmark",
    "Day": "Day of the Week",
    "Subject": "Academic Subject",
    "Faculty_ID": "Faculty Identifier",
    "Branch": "Academic Branch / Program",
    "Session_Type": "Session Format (Theory vs Lab)",
    "Time_of_Day": "Session Timing (Morning vs Afternoon)",
    "Classroom": "Assigned Classroom / Lab"
}

# Timetable Schedule Mapping
SLOT_TIMINGS = {
    1: "09:00 AM",
    2: "10:00 AM",
    3: "11:15 AM",
    4: "12:15 PM",
    5: "02:00 PM",
    6: "03:00 PM"
}

# Theme Styling Tokens
THEME_COLORS = {
    "primary": "#4F46E5",       # Indigo
    "secondary": "#06B6D4",     # Cyan
    "accent": "#8B5CF6",        # Purple
    "safe": "#10B981",          # Emerald Green
    "warning": "#F59E0B",       # Amber Orange
    "critical": "#EF4444",      # Crimson Red
    "background_dark": "#0F172A",
    "card_dark": "#1E293B",
    "text_muted": "#94A3B8"
}
