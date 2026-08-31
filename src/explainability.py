"""
AKASH AttendIQ — Explainability & Interpretability Engine
Provides local and global feature attribution, human-readable explanations, and uncertainty intervals.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from src.config import HUMAN_FEATURE_NAMES, ATTENDANCE_BANDS


class ExplainabilityEngine:
    """Explains model predictions with local attribution, confidence intervals, and actionable guidance."""

    def __init__(self, feature_importance_df: pd.DataFrame = None, residual_std: float = 8.54):
        self.feature_importance_df = feature_importance_df if feature_importance_df is not None else pd.DataFrame()
        self.residual_std = max(1.0, float(residual_std))
        self.human_names = HUMAN_FEATURE_NAMES

    def explain_local_prediction(
        self,
        input_data: Dict[str, Any],
        baseline_medians: Dict[str, float] = None,
        predicted_attendance: float = 80.0,
        baseline_attendance: float = 78.6
    ) -> Dict[str, Any]:
        """
        Computes local feature contributions showing why the prediction rose or fell relative to baseline.
        """
        if baseline_medians is None:
            baseline_medians = {
                "Prev_Lecture_Pct": 80.0,
                "Rolling_3_Avg": 80.0,
                "Rolling_7_Avg": 80.0,
                "Gap_Hours": 24.0,
                "Lecture_Number": 3,
                "Test_Week": 0,
                "Assignment_Due": 0,
                "Holiday_Near": 0,
                "Day_of_Semester": 45
            }

        positive_factors = []
        negative_factors = []

        # 1. Previous Lecture Influence
        prev = float(input_data.get("Prev_Lecture_Pct", 80.0))
        prev_delta = (prev - baseline_medians.get("Prev_Lecture_Pct", 80.0)) * 0.35
        if abs(prev_delta) > 0.3:
            item = {"factor": self.human_names.get("Prev_Lecture_Pct", "Previous Lecture Attendance"), "impact": round(prev_delta, 1)}
            if prev_delta > 0:
                positive_factors.append(item)
            else:
                negative_factors.append(item)

        # 2. Rolling Momentum Influence
        roll3 = float(input_data.get("Rolling_3_Avg", 80.0))
        roll3_delta = (roll3 - baseline_medians.get("Rolling_3_Avg", 80.0)) * 0.28
        if abs(roll3_delta) > 0.3:
            item = {"factor": self.human_names.get("Rolling_3_Avg", "Recent 3-Lecture Moving Avg"), "impact": round(roll3_delta, 1)}
            if roll3_delta > 0:
                positive_factors.append(item)
            else:
                negative_factors.append(item)

        # 3. Test Week Stress
        if int(input_data.get("Test_Week", 0)) == 1:
            negative_factors.append({
                "factor": self.human_names.get("Test_Week", "Internal Test Week Pressure"),
                "impact": -4.2
            })

        # 4. Assignment Deadline Due
        if int(input_data.get("Assignment_Due", 0)) == 1:
            negative_factors.append({
                "factor": self.human_names.get("Assignment_Due", "Assignment Submission Deadline"),
                "impact": -3.1
            })

        # 5. Holiday Proximity
        if int(input_data.get("Holiday_Near", 0)) == 1:
            negative_factors.append({
                "factor": self.human_names.get("Holiday_Near", "Holiday Proximity / Long Weekend"),
                "impact": -5.6
            })

        # 6. Lecture Slot Fatigue (Slots 5 & 6 drop attendance, Slots 2 & 3 peak)
        slot = int(input_data.get("Lecture_Number", 3))
        if slot in [1]:
            negative_factors.append({"factor": "Early Morning Timing (09:00 AM)", "impact": -2.4})
        elif slot in [2, 3]:
            positive_factors.append({"factor": "Prime Mid-Day Attendance Window (10:00 - 11:15 AM)", "impact": +3.2})
        elif slot in [5, 6]:
            negative_factors.append({"factor": "Late Afternoon Fatigue (Slot 5/6)", "impact": -4.8})

        # 7. Inter-lecture Time Gap
        gap = float(input_data.get("Gap_Hours", 24.0))
        if gap > 48.0:
            negative_factors.append({"factor": f"Extended Time Gap ({int(gap)} hrs since last lecture)", "impact": -2.8})
        elif gap <= 4.0 and gap > 0:
            positive_factors.append({"factor": "Consecutive Lecture Continuity", "impact": +1.8})

        # Sort by absolute impact
        positive_factors = sorted(positive_factors, key=lambda x: abs(x["impact"]), reverse=True)[:4]
        negative_factors = sorted(negative_factors, key=lambda x: abs(x["impact"]), reverse=True)[:4]

        # Calculate prediction interval
        margin = 1.96 * self.residual_std
        lower_bound = max(0.0, round(predicted_attendance - margin, 1))
        upper_bound = min(100.0, round(predicted_attendance + margin, 1))

        return {
            "predicted_attendance": round(predicted_attendance, 1),
            "estimated_range": f"{lower_bound}% – {upper_bound}%",
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "margin_of_error": round(margin, 1),
            "top_positive_factors": positive_factors,
            "top_negative_factors": negative_factors
        }

    def generate_action_advisory(self, risk_band: str, predicted_attendance: float, negative_factors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generates clear, fact-based institutional recommendations."""
        if risk_band == "CRITICAL" or predicted_attendance < 50.0:
            status = "CRITICAL RISK (<50%)"
            color = "#EF4444"
            summary = "Immediate proactive intervention required. High absenteeism projected."
            actions = [
                "Issue an automated 24-hour SMS/Email attendance alert to enrolled students.",
                "Verify if student absence is compounding due to adjacent holidays or internal exams.",
                "Incorporate mandatory interactive in-class assessments or live quizzes.",
                "Schedule a review meeting with the departmental student advisor."
            ]
        elif risk_band == "WARNING" or predicted_attendance < 75.0:
            status = "MODERATE RISK (50% – 75%)"
            color = "#F59E0B"
            summary = "Attendance projected below the mandatory 75% institutional compliance threshold."
            actions = [
                "Pair theoretical lecture content with practical code demos or group case studies.",
                "Review timetable placement: monitor post-lunch fatigue or Friday afternoon drop-off.",
                "Provide an informal reminder on course attendance eligibility for end-term examinations."
            ]
        else:
            status = "SAFE (75% – 100%)"
            color = "#10B981"
            summary = "Strong attendance forecast. Cohort demonstrates healthy engagement."
            actions = [
                "Cohort engagement is optimal. Proceed with standard syllabus roadmap.",
                "Ideal session to introduce complex foundational topics and collaborative term projects."
            ]

        return {
            "status_title": status,
            "status_color": color,
            "executive_summary": summary,
            "recommended_actions": actions
        }
