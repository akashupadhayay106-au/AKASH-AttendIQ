"""
AKASH AttendIQ — Smart Guidance Engine
Generates actionable, natural-language guidance based on prediction results.
"""

def generate_smart_guidance(
    predicted_attendance: float, 
    risk_band: str, 
    top_positives: list, 
    top_negatives: list
) -> str:
    """
    Analyzes the inference output and returns a contextual, actionable sentence.
    """
    
    # Extract highest impact factors if they exist
    top_pos_factor = top_positives[0]['factor'] if top_positives else None
    top_neg_factor = top_negatives[0]['factor'] if top_negatives else None

    # Base rule sets depending on Risk Band
    if risk_band == "SAFE":
        if top_pos_factor == "Recent 3-Lecture Moving Avg":
            return "Your recent attendance pattern is strong. Maintaining this trajectory keeps the forecast stable."
        elif top_pos_factor == "Prev_Lecture_Pct":
            return "Your attendance in the immediate previous session is providing strong momentum. Keep it up."
        else:
            return "Your predicted attendance is comfortably above the risk zone. Maintaining your general attendance pattern should keep the forecast stable."

    elif risk_band == "WARNING":
        if top_neg_factor == "Lecture_Number":
            return "Attendance is approaching the risk zone. Session timing (early morning or late afternoon fatigue) is dragging down the forecast. Prioritizing attendance for this specific slot will reverse this trend."
        elif top_neg_factor == "Recent 3-Lecture Moving Avg":
            return "Your attendance is approaching the risk zone due to a recent dip in attendance. Consistently attending the next few lectures is strongly recommended to stabilize your forecast."
        else:
            return "Your attendance forecast is in the warning zone. Consistently attending the next few lectures may significantly improve your forecast."

    elif risk_band == "CRITICAL":
        return f"Critical absenteeism risk detected. Your current pattern indicates high risk of missing this session. Prioritize upcoming sessions and avoid consecutive absences immediately."
    
    return "Prediction generated successfully based on historical patterns."
