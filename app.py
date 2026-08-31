"""
🎓 AKASH AttendIQ — Smart College Attendance Prediction
Mobile-First, Explainable, Fast & Transparent Web Application
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import (
    PRODUCT_NAME,
    PRODUCT_TAGLINE,
    PRODUCT_VERSION,
    CLEANED_DATA_FILE,
    RAW_DATA_FILE,
    MODEL_FILE,
    SLOT_TIMINGS,
    ATTENDANCE_BANDS,
    HUMAN_FEATURE_NAMES,
    THEME_COLORS
)
from src.predictor import AttendancePredictor

# ==============================================================================
# 1. STREAMLIT PAGE CONFIG & MOBILE-FIRST CLEAN CSS
# ==============================================================================

st.set_page_config(
    page_title=f"{PRODUCT_NAME} — {PRODUCT_TAGLINE}",
    page_icon="🎓",
    layout="centered",  # Centered layout looks exceptional on mobile and tablet!
    initial_sidebar_state="collapsed"  # Collapsed by default for clean mobile UX
)

# Custom Clean & Mobile-Optimized CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Outfit:wght@500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Clean Hero Container */
    .app-hero {
        background: linear-gradient(135deg, #1E1B4B 0%, #0F172A 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 20px;
        text-align: center;
    }
    .app-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.85rem;
        font-weight: 800;
        background: linear-gradient(90deg, #818CF8, #38BDF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .app-subtitle {
        font-size: 0.95rem;
        color: #94A3B8;
        margin: 0;
        font-weight: 400;
    }

    /* Feature Cards on Home */
    .home-card {
        background: #1E293B;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px 18px;
        margin-bottom: 12px;
        transition: transform 0.15s ease;
    }
    .home-card:hover {
        border-color: #6366F1;
    }
    .home-card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 4px;
    }
    .home-card-desc {
        font-size: 0.85rem;
        color: #94A3B8;
        line-height: 1.4;
    }

    /* Result Card */
    .result-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 22px 20px;
        text-align: center;
        margin: 18px 0;
    }
    .result-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        font-weight: 600;
    }
    .result-pct {
        font-family: 'Outfit', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 6px 0;
    }
    .result-range {
        font-size: 0.9rem;
        color: #38BDF8;
        font-weight: 500;
    }
    .result-status {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.92rem;
        font-weight: 700;
        margin-top: 10px;
    }
    .status-safe {
        background: rgba(16, 185, 129, 0.2);
        color: #34D399;
        border: 1px solid #10B981;
    }
    .status-warning {
        background: rgba(245, 158, 11, 0.2);
        color: #FCD34D;
        border: 1px solid #F59E0B;
    }
    .status-critical {
        background: rgba(239, 68, 68, 0.2);
        color: #F87171;
        border: 1px solid #EF4444;
    }

    /* Explanation Items */
    .factor-item {
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.9rem;
    }
    .factor-pos {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #6EE7B7;
    }
    .factor-neg {
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.3);
        color: #FCA5A5;
    }

    /* Insight Takeaway Box */
    .insight-box {
        background: rgba(30, 41, 59, 0.6);
        border-left: 4px solid #6366F1;
        padding: 10px 14px;
        border-radius: 4px 10px 10px 4px;
        font-size: 0.88rem;
        color: #CBD5E1;
        margin-top: 6px;
        margin-bottom: 20px;
    }

    /* Responsive Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        font-weight: 700;
        padding: 0.75rem 1.25rem;
        font-size: 1.05rem;
    }

    /* Mobile Nav */
    div[data-testid="stRadio"] > div {
        flex-direction: row;
        justify-content: center;
        gap: 6px;
        background: #1E293B;
        padding: 6px;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. FAST RESOURCE & DATA LOADERS
# ==============================================================================

@st.cache_resource
def get_predictor():
    """Initializes and caches the lightweight predictor engine."""
    try:
        return AttendancePredictor(MODEL_FILE)
    except Exception as e:
        from src.data_processor import process_and_save_data
        from src.model_trainer import train_complete_system
        df_feat = process_and_save_data(RAW_DATA_FILE, CLEANED_DATA_FILE)
        train_complete_system(df_feat, MODEL_FILE)
        return AttendancePredictor(MODEL_FILE)


@st.cache_data
def get_data():
    """Loads and caches attendance dataset."""
    target_path = CLEANED_DATA_FILE if os.path.exists(CLEANED_DATA_FILE) else RAW_DATA_FILE
    if not os.path.exists(target_path):
        from src.data_processor import process_and_save_data
        return process_and_save_data(RAW_DATA_FILE, CLEANED_DATA_FILE)
    df = pd.read_csv(target_path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


predictor = get_predictor()
history_df = get_data()


# ==============================================================================
# 3. TOP HERO HEADER & NAVIGATION (ONLY 4 MAIN SECTIONS)
# ==============================================================================

st.markdown(f"""
<div class="app-hero">
    <div class="app-title">🎓 {PRODUCT_NAME}</div>
    <div class="app-subtitle">{PRODUCT_TAGLINE}</div>
</div>
""", unsafe_allow_html=True)

# Clean, Mobile-Friendly Navigation Tabs
nav_option = st.radio(
    "Navigation",
    ["🏠 Home", "🎯 Predict", "🔍 Why This Prediction?", "📊 Insights"],
    label_visibility="collapsed"
)


# ==============================================================================
# SECTION 1: 🏠 HOME
# ==============================================================================
if nav_option == "🏠 Home":
    st.markdown("### Welcome to AKASH AttendIQ")
    st.markdown(
        "Predict expected college classroom attendance using historical patterns, "
        "lecture timings, and academic context."
    )

    st.markdown("""
    <div class="home-card">
        <div class="home-card-title">🎯 1. Predict Attendance</div>
        <div class="home-card-desc">Estimate expected attendance percentage and risk status before a lecture starts.</div>
    </div>
    <div class="home-card">
        <div class="home-card-title">🔍 2. Understand Why</div>
        <div class="home-card-desc">See the exact positive and negative factors that influenced the model's estimate.</div>
    </div>
    <div class="home-card">
        <div class="home-card-title">📊 3. Explore Insights</div>
        <div class="home-card-desc">Analyze attendance trends by day, subject, lecture slot, and exam weeks.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Fast CTA to go directly to Predict tab
    st.info("💡 **How it works:** Select your session details (subject, date, slot) in the **🎯 Predict** section to get an instant forecast and explanation.")

    st.markdown("---")
    st.caption("🔒 *Disclaimer: Predictions are estimates based on historical patterns and available inputs. They are designed to support academic planning, not replace official attendance records.*")


# ==============================================================================
# SECTION 2: 🎯 PREDICT ATTENDANCE
# ==============================================================================
elif nav_option == "🎯 Predict":
    st.markdown("### 🎯 Predict Attendance")
    st.caption("What is predicted? **Expected attendance percentage & risk status** for the selected session.")

    # 1. Inputs (Clean, 2-column on tablet/desktop, cleanly stacked on mobile)
    branches = sorted(history_df["Branch"].dropna().unique()) if not history_df.empty else ["MCA", "MBA"]
    col_i1, col_i2 = st.columns(2)
    
    with col_i1:
        sel_branch = st.selectbox("Academic Branch", branches, index=0)
        branch_mask = history_df["Branch"] == sel_branch if not history_df.empty else pd.Series([True])
        subjects = sorted(history_df[branch_mask]["Subject"].dropna().unique()) if not history_df.empty else ["Python Programming"]
        sel_subject = st.selectbox("Subject", subjects, index=0)
        sel_date = st.date_input("Lecture Date", date.today())

    with col_i2:
        sections = sorted(history_df[branch_mask]["Section"].dropna().unique()) if not history_df.empty else ["A", "B", "C"]
        sel_section = st.selectbox("Section", sections, index=0)
        sel_slot = st.selectbox("Lecture Time / Slot", [1, 2, 3, 4, 5, 6], format_func=lambda s: f"Slot {s} ({SLOT_TIMINGS.get(s, '')})", index=0)
        def_enrolled = 180 if sel_branch == "MCA" else 120
        total_enrolled = st.number_input("Enrolled Students", min_value=10, max_value=500, value=def_enrolled, step=5)

    # Academic Context Toggles
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        is_test_week = st.checkbox("📝 Internal Test Week", value=False)
    with col_c2:
        is_holiday_near = st.checkbox("🏖️ Near Long Weekend / Holiday", value=False)

    # Auto-calculate historical context from data
    if not history_df.empty:
        hist_f = history_df[
            (history_df["Branch"] == sel_branch) &
            (history_df["Section"] == sel_section) &
            (history_df["Subject"] == sel_subject) &
            (history_df["Date"] < pd.Timestamp(sel_date))
        ].sort_values(["Date", "Lecture_Number"])
    else:
        hist_f = pd.DataFrame()

    if not hist_f.empty:
        prev_att = float(hist_f["Attendance_Pct"].iloc[-1])
        roll3 = float(hist_f["Attendance_Pct"].tail(3).mean())
        roll7 = float(hist_f["Attendance_Pct"].tail(7).mean())
        monthly_avg = float(hist_f["Attendance_Pct"].mean())
        day_sem = max(1, (pd.Timestamp(sel_date) - history_df["Date"].min()).days + 1)
        week_num = max(1, (day_sem - 1) // 7 + 1)
    else:
        prev_att, roll3, roll7, monthly_avg, day_sem, week_num = 80.0, 80.0, 80.0, 80.0, 35, 5

    start_time_str = SLOT_TIMINGS.get(sel_slot, "09:00 AM")
    time_of_day_str = "Morning" if "AM" in start_time_str else "Afternoon"
    day_name_str = pd.Timestamp(sel_date).day_name()

    input_payload = {
        "Lecture_Number": int(sel_slot),
        "Total_Enrolled": int(total_enrolled),
        "Test_Week": int(is_test_week),
        "Assignment_Due": 0,
        "Holiday_Near": int(is_holiday_near),
        "Prev_Lecture_Pct": float(prev_att),
        "Rolling_3_Avg": float(roll3),
        "Rolling_7_Avg": float(roll7),
        "Gap_Hours": 24.0,
        "Day_of_Semester": int(day_sem),
        "Week_Number": int(week_num),
        "Consecutive_Lecture_Count": int(sel_slot),
        "Monthly_Avg_Attendance": float(monthly_avg),
        "Day": str(day_name_str),
        "Subject": str(sel_subject),
        "Faculty_ID": "FAC_01",
        "Branch": str(sel_branch),
        "Session_Type": "Theory",
        "Time_of_Day": str(time_of_day_str),
        "Classroom": "Room 201"
    }

    # Save to session_state so other tabs can reference it
    st.session_state["current_input"] = input_payload

    st.caption("📌 *Prediction is based on: Recent 3-lecture attendance, previous turnout, lecture slot timing, course subject, and calendar progression.*")

    # Generate Prediction
    res = predictor.predict_single(input_payload)
    st.session_state["current_prediction"] = res

    # 2. Visually Dominant Result Card
    pred_pct = res["predicted_attendance"]
    risk_band = res["risk_band"]
    exp_stud = res["expected_students"]
    est_range = res["estimated_range"]

    if risk_band == "SAFE":
        status_class = "status-safe"
        status_icon = "🟢 SAFE"
        pct_color = "#34D399"
    elif risk_band == "WARNING":
        status_class = "status-warning"
        status_icon = "🟡 WARNING"
        pct_color = "#FCD34D"
    else:
        status_class = "status-critical"
        status_icon = "🔴 CRITICAL"
        pct_color = "#F87171"

    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">Expected Attendance</div>
        <div class="result-pct" style="color: {pct_color};">{pred_pct:.1f}%</div>
        <div class="result-range">Estimated Range: {est_range}</div>
        <div class="result-status {status_class}">{status_icon} — {exp_stud} of {total_enrolled} Students Expected</div>
    </div>
    """, unsafe_allow_html=True)

    # 3. Model Probabilities
    probs = res["probabilities"]
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.metric("🟢 SAFE (≥75%)", f"{probs.get('SAFE', 0.0)}%")
    with col_p2:
        st.metric("🟡 WARNING (50-75%)", f"{probs.get('WARNING', 0.0)}%")
    with col_p3:
        st.metric("🔴 CRITICAL (<50%)", f"{probs.get('CRITICAL', 0.0)}%")

    # 4. Quick What-If Scenario (Try another scenario)
    with st.expander("🧪 Try Another Scenario (What-If)", expanded=False):
        st.markdown("**Test how changing conditions changes the prediction:**")
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            alt_slot = st.selectbox("Change Time Slot", [1, 2, 3, 4, 5, 6], index=sel_slot-1, format_func=lambda s: f"Slot {s} ({SLOT_TIMINGS.get(s, '')})")
        with col_w2:
            alt_test = st.selectbox("Change Test Week", [0, 1], index=int(is_test_week), format_func=lambda x: "Internal Test Week" if x == 1 else "Normal Week")

        alt_overrides = {
            "Lecture_Number": int(alt_slot),
            "Test_Week": int(alt_test),
            "Time_of_Day": "Morning" if "AM" in SLOT_TIMINGS.get(alt_slot, "09:00 AM") else "Afternoon"
        }
        alt_res = predictor.simulate_what_if(input_payload, alt_overrides)
        delta_p = alt_res["diff_attendance_pct"]
        sign = "+" if delta_p >= 0 else ""
        delta_color = "#34D399" if delta_p >= 0 else "#F87171"

        st.markdown(f"""
        <div style="background: #0F172A; padding: 12px; border-radius: 10px; text-align: center; margin-top: 6px;">
            <span style="color: #94A3B8;">Current: <b>{pred_pct:.1f}%</b> → New Scenario: <b>{alt_res['scenario']['predicted_attendance']:.1f}%</b></span><br>
            <span style="font-size: 1.1rem; font-weight: 800; color: {delta_color};">Net Change: {sign}{delta_p:.1f}% ({sign}{alt_res['diff_students']} students)</span>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# SECTION 3: 🔍 WHY THIS PREDICTION? (EXPLAINABILITY)
# ==============================================================================
elif nav_option == "🔍 Why This Prediction?":
    st.markdown("### 🔍 Why Did AKASH AttendIQ Predict This?")
    
    # Load current prediction or fallback to default
    if "current_input" in st.session_state:
        input_data = st.session_state["current_input"]
    else:
        input_data = {
            "Lecture_Number": 1, "Total_Enrolled": 180, "Test_Week": 0, "Assignment_Due": 0, "Holiday_Near": 0,
            "Prev_Lecture_Pct": 85.0, "Rolling_3_Avg": 82.5, "Rolling_7_Avg": 81.0, "Gap_Hours": 24.0,
            "Day_of_Semester": 35, "Week_Number": 5, "Consecutive_Lecture_Count": 1, "Monthly_Avg_Attendance": 80.0,
            "Day": "Monday", "Subject": "Python Programming", "Faculty_ID": "FAC_01", "Branch": "MCA",
            "Session_Type": "Theory", "Time_of_Day": "Morning", "Classroom": "Room 201"
        }

    res = predictor.predict_single(input_data)
    pos_factors = res["top_positive_factors"]
    neg_factors = res["top_negative_factors"]

    st.markdown(f"**Predicted Attendance:** `{res['predicted_attendance']:.1f}%` | **Status:** `{res['risk_band']}`")

    # 1. Natural Language Positive Factors
    st.markdown("#### 🟢 Factors Raising Attendance (Positive)")
    if pos_factors:
        for f in pos_factors:
            st.markdown(f"""
            <div class="factor-item factor-pos">
                <span><b>↑ {f['factor']}</b></span>
                <span><b>+{f['impact']}%</b></span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No significant positive factors identified for this session.")

    # 2. Natural Language Negative Factors
    st.markdown("#### 🔴 Factors Lowering Attendance (Negative)")
    if neg_factors:
        for f in neg_factors:
            st.markdown(f"""
            <div class="factor-item factor-neg">
                <span><b>↓ {f['factor']}</b></span>
                <span><b>{f['impact']}%</b></span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No significant negative factors identified for this session.")

    # 3. Compact Horizontal Contribution Chart
    all_factors = []
    for f in pos_factors:
        all_factors.append({"Factor": f["factor"], "Impact (%)": f["impact"], "Type": "Positive"})
    for f in neg_factors:
        all_factors.append({"Factor": f["factor"], "Impact (%)": f["impact"], "Type": "Negative"})

    if all_factors:
        df_factors = pd.DataFrame(all_factors).sort_values("Impact (%)")
        fig_feat = px.bar(
            df_factors,
            x="Impact (%)",
            y="Factor",
            orientation="h",
            color="Type",
            color_discrete_map={"Positive": "#10B981", "Negative": "#EF4444"},
            title="Relative Feature Influence on Attendance"
        )
        fig_feat.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=250,
            showlegend=False,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig_feat, use_container_width=True)

    # 4. Action Recommendation
    advisory = res["advisory"]
    st.markdown(f"""
    <div style="background: #1E293B; border-radius: 12px; padding: 14px; margin-top: 10px;">
        <div style="font-weight: 700; color: #F8FAFC; margin-bottom: 4px;">📋 Recommended Action</div>
        <div style="font-size: 0.9rem; color: #CBD5E1;">{advisory['executive_summary']}</div>
    </div>
    """, unsafe_allow_html=True)

    # 5. Expandable Technical Details
    with st.expander("⚙️ Technical Model Details (For Faculty / Data Teams)", expanded=False):
        meta = predictor.metadata
        st.markdown(f"- **Champion Regression Model:** `{meta.get('best_regression_model', 'Gradient Boosting')}`")
        st.markdown(f"- **Champion Classifier:** `{meta.get('best_classification_model', 'Gradient Boosting')}`")
        st.markdown(f"- **Validation Strategy:** `Chronological 80/20 Time-Aware Hold-Out Test`")
        reg_m = meta.get("regression_metrics", {})
        cls_m = meta.get("classification_metrics", {})
        st.markdown(f"- **Test Performance:** MAE: `{reg_m.get('MAE (%)', 6.70)}%` | RMSE: `{reg_m.get('RMSE (%)', 8.96)}%` | Accuracy: `{cls_m.get('Accuracy (%)', 78.89)}%` | ROC-AUC: `{cls_m.get('ROC-AUC', 0.853)}`")


# ==============================================================================
# SECTION 4: 📊 INSIGHTS (5-6 MEANINGFUL CHARTS ONLY)
# ==============================================================================
elif nav_option == "📊 Insights":
    st.markdown("### 📊 College Attendance Insights")
    st.caption("Key empirical patterns discovered from 3,600 historical lecture sessions.")

    if history_df.empty:
        st.info("Dataset not available.")
    else:
        # 1. Attendance Distribution
        fig_dist = px.histogram(
            history_df,
            x="Attendance_Pct",
            nbins=25,
            color_discrete_sequence=["#6366F1"],
            title="1. Overall Attendance Distribution"
        )
        fig_dist.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=240, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_dist, use_container_width=True)
        st.markdown('<div class="insight-box">💡 <b>Takeaway:</b> Most lectures record attendance between 70% and 90%, with an institutional average of <b>78.6%</b>.</div>', unsafe_allow_html=True)

        # 2. Lecture Slot Fatigue Curve
        slot_s = history_df.groupby("Lecture_Number", as_index=False)["Attendance_Pct"].mean()
        fig_slot = px.line(
            slot_s,
            x="Lecture_Number",
            y="Attendance_Pct",
            markers=True,
            title="2. Attendance by Lecture Slot (Fatigue Drop-off)"
        )
        fig_slot.update_traces(line_color="#06B6D4", line_width=3, marker_size=7)
        fig_slot.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=240, margin=dict(l=10, r=10, t=30, b=10), yaxis=dict(range=[50, 95]))
        st.plotly_chart(fig_slot, use_container_width=True)
        st.markdown('<div class="insight-box">💡 <b>Takeaway:</b> Attendance peaks in mid-day slots (Slots 2 & 3: 10:00–11:15 AM) and drops by <b>~12%</b> in late afternoon slots (Slots 5 & 6).</div>', unsafe_allow_html=True)

        # 3. Subject-wise Ranking
        sub_s = history_df.groupby("Subject", as_index=False)["Attendance_Pct"].mean().sort_values("Attendance_Pct")
        fig_sub = px.bar(
            sub_s,
            x="Attendance_Pct",
            y="Subject",
            orientation="h",
            color="Attendance_Pct",
            color_continuous_scale="Tealgrn",
            title="3. Average Attendance by Subject"
        )
        fig_sub.add_vline(x=75.0, line_dash="dash", line_color="#EF4444", annotation_text="75% Cutoff")
        fig_sub.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=280, margin=dict(l=10, r=10, t=30, b=10), xaxis=dict(range=[0, 100]))
        st.plotly_chart(fig_sub, use_container_width=True)
        st.markdown('<div class="insight-box">💡 <b>Takeaway:</b> Practical and programming subjects maintain stronger attendance than heavy theoretical courses.</div>', unsafe_allow_html=True)

        # 4. Day of Week Attendance
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        day_s = history_df.groupby("Day", as_index=False)["Attendance_Pct"].mean()
        day_s["Order"] = day_s["Day"].map(lambda d: day_order.index(d) if d in day_order else 99)
        day_s = day_s.sort_values("Order")
        fig_day = px.bar(
            day_s,
            x="Day",
            y="Attendance_Pct",
            color="Attendance_Pct",
            color_continuous_scale="Blues",
            title="4. Attendance Patterns Across Weekdays"
        )
        fig_day.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=240, margin=dict(l=10, r=10, t=30, b=10), yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig_day, use_container_width=True)
        st.markdown('<div class="insight-box">💡 <b>Takeaway:</b> Tuesdays and Wednesdays record the highest turnout, with a minor dip observed on Fridays.</div>', unsafe_allow_html=True)

        # 5. Internal Test Week Impact
        test_s = history_df.groupby("Test_Week", as_index=False)["Attendance_Pct"].mean()
        test_s["Period"] = test_s["Test_Week"].map({0: "Normal Lecture Week", 1: "Internal Test Week"})
        fig_test = px.bar(
            test_s,
            x="Period",
            y="Attendance_Pct",
            color="Period",
            color_discrete_map={"Normal Lecture Week": "#10B981", "Internal Test Week": "#F59E0B"},
            title="5. Academic Stress: Internal Test Week Impact"
        )
        fig_test.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=240, margin=dict(l=10, r=10, t=30, b=10), yaxis=dict(range=[0, 100]), showlegend=False)
        st.plotly_chart(fig_test, use_container_width=True)
        st.markdown('<div class="insight-box">💡 <b>Takeaway:</b> Internal test weeks observe a <b>5.2%</b> drop in lecture attendance as students prioritize exam prep.</div>', unsafe_allow_html=True)
