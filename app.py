"""
AKASH AttendIQ — Production Streamlit Web Dashboard
AI-Powered College Attendance Prediction, Risk Intelligence & Analytics
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
# 1. STREAMLIT PAGE CONFIG & MODERN CSS DESIGN SYSTEM
# ==============================================================================

st.set_page_config(
    page_title=f"{PRODUCT_NAME} — College Attendance Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', 'Inter', sans-serif;
    }

    /* Executive Hero Header */
    .hero-banner {
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.18) 0%, rgba(6, 182, 212, 0.12) 50%, rgba(139, 92, 246, 0.18) 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 18px;
        padding: 24px 32px;
        margin-bottom: 24px;
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.22);
    }
    
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #60A5FA, #A78BFA, #34D399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    
    .hero-subtitle {
        font-size: 1.08rem;
        color: #94A3B8;
        font-weight: 400;
        margin: 0;
    }

    /* Premium Metric KPI Card */
    .kpi-card {
        background: rgba(30, 41, 59, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 20px;
        text-align: center;
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 24px -10px rgba(79, 70, 229, 0.4);
    }
    .kpi-title {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94A3B8;
        margin-bottom: 4px;
        font-weight: 600;
    }
    .kpi-value {
        font-size: 1.95rem;
        font-weight: 800;
        color: #F8FAFC;
    }
    .kpi-sub {
        font-size: 0.82rem;
        color: #38BDF8;
        margin-top: 4px;
    }

    /* Factor Contribution Tags */
    .factor-tag-pos {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #34D399;
        border-radius: 8px;
        padding: 6px 12px;
        margin: 4px 0;
        font-size: 0.88rem;
        font-weight: 500;
        display: flex;
        justify-content: space-between;
    }
    .factor-tag-neg {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.35);
        color: #F87171;
        border-radius: 8px;
        padding: 6px 12px;
        margin: 4px 0;
        font-size: 0.88rem;
        font-weight: 500;
        display: flex;
        justify-content: space-between;
    }

    /* Risk Alerts */
    .risk-alert {
        border-radius: 12px;
        padding: 16px 20px;
        margin: 16px 0;
    }
    .risk-alert-crit {
        background: rgba(239, 68, 68, 0.14);
        border: 1px solid rgba(239, 68, 68, 0.4);
        color: #FCA5A5;
    }
    .risk-alert-warn {
        background: rgba(245, 158, 11, 0.14);
        border: 1px solid rgba(245, 158, 11, 0.4);
        color: #FCD34D;
    }
    .risk-alert-safe {
        background: rgba(16, 185, 129, 0.14);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #6EE7B7;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 23, 42, 0.6);
        padding: 6px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. DATA & ENGINE LOADERS
# ==============================================================================

@st.cache_resource
def get_predictor_engine():
    """Initializes and caches the AKASH AttendIQ inference engine with automated self-healing."""
    try:
        return AttendancePredictor(MODEL_FILE)
    except Exception as e:
        with st.spinner("⚡ Initializing & compiling ML models for cloud environment..."):
            from src.data_processor import process_and_save_data
            from src.model_trainer import train_complete_system
            df_featured = process_and_save_data(RAW_DATA_FILE, CLEANED_DATA_FILE)
            train_complete_system(df_featured, MODEL_FILE)
            return AttendancePredictor(MODEL_FILE)


@st.cache_data
def get_dataset():
    """Loads and caches cleaned institutional attendance data."""
    target_path = CLEANED_DATA_FILE if os.path.exists(CLEANED_DATA_FILE) else RAW_DATA_FILE
    if not os.path.exists(target_path):
        from src.data_processor import process_and_save_data
        return process_and_save_data(RAW_DATA_FILE, CLEANED_DATA_FILE)
    df = pd.read_csv(target_path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


predictor = get_predictor_engine()
history_df = get_dataset()


# ==============================================================================
# 3. EXECUTIVE HERO HEADER
# ==============================================================================

st.markdown(f"""
<div class="hero-banner">
    <div class="hero-title">🎓 {PRODUCT_NAME}</div>
    <div class="hero-subtitle">{PRODUCT_TAGLINE}</div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# 4. SIDEBAR INPUT PARAMETERS
# ==============================================================================

st.sidebar.markdown(f"### ⚙️ Session Setup — {PRODUCT_NAME}")

# Branch
branches = sorted(history_df["Branch"].dropna().unique()) if not history_df.empty else ["MCA", "MBA"]
selected_branch = st.sidebar.selectbox("🏛️ Academic Branch", branches, index=0)

# Section
branch_mask = history_df["Branch"] == selected_branch if not history_df.empty else pd.Series([True])
sections = sorted(history_df[branch_mask]["Section"].dropna().unique()) if not history_df.empty else ["A", "B", "C"]
selected_section = st.sidebar.selectbox("👥 Section / Cohort", sections, index=0)

# Subject
subjects = sorted(history_df[branch_mask]["Subject"].dropna().unique()) if not history_df.empty else ["Python Programming"]
selected_subject = st.sidebar.selectbox("📚 Course Subject", subjects, index=0)

# Faculty
subj_mask = (history_df["Branch"] == selected_branch) & (history_df["Subject"] == selected_subject) if not history_df.empty else pd.Series([True])
faculties = sorted(history_df[subj_mask]["Faculty_ID"].dropna().unique()) if not history_df.empty else ["FAC_01"]
selected_faculty = st.sidebar.selectbox("👨‍🏫 Faculty ID", faculties, index=0)

# Classroom
classrooms = sorted(history_df[subj_mask]["Classroom"].dropna().unique()) if not history_df.empty else ["Room 201"]
selected_classroom = st.sidebar.selectbox("🚪 Classroom / Hall", classrooms, index=0)
session_type = "Practical" if "Lab" in selected_classroom else "Theory"

st.sidebar.divider()

# Date & Slot Selection
col_d1, col_d2 = st.sidebar.columns(2)
with col_d1:
    selected_date = st.date_input("📅 Date", date.today())
with col_d2:
    selected_slot = st.selectbox("⏰ Lecture Slot", [1, 2, 3, 4, 5, 6], index=0)

start_time = SLOT_TIMINGS.get(selected_slot, "09:00 AM")
time_of_day = "Morning" if "AM" in start_time else "Afternoon"
day_name = pd.Timestamp(selected_date).day_name()

# Total Enrolled
if not history_df.empty and "Total_Enrolled" in history_df.columns:
    def_enrolled = int(history_df[subj_mask]["Total_Enrolled"].median()) if not history_df[subj_mask].empty else 120
else:
    def_enrolled = 180 if selected_branch == "MCA" else 120

total_enrolled = st.sidebar.number_input("🎒 Classroom Enrollment", min_value=10, max_value=500, value=def_enrolled, step=5)

# Academic Context Flags
st.sidebar.markdown("**⚡ Academic Context & Constraints**")
col_c1, col_c2 = st.sidebar.columns(2)
with col_c1:
    is_test_week = st.checkbox("📝 Internal Test Week", value=False)
    is_assignment_due = st.checkbox("📑 Assignment Due", value=False)
with col_c2:
    is_holiday_near = st.checkbox("🏖️ Holiday Near", value=False)
    manual_lags = st.checkbox("🛠️ Manual Overrides", value=False)

# Historical auto-calculation
if not history_df.empty:
    hist_filtered = history_df[
        (history_df["Branch"] == selected_branch) &
        (history_df["Section"] == selected_section) &
        (history_df["Subject"] == selected_subject) &
        (history_df["Date"] < pd.Timestamp(selected_date))
    ].sort_values(["Date", "Lecture_Number"])
else:
    hist_filtered = pd.DataFrame()

if not hist_filtered.empty:
    auto_prev = float(hist_filtered["Attendance_Pct"].iloc[-1])
    auto_roll3 = float(hist_filtered["Attendance_Pct"].tail(3).mean())
    auto_roll7 = float(hist_filtered["Attendance_Pct"].tail(7).mean())
    auto_monthly = float(hist_filtered["Attendance_Pct"].mean())
    sem_start = history_df["Date"].min()
    auto_day_sem = max(1, (pd.Timestamp(selected_date) - sem_start).days + 1)
    auto_week = max(1, (auto_day_sem - 1) // 7 + 1)
    
    last_dt = pd.Timestamp(hist_filtered["Date"].iloc[-1])
    cur_dt = pd.Timestamp(f"{selected_date} {start_time}")
    auto_gap = max(1.0, (cur_dt - last_dt).total_seconds() / 3600.0)
else:
    auto_prev = 80.0
    auto_roll3 = 80.0
    auto_roll7 = 80.0
    auto_monthly = 80.0
    auto_day_sem = 35
    auto_week = 5
    auto_gap = 24.0

if manual_lags:
    st.sidebar.markdown("**⚙️ Custom Feature Inputs**")
    prev_lecture_pct = st.sidebar.slider("Previous Lecture Attendance (%)", 0.0, 100.0, float(auto_prev), 0.5)
    rolling_3_avg = st.sidebar.slider("Recent 3-Lecture Avg (%)", 0.0, 100.0, float(auto_roll3), 0.5)
    rolling_7_avg = st.sidebar.slider("Recent 7-Lecture Avg (%)", 0.0, 100.0, float(auto_roll7), 0.5)
    gap_hours = st.sidebar.slider("Inter-Lecture Gap (Hours)", 1.0, 168.0, float(auto_gap), 1.0)
else:
    prev_lecture_pct = auto_prev
    rolling_3_avg = auto_roll3
    rolling_7_avg = auto_roll7
    gap_hours = auto_gap

# Bundle input dict
input_payload = {
    "Lecture_Number": int(selected_slot),
    "Total_Enrolled": int(total_enrolled),
    "Test_Week": int(is_test_week),
    "Assignment_Due": int(is_assignment_due),
    "Holiday_Near": int(is_holiday_near),
    "Prev_Lecture_Pct": float(prev_lecture_pct),
    "Rolling_3_Avg": float(rolling_3_avg),
    "Rolling_7_Avg": float(rolling_7_avg),
    "Gap_Hours": float(gap_hours),
    "Day_of_Semester": int(auto_day_sem),
    "Week_Number": int(auto_week),
    "Consecutive_Lecture_Count": int(selected_slot),
    "Monthly_Avg_Attendance": float(auto_monthly),
    "Day": str(day_name),
    "Subject": str(selected_subject),
    "Faculty_ID": str(selected_faculty),
    "Branch": str(selected_branch),
    "Session_Type": str(session_type),
    "Time_of_Day": str(time_of_day),
    "Classroom": str(selected_classroom)
}


# ==============================================================================
# 5. NAVIGATION TABS
# ==============================================================================

tab_exec, tab_pred, tab_whatif, tab_eda, tab_model, tab_batch = st.tabs([
    "📈 Executive Dashboard",
    "🔮 Attendance Predictor",
    "🧪 What-If Simulator",
    "📊 Deep Analytics & EDA",
    "🤖 Model Intelligence & Explainability",
    "📁 Batch Processing & Alerts"
])


# ==============================================================================
# TAB 1: EXECUTIVE DASHBOARD
# ==============================================================================
with tab_exec:
    st.subheader("📈 Executive Attendance Overview & KPIs")

    if not history_df.empty:
        total_recs = len(history_df)
        avg_att = history_df["Attendance_Pct"].mean()
        crit_count = len(history_df[history_df["Attendance_Pct"] < 50.0])
        warn_count = len(history_df[(history_df["Attendance_Pct"] >= 50.0) & (history_df["Attendance_Pct"] < 75.0)])
        safe_count = len(history_df[history_df["Attendance_Pct"] >= 75.0])
    else:
        total_recs, avg_att, crit_count, warn_count, safe_count = 3600, 78.6, 88, 1202, 2310

    meta = predictor.metadata
    reg_m = meta.get("regression_metrics", {})
    cls_m = meta.get("classification_metrics", {})

    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
    with col_e1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Monitored Sessions</div>
            <div class="kpi-value">{total_recs:,}</div>
            <div class="kpi-sub">Spring 2026 Semester</div>
        </div>
        """, unsafe_allow_html=True)
    with col_e2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Institutional Average</div>
            <div class="kpi-value">{avg_att:.1f}%</div>
            <div class="kpi-sub">Target: ≥ 75.0%</div>
        </div>
        """, unsafe_allow_html=True)
    with col_e3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Sessions At-Risk (&lt;75%)</div>
            <div class="kpi-value" style="color: #F59E0B;">{crit_count + warn_count:,}</div>
            <div class="kpi-sub">{((crit_count + warn_count)/total_recs)*100:.1f}% of total sessions</div>
        </div>
        """, unsafe_allow_html=True)
    with col_e4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Champion Model Accuracy</div>
            <div class="kpi-value" style="color: #10B981;">{cls_m.get('Accuracy (%)', 78.89):.1f}%</div>
            <div class="kpi-sub">ROC-AUC: {cls_m.get('ROC-AUC', 0.853):.3f} | R²: {reg_m.get('R2 Score', 0.466):.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    col_t1, col_t2 = st.columns([1.5, 1])
    with col_t1:
        st.markdown("#### 📅 Attendance Trend Across Semester")
        if not history_df.empty:
            daily_t = history_df.groupby("Date", as_index=False).agg(
                Attendance_Pct=("Attendance_Pct", "mean"),
                Test_Week=("Test_Week", "max")
            ).sort_values("Date")
            fig_trend = px.line(daily_t, x="Date", y="Attendance_Pct", title="Daily Attendance Trajectory")
            fig_trend.add_hline(y=75.0, line_dash="dash", line_color="#EF4444", annotation_text="75% Mandatory Benchmark")
            fig_trend.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=320)
            st.plotly_chart(fig_trend, use_container_width=True)

    with col_t2:
        st.markdown("#### 🛡️ Institutional Risk Distribution")
        fig_risk_pie = go.Figure(data=[go.Pie(
            labels=["SAFE (≥75%)", "WARNING (50-75%)", "CRITICAL (<50%)"],
            values=[safe_count, warn_count, crit_count],
            hole=0.5,
            marker_colors=["#10B981", "#F59E0B", "#EF4444"]
        )])
        fig_risk_pie.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=320, margin=dict(l=10, r=10, t=20, b=20))
        st.plotly_chart(fig_risk_pie, use_container_width=True)


# ==============================================================================
# TAB 2: ATTENDANCE PREDICTOR
# ==============================================================================
with tab_pred:
    st.subheader("🔮 Explainable Future Lecture Attendance Forecast")

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        calc = st.button("🚀 Calculate Attendance Forecast", type="primary", use_container_width=True)

    # Perform Prediction
    res = predictor.predict_single(input_payload)
    pred_pct = res["predicted_attendance"]
    pred_band = res["risk_band"]
    pred_color = res["risk_color"]
    est_range = res["estimated_range"]
    exp_stud = res["expected_students"]
    exp_abs = res["expected_absent"]
    tot_stud = res["total_enrolled"]
    probs = res["probabilities"]

    # 4 KPI Cards
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Predicted Attendance</div>
            <div class="kpi-value" style="color: {pred_color};">{pred_pct:.1f}%</div>
            <div class="kpi-sub">Range: {est_range}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_p2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Risk Classification</div>
            <div class="kpi-value" style="color: {pred_color}; font-size: 1.6rem; margin-top: 4px;">{pred_band}</div>
            <div class="kpi-sub">Confidence: {res['confidence_pct']}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col_p3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Expected Headcount</div>
            <div class="kpi-value">{exp_stud}</div>
            <div class="kpi-sub">Present / {tot_stud} Enrolled</div>
        </div>
        """, unsafe_allow_html=True)
    with col_p4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Expected Absenteeism</div>
            <div class="kpi-value" style="color: #F87171;">{exp_abs}</div>
            <div class="kpi-sub">Absentee Rate: {100.0 - pred_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # Risk Probabilities Progress Bar
    st.markdown("#### 🎯 Prediction Probability Breakdown")
    col_pb1, col_pb2, col_pb3 = st.columns(3)
    with col_pb1:
        st.markdown(f"**SAFE (≥75%):** `{probs.get('SAFE', 0.0)}%`")
        st.progress(probs.get('SAFE', 0.0) / 100.0)
    with col_pb2:
        st.markdown(f"**WARNING (50-75%):** `{probs.get('WARNING', 0.0)}%`")
        st.progress(probs.get('WARNING', 0.0) / 100.0)
    with col_pb3:
        st.markdown(f"**CRITICAL (&lt;50%):** `{probs.get('CRITICAL', 0.0)}%`")
        st.progress(probs.get('CRITICAL', 0.0) / 100.0)

    # Visual Gauge & Headcount Comparison
    col_vg1, col_vg2 = st.columns(2)
    with col_vg1:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred_pct,
            delta={"reference": prev_lecture_pct, "valueformat": ".1f%", "increasing": {"color": "#10B981"}, "decreasing": {"color": "#EF4444"}},
            title={"text": "Predicted vs Previous Attendance", "font": {"size": 15, "color": "#F8FAFC"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94A3B8"},
                "bar": {"color": pred_color, "thickness": 0.28},
                "bgcolor": "rgba(30, 41, 59, 0.6)",
                "steps": [
                    {"range": [0, 50], "color": "rgba(239, 68, 68, 0.25)"},
                    {"range": [50, 75], "color": "rgba(245, 158, 11, 0.25)"},
                    {"range": [75, 100], "color": "rgba(16, 185, 129, 0.25)"}
                ],
                "threshold": {"line": {"color": "#38BDF8", "width": 3}, "thickness": 0.8, "value": 75.0}
            }
        ))
        fig_g.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=260, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_g, use_container_width=True)

    with col_vg2:
        fig_d = go.Figure(data=[go.Pie(
            labels=["Expected Present", "Expected Absent"],
            values=[exp_stud, exp_abs],
            hole=0.55,
            marker_colors=["#10B981", "#EF4444"],
            textinfo="label+percent+value"
        )])
        fig_d.update_layout(
            title={"text": f"Projected Classroom Fill ({selected_classroom})", "font": {"size": 15, "color": "#F8FAFC"}},
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=260, showlegend=False, margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_d, use_container_width=True)

    # Explainability: Why did the model make this prediction?
    st.markdown("---")
    st.markdown("### 🔍 Why Did AKASH AttendIQ Make This Prediction?")
    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        st.markdown("**🟢 Top Positive Catalysts (Boosting Attendance)**")
        pos_factors = res["top_positive_factors"]
        if pos_factors:
            for f in pos_factors:
                st.markdown(f"""
                <div class="factor-tag-pos">
                    <span>{f['factor']}</span>
                    <span>+{f['impact']}%</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No significant positive catalysts identified for this session.")

    with col_exp2:
        st.markdown("**🔴 Top Negative Catalysts (Suppressing Attendance)**")
        neg_factors = res["top_negative_factors"]
        if neg_factors:
            for f in neg_factors:
                st.markdown(f"""
                <div class="factor-tag-neg">
                    <span>{f['factor']}</span>
                    <span>{f['impact']}%</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No significant negative catalysts identified for this session.")

    # Actionable Advisory
    advisory = res["advisory"]
    adv_class = "risk-alert-crit" if "CRITICAL" in pred_band else ("risk-alert-warn" if "WARNING" in pred_band else "risk-alert-safe")
    st.markdown(f"""
    <div class="risk-alert {adv_class}">
        <h4 style="margin:0 0 6px 0;">📋 Institutional Action Advisory: {advisory['status_title']}</h4>
        <p style="margin-bottom: 8px;">{advisory['executive_summary']}</p>
        <ul style="margin: 0; padding-left: 20px;">
            {''.join([f'<li>{a}</li>' for a in advisory['recommended_actions']])}
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# TAB 3: WHAT-IF SIMULATOR
# ==============================================================================
with tab_whatif:
    st.subheader("🧪 Interactive What-If Scenario Simulator")
    st.markdown("Modify operational parameters to simulate real-time attendance sensitivity and measure the exact delta.")

    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1:
        sim_slot = st.selectbox("Simulate Slot Change", [1, 2, 3, 4, 5, 6], index=selected_slot-1)
    with col_w2:
        sim_test = st.selectbox("Simulate Test Week", [0, 1], index=int(is_test_week), format_func=lambda x: "Yes (Test Week)" if x == 1 else "No (Normal Week)")
    with col_w3:
        sim_holiday = st.selectbox("Simulate Holiday Proximity", [0, 1], index=int(is_holiday_near), format_func=lambda x: "Yes (Near Holiday)" if x == 1 else "No (Normal Day)")

    sim_overrides = {
        "Lecture_Number": int(sim_slot),
        "Test_Week": int(sim_test),
        "Holiday_Near": int(sim_holiday),
        "Time_of_Day": "Morning" if "AM" in SLOT_TIMINGS.get(sim_slot, "09:00 AM") else "Afternoon"
    }

    sim_res = predictor.simulate_what_if(input_payload, sim_overrides)
    b_res = sim_res["baseline"]
    s_res = sim_res["scenario"]
    diff_p = sim_res["diff_attendance_pct"]
    diff_s = sim_res["diff_students"]

    col_cmp1, col_cmp2, col_cmp3 = st.columns(3)
    with col_cmp1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Baseline Forecast</div>
            <div class="kpi-value">{b_res['predicted_attendance']:.1f}%</div>
            <div class="kpi-sub">Risk: {b_res['risk_band']} ({b_res['expected_students']} students)</div>
        </div>
        """, unsafe_allow_html=True)
    with col_cmp2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Scenario Forecast</div>
            <div class="kpi-value" style="color: {s_res['risk_color']};">{s_res['predicted_attendance']:.1f}%</div>
            <div class="kpi-sub">Risk: {s_res['risk_band']} ({s_res['expected_students']} students)</div>
        </div>
        """, unsafe_allow_html=True)
    with col_cmp3:
        diff_color = "#10B981" if diff_p >= 0 else "#EF4444"
        diff_sign = "+" if diff_p >= 0 else ""
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Net Impact (Delta)</div>
            <div class="kpi-value" style="color: {diff_color};">{diff_sign}{diff_p:.1f}%</div>
            <div class="kpi-sub">{diff_sign}{diff_s} Expected Students</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### 🔄 Changed Factors")
    if sim_res["changed_factors"]:
        for factor in sim_res["changed_factors"]:
            st.markdown(f"- `{factor}`")
    else:
        st.info("Scenario matches baseline configuration.")


# ==============================================================================
# TAB 4: DEEP ANALYTICS & EDA (10 Meaningful Charts)
# ==============================================================================
with tab_eda:
    st.subheader("📊 Deep Exploratory Data Analysis & Analytical Charts")

    if history_df.empty:
        st.info("Historical dataset not loaded.")
    else:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            sel_b_eda = st.selectbox("Select Branch for Analytics", ["All"] + branches)
        with col_f2:
            eda_df = history_df if sel_b_eda == "All" else history_df[history_df["Branch"] == sel_b_eda]
            st.metric("Analyzed Records", f"{len(eda_df):,} Sessions")

        # Row 1: Distribution & Risk breakdown
        col_r1_1, col_r1_2 = st.columns(2)
        with col_r1_1:
            # 1. Attendance Distribution
            fig_dist = px.histogram(eda_df, x="Attendance_Pct", nbins=30, marginal="box", title="1. Overall Attendance % Distribution", color_discrete_sequence=["#4F46E5"])
            fig_dist.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_dist, use_container_width=True)

        with col_r1_2:
            # 2. Risk Distribution by Branch
            fig_rb = px.histogram(history_df, x="Branch", color="Attendance_Risk", barmode="group", title="2. Risk Tier Frequency Across Branches", color_discrete_map={"SAFE": "#10B981", "WARNING": "#F59E0B", "CRITICAL": "#EF4444"})
            fig_rb.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_rb, use_container_width=True)

        # Row 2: Day of Week & Slot Fatigue
        col_r2_1, col_r2_2 = st.columns(2)
        with col_r2_1:
            # 3. Attendance by Day
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
            day_s = eda_df.groupby("Day", as_index=False)["Attendance_Pct"].mean()
            day_s["Order"] = day_s["Day"].map(lambda d: day_order.index(d) if d in day_order else 99)
            day_s = day_s.sort_values("Order")
            fig_day = px.bar(day_s, x="Day", y="Attendance_Pct", color="Attendance_Pct", color_continuous_scale="Blues", title="3. Day-of-Week Attendance Pattern")
            fig_day.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(range=[0, 100]))
            st.plotly_chart(fig_day, use_container_width=True)

        with col_r2_2:
            # 4. Lecture Slot Fatigue
            slot_s = eda_df.groupby("Lecture_Number", as_index=False)["Attendance_Pct"].mean()
            fig_slot = px.line(slot_s, x="Lecture_Number", y="Attendance_Pct", markers=True, title="4. Lecture Slot Attendance Drop-Off Curve")
            fig_slot.update_traces(line_color="#06B6D4", line_width=3)
            fig_slot.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(range=[0, 100]))
            st.plotly_chart(fig_slot, use_container_width=True)

        # Row 3: Subject Rankings & Test Week Boxplot
        col_r3_1, col_r3_2 = st.columns(2)
        with col_r3_1:
            # 5. Attendance by Subject
            sub_s = eda_df.groupby("Subject", as_index=False)["Attendance_Pct"].mean().sort_values("Attendance_Pct")
            fig_sub = px.bar(sub_s, x="Attendance_Pct", y="Subject", orientation="h", color="Attendance_Pct", color_continuous_scale="Tealgrn", title="5. Course Subject Performance Ranking")
            fig_sub.add_vline(x=75.0, line_dash="dash", line_color="#EF4444", annotation_text="75% Cutoff")
            fig_sub.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(range=[0, 100]))
            st.plotly_chart(fig_sub, use_container_width=True)

        with col_r3_2:
            # 6. Test Week vs Normal Week Impact
            fig_box = px.box(eda_df, x="Test_Week", y="Attendance_Pct", color="Test_Week", title="6. Academic Pressure: Test Week Impact", labels={"Test_Week": "0 = Normal Week, 1 = Test Week"})
            fig_box.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_box, use_container_width=True)

        # Row 4: Classroom & Correlation
        col_r4_1, col_r4_2 = st.columns(2)
        with col_r4_1:
            # 7. Classroom Attendance
            cr_s = eda_df.groupby("Classroom", as_index=False)["Attendance_Pct"].mean()
            fig_cr = px.bar(cr_s, x="Classroom", y="Attendance_Pct", color="Attendance_Pct", color_continuous_scale="Purples", title="7. Classroom & Lab Utilization")
            fig_cr.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(range=[0, 100]))
            st.plotly_chart(fig_cr, use_container_width=True)

        with col_r4_2:
            # 8. Feature Correlation Matrix
            num_cols = ["Attendance_Pct", "Prev_Lecture_Pct", "Rolling_3_Avg", "Rolling_7_Avg", "Gap_Hours", "Day_of_Semester"]
            corr = eda_df[num_cols].corr()
            fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", title="8. Numerical Feature Correlation Heatmap")
            fig_corr.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_corr, use_container_width=True)


# ==============================================================================
# TAB 5: MODEL INTELLIGENCE & EXPLAINABILITY
# ==============================================================================
with tab_model:
    st.subheader("🤖 Model Intelligence, Leaderboards & Explainability")

    meta = predictor.metadata
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        st.markdown("#### 📈 Regression Model Leaderboard")
        if "regression_leaderboard" in meta:
            st.dataframe(pd.DataFrame(meta["regression_leaderboard"]), use_container_width=True, hide_index=True)
    with col_l2:
        st.markdown("#### 🎯 Classification Model Leaderboard")
        if "classification_leaderboard" in meta:
            st.dataframe(pd.DataFrame(meta["classification_leaderboard"]), use_container_width=True, hide_index=True)

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("#### 🔍 Global Feature Importance")
        if "feature_importance" in meta and meta["feature_importance"]:
            feat_df = pd.DataFrame(meta["feature_importance"]).head(10).sort_values("Importance")
            fig_fi = px.bar(feat_df, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Viridis", title="Top 10 Global Attendance Drivers")
            fig_fi.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=320)
            st.plotly_chart(fig_fi, use_container_width=True)

    with col_m2:
        st.markdown("#### 🧩 Multi-Class Confusion Matrix")
        if "confusion_matrix" in meta:
            cm = np.array(meta["confusion_matrix"])
            fig_cm = px.imshow(cm, text_auto=True, x=ATTENDANCE_BANDS, y=ATTENDANCE_BANDS, labels=dict(x="Predicted", y="Actual"), color_continuous_scale="Blues", title="Confusion Matrix (Champion Model)")
            fig_cm.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=320)
            st.plotly_chart(fig_cm, use_container_width=True)


# ==============================================================================
# TAB 6: BATCH PROCESSING & ALERTS
# ==============================================================================
with tab_batch:
    st.subheader("📁 Batch CSV Prediction & Institutional Low-Attendance Alerts")

    col_u1, col_u2 = st.columns([2, 1])
    with col_u1:
        st.markdown("Upload a timetable or session schedule CSV to generate bulk attendance forecasts and risk classifications.")
        uploaded = st.file_uploader("Upload Lecture Schedule CSV", type=["csv"])

    with col_u2:
        st.markdown("Download ready-made testing template:")
        sample_template = pd.DataFrame([{
            "Lecture_Number": 1, "Total_Enrolled": 180, "Test_Week": 0, "Assignment_Due": 0, "Holiday_Near": 0,
            "Prev_Lecture_Pct": 82.0, "Rolling_3_Avg": 80.0, "Rolling_7_Avg": 79.0, "Gap_Hours": 24.0,
            "Day_of_Semester": 40, "Week_Number": 6, "Consecutive_Lecture_Count": 1, "Monthly_Avg_Attendance": 78.0,
            "Day": "Monday", "Subject": "Python Programming", "Faculty_ID": "FAC_01", "Branch": "MCA",
            "Session_Type": "Theory", "Time_of_Day": "Morning", "Classroom": "Room 201"
        }])
        st.download_button(
            "📥 Download Sample CSV Template",
            data=sample_template.to_csv(index=False).encode('utf-8'),
            file_name="akash_attendiq_sample_template.csv",
            mime="text/csv",
            use_container_width=True
        )

    if uploaded is not None:
        try:
            b_df = pd.read_csv(uploaded)
            st.success(f"Loaded {len(b_df)} records. Generating batch predictions...")
            with st.spinner("Processing ML batch inference..."):
                res_b_df = predictor.predict_batch(b_df)
            st.dataframe(res_b_df.head(50), use_container_width=True)

            st.download_button(
                "📥 Export Full Batch Predictions to CSV",
                data=res_b_df.to_csv(index=False).encode('utf-8'),
                file_name=f"akash_attendiq_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                type="primary"
            )
        except Exception as e:
            st.error(f"Error processing uploaded CSV: {e}")

    st.divider()
    st.subheader("🚨 Historical Low-Attendance Warning Registry (<75%)")
    if not history_df.empty:
        low_att = history_df[history_df["Attendance_Pct"] < 75.0].sort_values("Attendance_Pct")
        col_la1, col_la2 = st.columns([1, 3])
        with col_la1:
            st.metric("Flagged Sessions", f"{len(low_att)} / {len(history_df)}", f"{(len(low_att)/len(history_df))*100:.1f}%")
            st.download_button(
                "📥 Export Flagged CSV",
                data=low_att.to_csv(index=False).encode('utf-8'),
                file_name="akash_attendiq_flagged_sessions.csv",
                mime="text/csv"
            )
        with col_la2:
            st.dataframe(low_att[["Date", "Day", "Subject", "Branch", "Section", "Lecture_Number", "Attendance_Pct", "Classroom"]].head(20).style.format({"Attendance_Pct": "{:.1f}%"}), use_container_width=True)

# Sidebar Footer
st.sidebar.divider()
st.sidebar.caption(f"🚀 **{PRODUCT_NAME} v{PRODUCT_VERSION}**")
st.sidebar.caption(f"{PRODUCT_TAGLINE}")
