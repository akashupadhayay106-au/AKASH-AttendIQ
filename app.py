"""
🎓 AKASH AttendIQ — Smart College Attendance Prediction
Premium Desktop-First Machine Learning Analytics Dashboard
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
    CLEANED_DATA_FILE,
    SLOT_TIMINGS,
    ATTENDANCE_BANDS,
    HUMAN_FEATURE_NAMES,
    THEME_COLORS,
    ALL_MODEL_FEATURES
)
from src.predictor import AttendancePredictor

# ==============================================================================
# 1. PAGE CONFIG & PREMIUM CSS
# ==============================================================================

st.set_page_config(
    page_title=f"{PRODUCT_NAME} — {PRODUCT_TAGLINE}",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sophisticated Dashboard CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif;
    }

    /* Premium Header */
    .dashboard-header {
        background: linear-gradient(90deg, #1E1B4B 0%, #0F172A 100%);
        padding: 24px 32px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .dashboard-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #818CF8, #38BDF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 8px 0;
    }
    .dashboard-subtitle {
        color: #94A3B8;
        font-size: 1.1rem;
        margin: 0;
    }

    /* KPI Cards */
    .kpi-card {
        background: #1E293B;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.5);
    }
    .kpi-label {
        color: #94A3B8;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    .kpi-value {
        color: #F8FAFC;
        font-size: 1.8rem;
        font-weight: 700;
        font-family: 'Outfit', sans-serif;
    }

    /* Prediction Result Card */
    .pred-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        padding: 32px;
        border-radius: 16px;
        text-align: center;
        border: 1px solid rgba(99, 102, 241, 0.3);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
    }
    
    .status-safe { color: #10B981; }
    .status-warning { color: #F59E0B; }
    .status-critical { color: #EF4444; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. STATE & DATA CACHING
# ==============================================================================

@st.cache_resource(show_spinner="Loading ML Pipeline...")
def load_predictor():
    return AttendancePredictor()

@st.cache_data(show_spinner="Loading Analytical Data...")
def load_analytical_data():
    if os.path.exists(CLEANED_DATA_FILE):
        return pd.read_csv(CLEANED_DATA_FILE)
    return pd.DataFrame()

predictor = load_predictor()
df_analytics = load_analytical_data()

# ==============================================================================
# 3. SIDEBAR NAVIGATION
# ==============================================================================

with st.sidebar:
    st.markdown(f"<h2 style='font-family: Outfit; font-weight: 800; background: linear-gradient(90deg, #818CF8, #38BDF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>🎓 {PRODUCT_NAME}</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: #94A3B8; font-size: 0.9rem; margin-top: -15px;'>{PRODUCT_TAGLINE}</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    page = st.radio("Navigation", [
        "🏠 Overview Dashboard",
        "🎯 Prediction Studio",
        "🔮 What-If Simulator",
        "📊 Attendance Analytics",
        "🤖 Model Intelligence",
        "📁 Batch Prediction"
    ])
    
    st.markdown("---")
    st.markdown("### Model Status")
    st.success("● Pipeline Active")
    if predictor.package:
        metrics = predictor.metadata.get("metrics", {})
        acc = metrics.get("classification", {}).get("Accuracy", 0)
        st.caption(f"**Engine:** Gradient Boosting")
        st.caption(f"**Version:** Production Ready")
        st.caption(f"**Global Accuracy:** {acc*100:.1f}%")

# Helper for Header
def render_header(title, subtitle):
    st.markdown(f"""
    <div class="dashboard-header">
        <h1 class="dashboard-title">{title}</h1>
        <p class="dashboard-subtitle">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

# Helper for KPI
def render_kpi(label, value, color_hex="#F8FAFC"):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color: {color_hex}">{value}</div>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# 4. OVERVIEW DASHBOARD
# ==============================================================================
if page == "🏠 Overview Dashboard":
    render_header("Attendance Intelligence", "Monitor attendance patterns, predict upcoming session attendance, and identify risk signals.")
    
    if not df_analytics.empty:
        # Top KPI Row
        k1, k2, k3, k4, k5 = st.columns(5)
        
        metrics = predictor.metadata.get("metrics", {})
        cls_metrics = metrics.get("classification", {})
        reg_metrics = metrics.get("regression", {})
        
        with k1: render_kpi("📚 Total Sessions", f"{len(df_analytics):,}")
        with k2: render_kpi("📈 Avg Attendance", f"{df_analytics['Attendance_Pct'].mean():.1f}%", THEME_COLORS["primary"])
        with k3: render_kpi("🎯 Model Accuracy", f"{cls_metrics.get('Accuracy', 0.7889)*100:.1f}%", THEME_COLORS["safe"])
        with k4: render_kpi("⭐ ROC-AUC", f"{cls_metrics.get('ROC_AUC', 0.853):.3f}", THEME_COLORS["secondary"])
        with k5: render_kpi("📊 Prediction MAE", f"{reg_metrics.get('MAE', 6.7)}%", THEME_COLORS["warning"])
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Chart Grid
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.markdown("### 📈 Attendance Trend (Moving Average)")
            if "Day_of_Semester" in df_analytics.columns:
                trend_df = df_analytics.groupby("Day_of_Semester")["Attendance_Pct"].mean().reset_index()
                trend_df["Rolling"] = trend_df["Attendance_Pct"].rolling(7, min_periods=1).mean()
                
                fig = px.line(trend_df, x="Day_of_Semester", y="Rolling", 
                              color_discrete_sequence=[THEME_COLORS["primary"]])
                fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                  margin=dict(l=0, r=0, t=20, b=0), xaxis_title="Semester Progression", yaxis_title="Attendance %")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Insight: Displays the overall 7-day smoothed attendance pattern across the semester timeline.")
                
        with c2:
            st.markdown("### 🎯 Risk Distribution")
            if "Attendance_Risk" in df_analytics.columns:
                risk_counts = df_analytics["Attendance_Risk"].value_counts().reset_index()
                risk_counts.columns = ["Risk", "Count"]
                color_map = {"SAFE": THEME_COLORS["safe"], "WARNING": THEME_COLORS["warning"], "CRITICAL": THEME_COLORS["critical"]}
                
                fig2 = px.pie(risk_counts, values="Count", names="Risk", hole=0.6,
                              color="Risk", color_discrete_map=color_map)
                fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                   margin=dict(l=0, r=0, t=20, b=0), showlegend=False)
                fig2.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig2, use_container_width=True)
                st.caption("Insight: Actual historical distribution of safe vs at-risk lecture sessions.")

        st.markdown("---")
        st.markdown("### ⚡ Quick Prediction")
        st.caption("Experience the ML inference engine instantly with common inputs.")
        
        qc1, qc2, qc3, qc4 = st.columns(4)
        q_prev = qc1.slider("Previous Attendance %", 0, 100, 75, key="q1")
        q_rec = qc2.slider("Recent 3-Lec Avg", 0, 100, 78, key="q2")
        q_day = qc3.selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"], key="q3")
        q_slot = qc4.selectbox("Timing", list(SLOT_TIMINGS.values()), index=2, key="q4")
        
        if st.button("Predict Attendance →", use_container_width=True, type="primary"):
            slot_id = [k for k, v in SLOT_TIMINGS.items() if v == q_slot][0]
            base_input = predictor.baseline_medians.copy() if predictor.baseline_medians else {}
            base_input.update({
                "Prev_Lecture_Pct": q_prev,
                "Rolling_3_Avg": q_rec,
                "Day": q_day,
                "Lecture_Number": slot_id
            })
            
            res = predictor.predict_single(base_input)
            
            st.markdown(f"""
            <div style="background: #1E293B; padding: 20px; border-radius: 12px; border-left: 4px solid {THEME_COLORS['primary']}; margin-top: 10px;">
                <h3 style="margin: 0; color: #F8FAFC;">Expected: {res['predicted_attendance']:.1f}%</h3>
                <p style="margin: 5px 0 0 0; color: #94A3B8;">Risk Status: <strong class="status-{res['risk_band'].lower()}">{res['risk_band']}</strong> | Probability: {res['confidence_pct']:.1f}%</p>
                <p style="margin: 10px 0 0 0; color: #CBD5E1; font-style: italic;">💡 {res.get('guidance', '')}</p>
            </div>
            """, unsafe_allow_html=True)
            
    else:
        st.warning("Analytical dataset not found. Model inference is still available.")

# ==============================================================================
# 5. PREDICTION STUDIO
# ==============================================================================
elif page == "🎯 Prediction Studio":
    render_header("Prediction Studio", "Estimate expected attendance for an upcoming class using historical and session-level patterns.")
    
    c1, c2 = st.columns([1.2, 1])
    
    with c1:
        st.markdown("### Input Features")
        with st.container(border=True):
            st.markdown("#### 📅 Session Context")
            col_a, col_b = st.columns(2)
            day_val = col_a.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"])
            slot_name = col_b.selectbox("Lecture Timing", list(SLOT_TIMINGS.values()))
            slot_val = [k for k, v in SLOT_TIMINGS.items() if v == slot_name][0]
            subj_val = st.selectbox("Subject", df_analytics["Subject"].unique() if not df_analytics.empty else ["Database Management", "Data Structures", "Operating Systems"])
            
            st.markdown("#### 📈 Attendance History")
            col_c, col_d = st.columns(2)
            prev_val = col_c.number_input("Previous Lecture Attendance %", 0, 100, 75)
            roll3_val = col_d.number_input("Recent 3-Lecture Avg %", 0, 100, 78)
            roll7_val = col_c.number_input("Recent 7-Lecture Avg %", 0, 100, 76)
            
            st.markdown("#### 🎓 Academic Context")
            test_val = st.toggle("Internal Test Week", False)
            hol_val = st.toggle("Holiday Proximity", False)
            
        st.markdown("---")
        predict_btn = st.button("🎯 Generate Full Prediction", use_container_width=True, type="primary")

    with c2:
        if predict_btn:
            # Build input dictionary padding with baselines
            inp = predictor.baseline_medians.copy() if predictor.baseline_medians else {}
            inp.update({
                "Day": day_val,
                "Lecture_Number": slot_val,
                "Subject": subj_val,
                "Prev_Lecture_Pct": prev_val,
                "Rolling_3_Avg": roll3_val,
                "Rolling_7_Avg": roll7_val,
                "Test_Week": int(test_val),
                "Holiday_Near": int(hol_val)
            })
            
            res = predictor.predict_single(inp)
            
            status_color = THEME_COLORS.get(res['risk_band'].lower(), "#fff")
            
            st.markdown(f"""
            <div class="pred-card">
                <p style="color: #94A3B8; font-weight: 600; text-transform: uppercase; margin: 0;">Predicted Attendance</p>
                <h1 style="font-size: 4rem; color: #F8FAFC; margin: 10px 0; font-family: Outfit;">{res['predicted_attendance']:.1f}%</h1>
                <h2 class="status-{res['risk_band'].lower()}" style="margin: 0; font-weight: 800;">{res['risk_band']}</h2>
                <p style="color: #94A3B8; font-size: 0.9rem; margin-top: 15px;">Typical prediction error: ±6.7 percentage points</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Gauge Chart
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=res['predicted_attendance'],
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': status_color},
                    'bgcolor': "#1E293B",
                    'borderwidth': 2,
                    'bordercolor': "#334155",
                    'steps': [
                        {'range': [0, 50], 'color': "rgba(239, 68, 68, 0.2)"},
                        {'range': [50, 75], 'color': "rgba(245, 158, 11, 0.2)"},
                        {'range': [75, 100], 'color': "rgba(16, 185, 129, 0.2)"}
                    ]
                }
            ))
            fig_g.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "#F8FAFC"})
            st.plotly_chart(fig_g, use_container_width=True)
            
            # Probabilities
            st.markdown("### Model Risk Probability")
            probs = res['probabilities']
            for b_name in ["SAFE", "WARNING", "CRITICAL"]:
                val = probs.get(b_name, 0.0)
                col_c = THEME_COLORS.get(b_name.lower())
                st.markdown(f"**{b_name}**: &nbsp;&nbsp; {val:.1f}%")
                st.progress(val / 100.0)
                
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 💡 Smart Guidance")
            st.info(res.get('guidance', ''))
                
    # Explainability Section
    if predict_btn:
        st.markdown("---")
        st.markdown("## 🔍 Why This Prediction?")
        
        e1, e2 = st.columns([1.5, 1])
        
        with e1:
            st.markdown("### Feature Impact")
            impacts = []
            for f in res['top_positive_factors']:
                impacts.append({"Feature": HUMAN_FEATURE_NAMES.get(f['feature'], f['feature']), "Impact": f['impact'], "Type": "Positive"})
            for f in res['top_negative_factors']:
                impacts.append({"Feature": HUMAN_FEATURE_NAMES.get(f['feature'], f['feature']), "Impact": f['impact'], "Type": "Negative"})
                
            if impacts:
                df_imp = pd.DataFrame(impacts)
                df_imp = df_imp.sort_values("Impact", ascending=True)
                
                fig_imp = px.bar(df_imp, x="Impact", y="Feature", color="Type", orientation='h',
                                 color_discrete_map={"Positive": THEME_COLORS["safe"], "Negative": THEME_COLORS["critical"]})
                fig_imp.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                      margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
                st.plotly_chart(fig_imp, use_container_width=True)
        
        with e2:
            st.markdown("### Interpretation")
            if res['top_positive_factors']:
                top_p = res['top_positive_factors'][0]
                fname = HUMAN_FEATURE_NAMES.get(top_p['feature'], top_p['feature'])
                st.success(f"**Strongest Positive Factor:**\n\n{fname} ({top_p['impact']:+.1f}%) is strongly supporting attendance.")
            
            if res['top_negative_factors']:
                top_n = res['top_negative_factors'][0]
                fname = HUMAN_FEATURE_NAMES.get(top_n['feature'], top_n['feature'])
                st.error(f"**Strongest Negative Factor:**\n\n{fname} ({top_n['impact']:+.1f}%) is reducing the expected attendance.")
            
            st.info("**What this means:**\nThe model estimates attendance based strictly on the provided historical and session information compared to the institutional average. It does not track individual students.")

# ==============================================================================
# 6. WHAT-IF SIMULATOR
# ==============================================================================
elif page == "🔮 What-If Simulator":
    render_header("What-If Scenario Lab", "Compare a baseline prediction with an alternative scenario to see exactly how operational changes influence attendance.")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("### 1. Set Current Scenario")
        with st.container(border=True):
            b_prev = st.slider("Current Previous Attendance", 0, 100, 75)
            b_rec = st.slider("Current Recent Avg", 0, 100, 75)
            b_time = st.selectbox("Current Timing", list(SLOT_TIMINGS.values()), index=2)
            b_slot_val = [k for k, v in SLOT_TIMINGS.items() if v == b_time][0]
            
    with c2:
        st.markdown("### 2. Set Modified Scenario")
        with st.container(border=True):
            s_prev = st.slider("Scenario Previous Attendance", 0, 100, 85)
            s_rec = st.slider("Scenario Recent Avg", 0, 100, 85)
            s_time = st.selectbox("Scenario Timing", list(SLOT_TIMINGS.values()), index=0)
            s_slot_val = [k for k, v in SLOT_TIMINGS.items() if v == s_time][0]
            
    if st.button("⚡ Simulate Impact", type="primary", use_container_width=True):
        b_input = predictor.baseline_medians.copy()
        b_input.update({"Prev_Lecture_Pct": b_prev, "Rolling_3_Avg": b_rec, "Lecture_Number": b_slot_val})
        
        s_input = b_input.copy()
        s_input.update({"Prev_Lecture_Pct": s_prev, "Rolling_3_Avg": s_rec, "Lecture_Number": s_slot_val})
        
        res = predictor.simulate_what_if(b_input, s_input)
        
        st.markdown("---")
        st.markdown("### Simulation Result")
        
        r1, r2, r3 = st.columns(3)
        
        b_val = res['baseline_prediction']
        s_val = res['scenario_prediction']
        delta = res['delta_pct']
        
        d_color = THEME_COLORS['safe'] if delta > 0 else THEME_COLORS['critical'] if delta < 0 else THEME_COLORS['text_muted']
        
        with r1: render_kpi("CURRENT", f"{b_val:.1f}%")
        with r2: render_kpi("SCENARIO", f"{s_val:.1f}%")
        with r3: render_kpi("IMPACT", f"{delta:+.1f}%", d_color)
        
        # Plotly comparison
        fig = go.Figure()
        fig.add_trace(go.Bar(x=['Current', 'Scenario'], y=[b_val, s_val], 
                             marker_color=[THEME_COLORS['background_card'], THEME_COLORS['primary']],
                             text=[f"{b_val:.1f}%", f"{s_val:.1f}%"], textposition='auto'))
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", yaxis_range=[0, 100], height=300)
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# 7. ATTENDANCE ANALYTICS
# ==============================================================================
elif page == "📊 Attendance Analytics":
    render_header("Attendance Analytics", "Deep institutional exploratory data analysis across historical records.")
    
    if df_analytics.empty:
        st.warning("No analytics dataset found.")
    else:
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("### Attendance Distribution")
            fig = px.histogram(df_analytics, x="Attendance_Pct", nbins=40, color_discrete_sequence=[THEME_COLORS['primary']])
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Insight: Displays the overall historical frequency of attendance rates, revealing institutional baselines.")
            
            st.markdown("### Day-wise Attendance")
            if "Day" in df_analytics.columns:
                day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
                fig3 = px.box(df_analytics, x="Day", y="Attendance_Pct", category_orders={"Day": day_order}, color_discrete_sequence=[THEME_COLORS['secondary']])
                fig3.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig3, use_container_width=True)
                st.caption("Insight: Weekend proximity generally causes wider variance in Friday/Saturday attendance.")
                
        with c2:
            st.markdown("### Lecture Slot Analysis (Fatigue)")
            if "Lecture_Number" in df_analytics.columns:
                slot_avg = df_analytics.groupby("Lecture_Number")["Attendance_Pct"].mean().reset_index()
                slot_avg["Timing"] = slot_avg["Lecture_Number"].map(SLOT_TIMINGS)
                fig2 = px.bar(slot_avg, x="Timing", y="Attendance_Pct", color_discrete_sequence=[THEME_COLORS['safe']])
                fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", yaxis_range=[50, 100], margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig2, use_container_width=True)
                st.caption("Insight: Mid-day sessions typically hold maximum momentum, while early morning and late afternoon suffer from slot fatigue.")

            st.markdown("### Subject Performance")
            if "Subject" in df_analytics.columns:
                subj_avg = df_analytics.groupby("Subject")["Attendance_Pct"].mean().sort_values(ascending=False).reset_index()
                fig4 = px.bar(subj_avg, x="Attendance_Pct", y="Subject", orientation='h', color_discrete_sequence=[THEME_COLORS['warning']])
                fig4.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", xaxis_range=[50, 100], margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig4, use_container_width=True)
                st.caption("Insight: Shows inherent subject interest and mandatory vs elective attendance disparities.")

# ==============================================================================
# 8. MODEL INTELLIGENCE
# ==============================================================================
elif page == "🤖 Model Intelligence":
    render_header("Model Intelligence", "Transparency into model metrics, feature importance, and classification boundaries.")
    
    metrics = predictor.metadata.get("metrics", {})
    cls_metrics = metrics.get("classification", {})
    reg_metrics = metrics.get("regression", {})
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("### 🏆 Final Regression Model")
        st.info("**Gradient Boosting Regressor**")
        rm1, rm2, rm3 = st.columns(3)
        with rm1: render_kpi("MAE", f"{reg_metrics.get('MAE', 6.7)}%")
        with rm2: render_kpi("RMSE", f"{reg_metrics.get('RMSE', 8.9)}%")
        with rm3: render_kpi("R²", f"{reg_metrics.get('R2', 0.46):.3f}")
        
    with c2:
        st.markdown("### 🏆 Final Classification Model")
        st.info("**Gradient Boosting Classifier**")
        cm1, cm2, cm3 = st.columns(3)
        with cm1: render_kpi("Accuracy", f"{cls_metrics.get('Accuracy', 0.78)*100:.1f}%")
        with cm2: render_kpi("F1-Score", f"{cls_metrics.get('F1_Score', 0.77):.3f}")
        with cm3: render_kpi("ROC-AUC", f"{cls_metrics.get('ROC_AUC', 0.85):.3f}")
        
    st.markdown("---")
    
    st.markdown("### 🔐 Data Leakage Audit")
    st.markdown("""
    - ✅ **Historical Features Only**: All predictors are restricted to $t-1$ or earlier.
    - ✅ **Chronological Validation**: The pipeline splits data chronologically (80% past / 20% future).
    - ✅ **Future Sessions Excluded**: No forward-peeking moving averages are permitted.
    - ✅ **Unseen Test Set**: The test set remains completely unseen during the hyperparameter tuning and model selection phases.
    """)
    
    st.markdown("---")
    
    e1, e2 = st.columns([1.5, 1])
    
    with e1:
        st.markdown("### 🌍 Global Feature Importance")
        st.caption("What generally matters most to the model across the entire dataset?")
        feat_imp = predictor.metadata.get("feature_importance", [])
        if feat_imp:
            df_imp = pd.DataFrame(feat_imp).sort_values("Importance", ascending=True).tail(10)
            df_imp["Human_Name"] = df_imp["Feature"].map(lambda x: HUMAN_FEATURE_NAMES.get(x, x))
            fig = px.bar(df_imp, x="Importance", y="Human_Name", orientation='h', color_discrete_sequence=[THEME_COLORS['primary']])
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
    with e2:
        st.markdown("### 🧪 Confusion Matrix")
        st.caption("Real classification performance on hold-out test set.")
        cm = cls_metrics.get("Confusion_Matrix")
        if cm is not None:
            import plotly.figure_factory as ff
            z = cm
            x = ["CRITICAL", "SAFE", "WARNING"]
            y = ["CRITICAL", "SAFE", "WARNING"]
            fig_cm = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Blues')
            fig_cm.update_layout(margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.info("Confusion matrix data unavailable in package.")

# ==============================================================================
# 9. BATCH PREDICTION
# ==============================================================================
elif page == "📁 Batch Prediction":
    render_header("Batch Prediction", "Process multiple sessions from a CSV file to identify institutional risk instantly.")
    
    st.markdown("Upload a CSV with session features. The system will predict attendance and risk for every row.")
    
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    if uploaded_file is not None:
        with st.spinner("Processing batch..."):
            try:
                df_batch = pd.read_csv(uploaded_file)
                st.success(f"Successfully loaded {len(df_batch)} rows.")
                
                # We expect the CSV to have the necessary features
                # Missing features will trigger imputation or errors in pipeline, so wrap carefully
                results = []
                for _, row in df_batch.iterrows():
                    inp = row.to_dict()
                    try:
                        res = predictor.predict_single(inp)
                        results.append({
                            "Expected_Attendance_%": round(res['predicted_attendance'], 1),
                            "Risk_Level": res['risk_band']
                        })
                    except Exception as e:
                        results.append({
                            "Expected_Attendance_%": np.nan,
                            "Risk_Level": "ERROR"
                        })
                
                df_res = pd.DataFrame(results)
                df_final = pd.concat([df_batch, df_res], axis=1)
                
                st.markdown("### Batch Results")
                st.dataframe(df_final.head(50), use_container_width=True)
                
                # Dashboard
                st.markdown("### Batch Risk Summary")
                bc1, bc2, bc3 = st.columns(3)
                sc = len(df_res[df_res["Risk_Level"] == "SAFE"])
                wc = len(df_res[df_res["Risk_Level"] == "WARNING"])
                cc = len(df_res[df_res["Risk_Level"] == "CRITICAL"])
                
                with bc1: render_kpi("SAFE", sc, THEME_COLORS['safe'])
                with bc2: render_kpi("WARNING", wc, THEME_COLORS['warning'])
                with bc3: render_kpi("CRITICAL", cc, THEME_COLORS['critical'])
                
                csv_out = df_final.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Annotated CSV",
                    data=csv_out,
                    file_name="akash_batch_predictions.csv",
                    mime="text/csv",
                    type="primary"
                )
                
            except Exception as e:
                st.error(f"Batch processing failed: {str(e)}")

