# 🎓 AKASH AttendIQ

**Smart College Attendance Prediction — Ultimate Product Enhancement**

AKASH AttendIQ is a premium, production-quality Machine Learning SaaS dashboard designed to predict college attendance patterns, estimate absenteeism risk, and provide actionable institutional guidance.

---

## 🚀 The Problem
College attendance drops significantly due to factors like mid-day fatigue, holiday proximity, and previous absences. Without early warning systems, institutions struggle to identify at-risk students before their attendance falls below mandatory compliance thresholds.

## 💡 The Solution
AKASH AttendIQ solves this using a **Chronologically Validated Machine Learning Pipeline**. The system generates precise, explainable attendance estimates and automatically generates **Smart Guidance** to recommend actionable interventions based purely on historical patterns.

---

## ✨ Features

- **🎯 Prediction Studio**: Estimate expected attendance for any session using context (Subject, Time, Test Week).
- **🔮 What-If Simulator**: A/B test scenarios (e.g., changing from Morning to Afternoon) to observe dynamic risk deltas.
- **📊 Attendance Analytics**: Comprehensive EDA dashboard (Day-wise fatigue, Subject distributions, etc.).
- **💡 Smart Guidance Engine**: Contextual, natural-language recommendations derived directly from local feature attribution.
- **📁 Batch Prediction**: Enterprise CSV upload workflow for mass-evaluating institutional risk.

---

## 🧠 ML Architecture

### 1. Leakage Prevention
**Zero-Leakage Guarantee**: The entire pipeline ensures no future or target-derived information is exposed during inference. 
- All predictors (e.g., `Rolling_3_Avg`) are strictly restricted to $t-1$ or earlier.
- The pipeline utilizes **Chronological Validation**, splitting data chronologically (earliest 80% for training, latest 20% untouched for hold-out evaluation).

### 2. Model Selection
We dynamically benchmark multiple high-performance architectures (GradientBoosting, HistGradientBoosting, RandomForest, ExtraTrees).
- **Champion Regressor**: Selected via strict Chronological Hold-Out MAE & RMSE.
- **Champion Classifier**: Wraps the base estimator in a `CalibratedClassifierCV` (Sigmoid) to generate reliable, empirically calibrated **Model Risk Probabilities**.

### 3. Explainability
Integrated permutation-style baseline-difference mapping allows the application to isolate the strongest positive and negative factors influencing every single prediction.

---

## 📈 Results (Chronological Hold-Out)

- **Regression Typical Error**: ±6.7 percentage points (MAE)
- **Classification Accuracy**: ~78.5%
- **ROC-AUC**: ~0.855

*These metrics represent true unseen generalization capability on future timeline data.*

---

## 💻 Installation & Local Run

1. **Clone the repository**
```bash
git clone https://github.com/akashupadhayay106-au/AKASH-AttendIQ.git
cd AKASH-AttendIQ
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **(Optional) Re-train the Pipeline**
```bash
python train.py
```

4. **Launch the Dashboard**
```bash
python -m streamlit run app.py
```

---

## ☁️ Streamlit Deployment

This application is fully compatible with **Streamlit Community Cloud**. 
1. Connect your GitHub repository.
2. Set the entrypoint to `app.py`.
3. The platform will automatically install `requirements.txt` and serve the dashboard globally.

---

## ⚠️ Limitations & Disclaimers
- **Estimation, Not Guarantee**: The model generates a *Model-based attendance estimate* and a *Model Risk Probability*. It does not guarantee future human behavior.
- **No Individual Tracking**: The model predicts cohort/session risk based on aggregate historical and structural variables. It does not track personal student identities.
- **Context Dependent**: Predictions rely heavily on the specific operational distributions of the dataset provided.

---

*Architected for Educational Intelligence.*
