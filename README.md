# 🎓 AKASH AttendIQ

### Smart College Attendance Prediction

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)]()

## What is AKASH AttendIQ?
AKASH AttendIQ is a simple, attractive, mobile-friendly, explainable college attendance prediction application. It allows students, faculty members, and recruiters to instantly understand attendance patterns and predict future absenteeism risks. 

The application is built on a "zero-leakage" Machine Learning principle, ensuring it only uses information genuinely available before a lecture occurs.

## What exactly does it predict?
AKASH AttendIQ predicts the **expected attendance percentage** for an upcoming college lecture session. Simultaneously, it categorizes this prediction into a **Risk Status** (`🟢 SAFE`, `🟡 WARNING`, `🔴 CRITICAL`) representing the likelihood of severe absenteeism, calculating specific confidence probabilities for each risk tier.

## Which features are used?
The model relies exclusively on information known *before* the prediction target session:
* **Historical & Recent Trends**: Recent 3-Lecture Moving Avg, Recent 7-Lecture Moving Avg, Previous Lecture Attendance, Monthly Historical Benchmark.
* **Academic Context**: Internal Test Week Pressure, Holiday Proximity Indicator.
* **Session Details**: Lecture Slot (Timing Fatigue), Classroom Enrollment Strength, Inter-Lecture Time Gap.
* **Categoricals**: Day of the Week, Academic Subject, Session Format (Theory vs Lab), Session Timing (Morning vs Afternoon).

## How is the prediction generated?
1. The user inputs their desired session details via a mobile-first Streamlit interface.
2. The input is passed to an offline, serialized Scikit-Learn `Pipeline`.
3. A **Gradient Boosting Regressor** predicts the raw percentage.
4. A **Gradient Boosting Classifier** independently predicts the exact risk category probabilities.
5. The system performs localized feature attribution by comparing the session's input values against the historical medians (neutral baseline) to understand the directional push of each feature.

## How is the model evaluated?
The model is strictly evaluated using a **Time-Aware Chronological Split (80/20)**. 
* **PAST → TRAIN**: The early weeks of the semester are used for training.
* **FUTURE → TEST**: The hold-out set represents entirely unseen *future* sessions.
This strict setup ensures zero data leakage and exactly replicates how the model will be used in reality.

## What are the actual metrics?
Based on the hold-out evaluation dataset:
* **Regression (Gradient Boosting)**:
  * MAE: **6.70%**
  * RMSE: **8.96%**
  * $R^2$ Score: **0.4660**
* **Classification (Gradient Boosting)**:
  * Accuracy: **78.89%**
  * F1-Score: **0.7769**
  * ROC-AUC: **0.8533**

## How is the prediction explained?
The **"Why this prediction?"** engine calculates the localized difference between a neutral historical baseline and the user's specific inputs. The UI clearly visualizes the top positive constraints (factors that increased attendance) and the top negative constraints (factors that lowered it). This is presented in plain English (e.g., "Early morning slot introduces timing fatigue").

## What are the limitations?
* **Individual Student Tracking**: The model predicts *aggregate cohort attendance* (e.g. 78% of the class), but cannot predict *which specific student* will be absent.
* **Unprecedented Outliers**: The model cannot accurately account for spontaneous extreme events (e.g., severe weather, sudden campus closures).
* $R^2$ **Ceiling**: Human behavior is inherently noisy. An $R^2$ of 0.46 indicates the model captures nearly half the variance based purely on operational constraints, but the remaining variance is driven by individual student decisions.

## How to run locally?

```bash
# Clone the repository
git clone https://github.com/akashupadhayay106-au/AKASH-AttendIQ.git
cd AKASH-AttendIQ

# Create environment and install requirements
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run tests
pytest

# Start the application
python -m streamlit run app.py
```

## GitHub repository
[https://github.com/akashupadhayay106-au/AKASH-AttendIQ](https://github.com/akashupadhayay106-au/AKASH-AttendIQ)

## Live application
The application is permanently deployed and updated continuously from the `main` branch via Streamlit Community Cloud. 

🔗 **[Live Streamlit URL]** *(Available upon final user connection to Streamlit Cloud)*
