"""
AKASH AttendIQ — Prediction, Simulation & Explainable Inference Engine
Provides single-session forecasting with local feature attribution, scenario simulation, and batch CSV processing.
Includes automated self-healing pipeline generation for cloud deployment resilience.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional

from src.config import (
    MODEL_FILE,
    ROOT_MODEL_FILE,
    RAW_DATA_FILE,
    CLEANED_DATA_FILE,
    ALL_MODEL_FEATURES,
    ATTENDANCE_BANDS,
    HUMAN_FEATURE_NAMES,
    THEME_COLORS
)
from src.explainability import ExplainabilityEngine


class AttendancePredictor:
    """Production Inference, Simulation, and Explainability Engine for AKASH AttendIQ."""

    def __init__(self, model_path: str = MODEL_FILE):
        self.model_path = model_path
        self.package = None
        self.reg_model = None
        self.cls_model = None
        self.band_labels = ATTENDANCE_BANDS
        self.baseline_medians = {}
        self.metadata = {}
        self.explainability = None
        self._load()

    def _load(self):
        """Loads serialized pipeline package with automated self-healing fallback."""
        path_to_load = self.model_path
        loaded_successfully = False

        if os.path.exists(path_to_load):
            try:
                self.package = joblib.load(path_to_load)
                self.reg_model = self.package["regression_model"]
                self.cls_model = self.package["classification_model"]
                loaded_successfully = True
            except Exception as load_err:
                print(f"[WARN] Pickled model incompatible with current environment ({load_err}). Triggering self-healing training...")

        if not loaded_successfully and os.path.exists(ROOT_MODEL_FILE):
            try:
                self.package = joblib.load(ROOT_MODEL_FILE)
                self.reg_model = self.package["regression_model"]
                self.cls_model = self.package["classification_model"]
                loaded_successfully = True
            except Exception as load_err:
                print(f"[WARN] Root pickled model incompatible ({load_err}). Triggering self-healing training...")

        # Self-Healing: If model is missing or cannot be unpickled in cloud container, train on-the-fly
        if not loaded_successfully:
            print("[INFO] Initiating self-healing pipeline training...")
            from src.data_processor import process_and_save_data
            from src.model_trainer import train_complete_system

            df_featured = process_and_save_data(RAW_DATA_FILE, CLEANED_DATA_FILE)
            self.package, _, _ = train_complete_system(df_featured, self.model_path)
            self.reg_model = self.package["regression_model"]
            self.cls_model = self.package["classification_model"]
            print("[SUCCESS] Self-healing training completed successfully!")

        self.band_labels = self.package.get("band_labels", ATTENDANCE_BANDS)
        self.baseline_medians = self.package.get("baseline_medians", {})
        self.metadata = self.package.get("metadata", {})
        
        residual_std = self.package.get("residual_std", 8.54)
        feat_df = pd.DataFrame(self.metadata.get("feature_importance", []))
        self.explainability = ExplainabilityEngine(feature_importance_df=feat_df, residual_std=residual_std)

    def predict_single(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes explainable single-session inference.
        Returns:
            - predicted_attendance: float (0.0 to 100.0)
            - estimated_range: str (e.g. '78.2% – 86.5%')
            - risk_band: str ('SAFE', 'WARNING', 'CRITICAL')
            - probabilities: Dict[str, float] (e.g. {'SAFE': 78.0, 'WARNING': 17.0, 'CRITICAL': 5.0})
            - confidence_pct: float
            - expected_students: int
            - total_enrolled: int
            - top_positive_factors: List[Dict]
            - top_negative_factors: List[Dict]
            - action_advisory: Dict
        """
        df_input = pd.DataFrame([input_data])
        # Ensure all columns present
        for col in ALL_MODEL_FEATURES:
            if col not in df_input.columns:
                df_input[col] = np.nan

        # 1. Regression Prediction
        pred_reg = float(np.clip(self.reg_model.predict(df_input[ALL_MODEL_FEATURES])[0], 0.0, 100.0))

        # 2. Classification Probabilities & Band
        probs = self.cls_model.predict_proba(df_input[ALL_MODEL_FEATURES])[0]
        pred_code = int(np.argmax(probs))
        pred_band = self.band_labels[pred_code] if pred_code < len(self.band_labels) else "SAFE"
        
        prob_dict = {
            self.band_labels[i]: round(float(probs[i]) * 100.0, 1)
            for i in range(min(len(self.band_labels), len(probs)))
        }
        confidence = float(np.max(probs) * 100.0)

        # 3. Headcount calculations
        total_enrolled = int(input_data.get("Total_Enrolled", 100))
        expected_students = int(round(total_enrolled * pred_reg / 100.0))
        expected_absent = max(0, total_enrolled - expected_students)

        # 4. Explainability & Feature Attributions
        explanation = self.explainability.explain_local_prediction(
            input_data=input_data,
            baseline_medians=self.baseline_medians,
            predicted_attendance=pred_reg
        )

        # 5. Institutional Action Advisory
        advisory = self.explainability.generate_action_advisory(
            risk_band=pred_band,
            predicted_attendance=pred_reg,
            negative_factors=explanation["top_negative_factors"]
        )

        # Risk color
        if pred_band == "CRITICAL":
            risk_color = THEME_COLORS["critical"]
        elif pred_band == "WARNING":
            risk_color = THEME_COLORS["warning"]
        else:
            risk_color = THEME_COLORS["safe"]

        return {
            "predicted_attendance": pred_reg,
            "estimated_range": explanation["estimated_range"],
            "lower_bound": explanation["lower_bound"],
            "upper_bound": explanation["upper_bound"],
            "margin_of_error": explanation["margin_of_error"],
            "risk_band": pred_band,
            "risk_color": risk_color,
            "probabilities": prob_dict,
            "confidence_pct": round(confidence, 1),
            "expected_students": expected_students,
            "expected_absent": expected_absent,
            "total_enrolled": total_enrolled,
            "top_positive_factors": explanation["top_positive_factors"],
            "top_negative_factors": explanation["top_negative_factors"],
            "advisory": advisory
        }

    def simulate_what_if(self, current_input: Dict[str, Any], scenario_overrides: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compares Baseline prediction vs What-If Scenario prediction, quantifying the exact delta.
        """
        baseline_res = self.predict_single(current_input)
        
        modified_input = current_input.copy()
        modified_input.update(scenario_overrides)
        scenario_res = self.predict_single(modified_input)

        diff_pct = round(scenario_res["predicted_attendance"] - baseline_res["predicted_attendance"], 2)
        diff_students = scenario_res["expected_students"] - baseline_res["expected_students"]

        # Identify which changed features drove the difference
        changed_factors = []
        for k, v in scenario_overrides.items():
            human_name = HUMAN_FEATURE_NAMES.get(k, k.replace("_", " ").title())
            orig_val = current_input.get(k, "N/A")
            changed_factors.append(f"{human_name}: '{orig_val}' → '{v}'")

        return {
            "baseline": baseline_res,
            "scenario": scenario_res,
            "diff_attendance_pct": diff_pct,
            "diff_students": diff_students,
            "risk_changed": baseline_res["risk_band"] != scenario_res["risk_band"],
            "changed_factors": changed_factors
        }

    def predict_batch(self, df_batch: pd.DataFrame) -> pd.DataFrame:
        """
        Processes a DataFrame of sessions in bulk, appending predictions, confidence, and risk flags.
        """
        df_out = df_batch.copy()
        for col in ALL_MODEL_FEATURES:
            if col not in df_out.columns:
                df_out[col] = np.nan

        preds_reg = np.clip(self.reg_model.predict(df_out[ALL_MODEL_FEATURES]), 0.0, 100.0)
        probs = self.cls_model.predict_proba(df_out[ALL_MODEL_FEATURES])
        pred_codes = np.argmax(probs, axis=1)

        df_out["Predicted_Attendance_Pct"] = np.round(preds_reg, 2)
        df_out["Predicted_Risk_Band"] = [self.band_labels[c] if c < len(self.band_labels) else "SAFE" for c in pred_codes]
        df_out["Confidence_Pct"] = np.round(np.max(probs, axis=1) * 100.0, 1)

        # Probabilities
        for i, band in enumerate(self.band_labels):
            if i < probs.shape[1]:
                df_out[f"Prob_{band}_Pct"] = np.round(probs[:, i] * 100.0, 1)

        if "Total_Enrolled" in df_out.columns:
            df_out["Expected_Students_Present"] = np.round(
                df_out["Total_Enrolled"] * df_out["Predicted_Attendance_Pct"] / 100.0
            ).astype(int)

        # Estimated Range
        margin = 1.96 * self.package.get("residual_std", 8.54)
        df_out["Estimated_Range"] = [
            f"{max(0.0, round(p - margin, 1))}% – {min(100.0, round(p + margin, 1))}%"
            for p in preds_reg
        ]

        return df_out
