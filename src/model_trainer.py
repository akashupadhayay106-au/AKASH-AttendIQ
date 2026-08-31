"""
AKASH AttendIQ — Model Training, Benchmarking, Tuning & Serialization Pipeline
Evaluates candidate regressors & classifiers, tunes hyper-parameters, and packages champion models.
"""

import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.config import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    ALL_MODEL_FEATURES,
    TARGET_REGRESSION,
    TARGET_CLASSIFICATION_CODE,
    ATTENDANCE_BANDS,
    MODEL_FILE,
    ROOT_MODEL_FILE,
    HUMAN_FEATURE_NAMES
)
from src.evaluation import calculate_regression_metrics, calculate_classification_metrics


def get_preprocessor(
    num_features=NUMERICAL_FEATURES,
    cat_features=CATEGORICAL_FEATURES
) -> ColumnTransformer:
    """Constructs the ColumnTransformer preprocessing pipeline."""
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_features),
            ("cat", cat_pipeline, cat_features)
        ],
        remainder="drop"
    )


def benchmark_regression_models(X_train, y_train, X_test, y_test, preprocessor):
    """Trains and benchmarks multiple regression architectures."""
    candidates = {
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.06,
            max_depth=5,
            min_samples_split=4,
            random_state=42
        ),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200,
            max_depth=14,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        ),
        "Extra Trees Regressor": ExtraTreesRegressor(
            n_estimators=200,
            max_depth=14,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        ),
        "Ridge Regression": Ridge(alpha=1.0)
    }

    results = []
    pipelines = {}

    for name, model in candidates.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        pipe.fit(X_train, y_train)
        preds = np.clip(pipe.predict(X_test), 0.0, 100.0)

        metrics = calculate_regression_metrics(y_test, preds)
        metrics["Algorithm"] = name
        pipelines[name] = pipe
        results.append(metrics)

    df_results = pd.DataFrame(results).sort_values("RMSE (%)").reset_index(drop=True)
    best_name = df_results.iloc[0]["Algorithm"]
    best_pipeline = pipelines[best_name]

    return df_results, best_pipeline, best_name, pipelines


def benchmark_classification_models(X_train, y_train, X_test, y_test, preprocessor):
    """Trains and benchmarks classification architectures with balanced class handling."""
    candidates = {
        "Gradient Boosting Classifier": GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.07,
            max_depth=4,
            random_state=42
        ),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "Extra Trees Classifier": ExtraTreesClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "Support Vector Machine": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )
    }

    results = []
    pipelines = {}

    for name, model in candidates.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        probs = pipe.predict_proba(X_test)

        metrics = calculate_classification_metrics(y_test, preds, probs, labels=ATTENDANCE_BANDS)
        summary = {
            "Classifier": name,
            "Accuracy (%)": metrics["Accuracy (%)"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1-Score": metrics["F1-Score"],
            "F1-Macro": metrics["F1-Macro"],
            "ROC-AUC": metrics["ROC-AUC"],
            "Recall_CRITICAL": metrics.get("Recall_CRITICAL", 0.0),
            "Recall_WARNING": metrics.get("Recall_WARNING", 0.0),
            "Recall_SAFE": metrics.get("Recall_SAFE", 0.0)
        }
        pipelines[name] = pipe
        results.append(summary)

    df_results = pd.DataFrame(results).sort_values("F1-Score", ascending=False).reset_index(drop=True)
    best_name = df_results.iloc[0]["Classifier"]
    best_pipeline = pipelines[best_name]

    return df_results, best_pipeline, best_name, pipelines


def extract_feature_importance_table(trained_model, preprocessor):
    """Extracts tree feature importances mapped to human-readable names."""
    try:
        model = trained_model.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
            encoded_cat = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
            all_feature_names = NUMERICAL_FEATURES + encoded_cat

            importances = model.feature_importances_
            feat_df = pd.DataFrame({
                "Feature_Raw": all_feature_names[:len(importances)],
                "Importance": importances
            })
            feat_df["Feature"] = feat_df["Feature_Raw"].map(
                lambda x: HUMAN_FEATURE_NAMES.get(x, x.replace("_", " ").title())
            )
            return feat_df.sort_values("Importance", ascending=False).reset_index(drop=True)
    except Exception:
        pass
    return pd.DataFrame(columns=["Feature_Raw", "Feature", "Importance"])


def train_complete_system(df: pd.DataFrame, output_path: str = MODEL_FILE):
    """
    Executes full ML lifecycle:
    1. Chronological & Stratified Train/Test split
    2. ColumnTransformer preprocessing
    3. Multi-model regression benchmarking
    4. Multi-model classification benchmarking
    5. Hyperparameter tuning & diagnostics
    6. K-Means profile clustering
    7. Packaging & model persistence
    """
    X = df[ALL_MODEL_FEATURES].copy()
    y_reg = df[TARGET_REGRESSION].copy()
    y_cls = df[TARGET_CLASSIFICATION_CODE].copy()

    # Time-aware Train/Test split: 80% chronological training, 20% future evaluation
    split_idx = int(len(df) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train_reg, y_test_reg = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]
    y_train_cls, y_test_cls = y_cls.iloc[:split_idx], y_cls.iloc[split_idx:]

    preprocessor = get_preprocessor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES)

    # 1. Benchmark Regressors
    df_reg, best_reg_model, best_reg_name, _ = benchmark_regression_models(
        X_train, y_train_reg, X_test, y_test_reg, preprocessor
    )

    # 2. Benchmark Classifiers
    df_cls, best_cls_model, best_cls_name, _ = benchmark_classification_models(
        X_train, y_train_cls, X_test, y_test_cls, preprocessor
    )

    # 3. Diagnostics & Detailed Test Metrics
    reg_test_preds = np.clip(best_reg_model.predict(X_test), 0.0, 100.0)
    reg_metrics = calculate_regression_metrics(y_test_reg, reg_test_preds)

    cls_test_preds = best_cls_model.predict(X_test)
    cls_test_probs = best_cls_model.predict_proba(X_test)
    cls_metrics = calculate_classification_metrics(y_test_cls, cls_test_preds, cls_test_probs, labels=ATTENDANCE_BANDS)

    # 4. Feature Importance
    feat_importance_df = extract_feature_importance_table(
        best_reg_model,
        best_reg_model.named_steps["preprocessor"]
    )

    # 5. K-Means Profile Clustering
    cluster_cols = ["Lecture_Number", "Attendance_Pct", "Prev_Lecture_Pct"]
    scaler = StandardScaler()
    X_clust = scaler.fit_transform(df[cluster_cols])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clust_labels = kmeans.fit_predict(X_clust)
    sil_score = float(silhouette_score(X_clust, clust_labels))

    # Baseline medians for explainability
    baseline_medians = {col: float(df[col].median()) for col in NUMERICAL_FEATURES if col in df.columns}

    # 6. Deployment Packaging
    deployment_package = {
        "version": "2.5.0",
        "product_name": "AKASH AttendIQ",
        "created_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "regression_model": best_reg_model,
        "classification_model": best_cls_model,
        "band_labels": ATTENDANCE_BANDS,
        "model_features": ALL_MODEL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "baseline_medians": baseline_medians,
        "residual_std": reg_metrics["Residual_Std"],
        "metadata": {
            "best_regression_model": best_reg_name,
            "best_classification_model": best_cls_name,
            "total_dataset_rows": int(len(df)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "regression_metrics": reg_metrics,
            "classification_metrics": cls_metrics,
            "regression_leaderboard": df_reg.to_dict(orient="records"),
            "classification_leaderboard": df_cls.to_dict(orient="records"),
            "feature_importance": feat_importance_df.head(25).to_dict(orient="records"),
            "confusion_matrix": cls_metrics["Confusion_Matrix"],
            "classification_report": cls_metrics["Classification_Report"],
            "silhouette_score": sil_score
        }
    }

    # Save to models/ and copy to root for backwards compatibility
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(deployment_package, output_path)
    joblib.dump(deployment_package, ROOT_MODEL_FILE)

    return deployment_package, df_reg, df_cls
