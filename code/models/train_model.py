# code/models/train_model.py
import os
from pathlib import Path
import json
import logging
from logging.handlers import RotatingFileHandler

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt
import joblib

# --------- Paths ---------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"
LOGS_DIR = REPO_ROOT / "logs"
MLRUNS_DIR = REPO_ROOT / "mlruns"
REPORTS_DIR = REPO_ROOT / "reports"

X_TRAIN = DATA_PROCESSED / "X_train.csv"
Y_TRAIN = DATA_PROCESSED / "y_train.csv"
X_TEST  = DATA_PROCESSED / "X_test.csv"
Y_TEST  = DATA_PROCESSED / "y_test.csv"

# --------- Logging setup ---------
def setup_logging():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "train_model.log"
    handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(fmt)

    logger = logging.getLogger("train_model")
    logger.setLevel(logging.INFO)
    # avoid duplicate handlers in Airflow workers
    if not logger.handlers:
        logger.addHandler(handler)
        stream = logging.StreamHandler()
        stream.setFormatter(fmt)
        logger.addHandler(stream)
    return logger

logger = setup_logging()

# --------- Utils ---------
def ensure_dirs():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    X_train = pd.read_csv(X_TRAIN)
    y_train = pd.read_csv(Y_TRAIN).iloc[:, 0]
    X_test  = pd.read_csv(X_TEST)
    y_test  = pd.read_csv(Y_TEST).iloc[:, 0]
    return X_train, y_train, X_test, y_test

def build_pipeline(n_estimators=300, max_depth=None, random_state=42, class_weight=None):
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        class_weight=class_weight,
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", rf),
    ])

def plot_and_save_confusion_matrix(y_true, y_pred, out_path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=sorted(np.unique(y_true)))
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(cm)))
    ax.set_yticks(range(len(cm)))
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return cm

def main():
    ensure_dirs()

    # MLflow local file store. Без устаревших параметров.
    mlflow.set_tracking_uri(f"file:{MLRUNS_DIR.as_posix()}")
    mlflow.set_experiment("wine_random_forest")

    X_train, y_train, X_test, y_test = load_data()
    feature_names = list(X_train.columns)

    params = {
        "n_estimators": 300,
        "max_depth": None,
        "random_state": 42,
        "class_weight": None,
    }

    with mlflow.start_run(run_name="rf_baseline") as run:
        logger.info("Start run_id=%s", run.info.run_id)

        model = build_pipeline(**params)

        # CV on train
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_acc = cross_val_score(model, X_train, y_train, scoring="accuracy", cv=cv, n_jobs=-1)
        mlflow.log_metric("cv_accuracy_mean", float(cv_acc.mean()))
        mlflow.log_metric("cv_accuracy_std", float(cv_acc.std()))
        logger.info("CV acc mean=%.5f std=%.5f", cv_acc.mean(), cv_acc.std())

        # Fit
        model.fit(X_train, y_train)

        # Test metrics
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        }
        mlflow.log_metrics(metrics)
        logger.info("Test metrics: %s", metrics)

        # Reports
        report = classification_report(y_test, y_pred, output_dict=True)
        report_path = REPORTS_DIR / "classification_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(report_path.as_posix(), artifact_path="reports")

        cm_path = REPORTS_DIR / "confusion_matrix.png"
        cm = plot_and_save_confusion_matrix(y_test, y_pred, cm_path)
        mlflow.log_artifact(cm_path.as_posix(), artifact_path="figures")
        logger.info("Confusion matrix:\n%s", cm)

        # Params
        mlflow.log_params(params)

        # Signature on float copy to avoid integer-NaN warning
        X_sig = X_train.astype("float64")
        signature = infer_signature(X_sig, model.predict(X_train))
        input_example = X_sig.head(3)

        # Log model using sklearn flavor (нет предупреждения про artifact_path)
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=input_example,
        )

        # --------- Packaging for API/Streamlit ---------
        # 1) Модель
        model_path = MODELS_DIR / "wine_model.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path.as_posix(), artifact_path="exported")

        # 2) Имена фич отдельно
        feat_path = MODELS_DIR / "feature_names.pkl"
        joblib.dump(feature_names, feat_path)
        mlflow.log_artifact(feat_path.as_posix(), artifact_path="exported")

        logger.info("Saved model: %s", model_path)
        logger.info("Saved feature names: %s", feat_path)
        logger.info("MLflow artifact store: %s", MLRUNS_DIR)

        # дубль метрик в логи JSON
        metrics_path = LOGS_DIR / "last_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Wrote metrics JSON: %s", metrics_path)

        return metrics

if __name__ == "__main__":
    main()
