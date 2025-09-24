# services/airflow/dags/train_model.py
from pathlib import Path
import mlflow
import os
import subprocess
import pendulum
from datetime import timedelta
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
REPO_ROOT = Path("/opt/airflow")
TRAIN_SCRIPT = REPO_ROOT / "code" / "models" / "train_model.py"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"

REQUIRED_FILES = [
    DATA_PROCESSED / "X_train.csv",
    DATA_PROCESSED / "y_train.csv",
    DATA_PROCESSED / "X_test.csv",
    DATA_PROCESSED / "y_test.csv",
]

def ensure_processed_data_available():
    missing = [str(p) for p in REQUIRED_FILES if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Нет нужных файлов: {missing}")
    return True

def run_training():
    subprocess.run(
        ["python", "-u", str(TRAIN_SCRIPT)],
        check=True,
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

with DAG(
    dag_id="wine_model_training",
    description="Train RF on processed wine data",
    schedule="*/5 * * * *",
    start_date=pendulum.datetime(2025, 9, 21, 0, 5, tz="UTC"),
    #start_date=pendulum.now("UTC").subtract(days=1),
    catchup=False,
    tags=["wine", "stage2"],
) as dag:
    wait_for_data = PythonOperator(
        task_id="wait_for_processed_datasets",
        python_callable=ensure_processed_data_available,
        retries=12,
        retry_delay=timedelta(minutes=1),
    )
    train_task = PythonOperator(
        task_id="train_random_forest",
        python_callable=run_training,
    )
    wait_for_data >> train_task
