
from pathlib import Path
from datetime import timedelta
import subprocess
import pendulum

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import PythonOperator

from data_processing import load_and_clean_data


PROCESSED_DIR = Path("/opt/airflow/data/processed")
TRAIN_SCRIPT = "/opt/airflow/code/models/train_model.py"         
COMPOSE_FILE = "/opt/airflow/code/deployment/docker-compose.yml"  

REQUIRED_FILES = [
    PROCESSED_DIR / "X_train.csv",
    PROCESSED_DIR / "y_train.csv",
    PROCESSED_DIR / "X_test.csv",
    PROCESSED_DIR / "y_test.csv",
]

def ensure_processed_data_available():
    missing = [str(p) for p in REQUIRED_FILES if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Нет нужных файлов Stage1: {missing}")
    return True

def run_training():
    subprocess.run(["python", TRAIN_SCRIPT], check=True) 

DEPLOY_CMD = f"docker compose -f {COMPOSE_FILE} up -d --build"
CLEANUP_CMD = f"docker compose -f {COMPOSE_FILE} down --remove-orphans"  


with DAG(
    dag_id="wine_pipeline",
    description="Stage1 -> Stage2 -> Stage3",
    schedule="*/5 * * * *",
    start_date=pendulum.datetime(2025, 9, 21, 0, 5, tz="UTC"),
    #start_date=pendulum.now("UTC").subtract(days=1),
    catchup=False,
    tags=["wine", "pipeline"],
) as dag:
    stage1 = PythonOperator(task_id="stage1_data_processing", python_callable=load_and_clean_data)
    wait_files = PythonOperator(task_id="wait_processed_files", python_callable=ensure_processed_data_available,
                                retries=10, retry_delay=timedelta(seconds=30))
    stage2 = PythonOperator(task_id="stage2_train_model", python_callable=run_training)
    cleanup = BashOperator(
        task_id="stage3_cleanup_containers",
        bash_command=CLEANUP_CMD,
        env={"COMPOSE_PROJECT_NAME": "wine"},
        retries=3,
        retry_delay=timedelta(minutes=1)
    )

    stage3 = BashOperator(task_id="stage3_deploy_compose", bash_command=DEPLOY_CMD,
                          env={"COMPOSE_PROJECT_NAME": "wine"}, retries=3, retry_delay=timedelta(minutes=1))
    stage1 >> wait_files >> stage2 >> cleanup >> stage3
