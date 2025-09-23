# services/airflow/dags/data_processing.py
from pathlib import Path
import pendulum
import pandas as pd
from sklearn.model_selection import train_test_split
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
# Корень репозитория в контейнере
REPO_ROOT = Path("/opt/airflow")
RAW_CSV = REPO_ROOT / "data" / "raw" / "wine.csv"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

def load_and_clean_data(test_size: float = 0.2, random_state: int = 42) -> None:
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Не найден датасет: {RAW_CSV}")

    df = pd.read_csv(RAW_CSV)
    df = df.drop_duplicates().reset_index(drop=True)

    # базовая очистка
    target_col = "Wine"
    cols = [c.strip() for c in df.columns]
    df.columns = cols
    feature_cols = [c for c in cols if c != target_col]

    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "X_train.csv").write_text("")  # гарантируем создание, если прав нет — упадёт явно
    # сохраняем файлы
    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

with DAG(
    dag_id="data_processing",
    description="Wine Classifier - Stage 1",
    schedule="*/5 * * * *",
    start_date=pendulum.datetime(2025, 9, 21, 0, 5, tz="UTC"),
    catchup=False,
    tags=["wine", "stage1"],
) as dag:
    PythonOperator(
        task_id="data_cleaning_task",
        python_callable=load_and_clean_data,
    )
