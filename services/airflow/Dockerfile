FROM apache/airflow:2.10.2-python3.11

# ваши зависимости пайплайна
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# docker-cli для BashOperator 'docker compose'
USER root
RUN apt-get update && apt-get install -y docker.io && rm -rf /var/lib/apt/lists/*
USER airflow


