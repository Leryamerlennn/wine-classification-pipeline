# Wine classification pipeline
This project provides an end-to-end ML pipeline for wine quality classification, including data processing, model training, experiment tracking, and deployment artifacts.

## Services

- **Airflow UI** – available at [http://localhost:8080](http://localhost:8080/). The Airflow deployment lives under `services/airflow`.
- **Streamlit application** – exposed at [http://localhost:8501](http://localhost:8501/) when the web app service is running. The application code is in `services/streamlit`

Use the Stage 3 deployment (`deployment/docker-compose.yml`) to build and run the full stack locally, exposing the Airflow UI and Streamlit application on their respective ports.
