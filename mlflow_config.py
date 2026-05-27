# mlflow_config.py
import os
import mlflow

os.environ["MLFLOW_S3_ENDPOINT_URL"] = ""
os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"
os.environ["AWS_ACCESS_KEY_ID"] = "your_key"
os.environ["AWS_SECRET_ACCESS_KEY"] = "your_secret"
mlflow.set_tracking_uri("")

