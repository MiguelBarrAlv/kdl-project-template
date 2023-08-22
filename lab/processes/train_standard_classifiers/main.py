"""
ML pipeline for breast cancer classification
Part 2: Training traditional ML models
"""
import asyncio
import configparser
import os
import mlflow

from lab.processes.train_standard_classifiers.classifiers import train_classifiers
from lab.processes.azure.train import MLFlowManager
from lab.processes.azure.storage import AzureDatastoreManager

PATH_CONFIG = os.getenv("PATH_CONFIG")
config = configparser.ConfigParser()
config.read(str(PATH_CONFIG))

MLFLOW_URL = os.getenv("MLFLOW_URL")
MLFLOW_TAGS = {"git_tag": "mlflow-azure"} # NOTE: Added git_tag to MLFLOW_TAGS hardcoded

if __name__ == "__main__":

    manager_mlflow = MLFlowManager(experiment_name=config["mlflow"]["mlflow_experiment"])
    manager_blob = AzureDatastoreManager()
    
    asyncio.run(train_classifiers(
        manager_mlflow=manager_mlflow,
        manager_blob=manager_blob,
        config=config,
        mlflow_tags=MLFLOW_TAGS
    ))

