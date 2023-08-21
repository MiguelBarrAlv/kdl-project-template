"""
ML pipeline for breast cancer classification
Part 2: Training traditional ML models
"""


import configparser
import os
import mlflow

from lab.processes.train_standard_classifiers.classifiers import train_classifiers
from lab.processes.azure.services import MLFlowManager

PATH_CONFIG = os.getenv("PATH_CONFIG")
config = configparser.ConfigParser()
config.read(str(PATH_CONFIG))

MLFLOW_URL = os.getenv("MLFLOW_URL")
# MLFLOW_TAGS = {"git_tag": os.getenv("DRONE_TAG")}
MLFLOW_TAGS = {"git_tag": "mlflow-azure"}

if __name__ == "__main__":

    # train_classifiers(
    #     mlflow=mlflow, config=config, mlflow_url=MLFLOW_URL, mlflow_tags=MLFLOW_TAGS
    # )

    manager = MLFlowManager(
        experiment_name=config["mlflow"]["mlflow_experiment"]
        )

    train_classifiers(
        manager=manager,
        config=config,
        mlflow_tags=MLFLOW_TAGS
    )

