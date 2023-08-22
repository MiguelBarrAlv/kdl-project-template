"""
Functions for instantiating and training traditional ML classifiers
"""
import numpy as np

from configparser import ConfigParser
from lab.processes.azure.train import MLFlowManager
from lab.processes.azure.storage import AzureDatastoreManager
from mock import MagicMock
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from pathlib import Path
from types import ModuleType
from typing import Union
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from lab.processes.prepare_data.cancer_data import load_data_splits
from lib.viz import plot_confusion_matrix


def create_classifiers() -> dict:
    """
    Instantiates a number of different sklearn classifiers and returns them in a dictionary.

    Returns:
        (dict of sklearn model objects)
    """

    models = {
        "Logistic regression": LogisticRegression(),
        "Naive Bayes": GaussianNB(),
        "K-nearest neighbour": KNeighborsClassifier(),
        "Random forest": RandomForestClassifier(),
        "Linear SVM": SVC(kernel="linear"),
        "GradientBoost": GradientBoostingClassifier(),
        "AdaBoost": AdaBoostClassifier(),
    }
    return models


async def train_classifiers(
    manager_mlflow: MLFlowManager,
    manager_blob: AzureDatastoreManager,
    config: ConfigParser,
    mlflow_tags: dict,
) -> None:
    """
    Trains a number of classifiers on the data that is found in the directory specified as dir_processed in config.

    Arguments:
        mlflow {MLFlowManager} --  MLflow module or its mock replacement
        config {Union[ConfigParser, dict]} -- configuration for the training, with the required sections:
            - "training": containing "random_seed";
            - "paths": containing "artifacts_temp" and "dir_processed";
            - "mlflow": containing "mlflow_experiment"
        mlflow_url {str} -- MLflow URL (empty if replacing mlflow with a mock)
        mlflow_tags {dict} -- MLflow tags (empty if replacing mlflow with a mock)
    """
    # Unpack config:
    random_seed = int(config["training"]["random_seed"])
    dir_processed = config["paths"]["dir_processed"]
    dir_artifacts = Path(config["paths"]["artifact_temp"])
    full_dir_artifacts = dir_artifacts # NOTE: Modified for local testing
    filepath_conf_matrix = full_dir_artifacts / "confusion_matrix.png"
    mlflow_experiment = config["mlflow"]["mlflow_experiment"]

    # Prepare before run
    np.random.seed(random_seed)
    full_dir_artifacts.mkdir(exist_ok=True)

    with manager_mlflow.start_run(run_name="sklearn_example_train", tags=mlflow_tags):

        # Load training and validation data from Azure Datastore
        X_train, X_val, X_test, y_train, y_val, y_test = await load_data_splits(manager_blob, "array")

        # Define a number of classifiers
        models = create_classifiers()

        # Iterate fitting and validation through all model types, logging results to MLflow:
        for model_name, model in models.items():

            with manager_mlflow.start_run(run_name=model_name, nested=True, tags=mlflow_tags):
                print(f"Fitting {model_name}...")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                val_accuracy = accuracy_score(y_pred, y_val)
                cm = confusion_matrix(y_val, y_pred)
                plot_confusion_matrix(
                    cm,
                    normalize=False,
                    title="Confusion matrix (validation set)",
                    savepath=filepath_conf_matrix,
                )

                # Save MLFLow model
                manager_mlflow.log_sklearn_model(model, "model")

                # Register Azure ML Studio model
                description = f"Model {model_name} trained on cancer data"
                manager_mlflow.register_model(
                    model_name=model_name,
                    description=description,
                    tags=mlflow_tags
                )

                manager_mlflow.log_artifacts(str(full_dir_artifacts)) # NOTE: Overwrite the artifact every iterate, so we can see the last run
                manager_mlflow.log_params({"classifier": model_name})
                manager_mlflow.log_metrics({"val_acc": val_accuracy})
                print(f"Validation accuracy: {val_accuracy}")
                print("classifier: ", model_name)
                
