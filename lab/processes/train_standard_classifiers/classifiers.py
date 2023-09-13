"""
Functions for instantiating and training traditional ML classifiers
"""
import numpy as np
import os

from configparser import ConfigParser
from dotenv import load_dotenv
from lab.processes.prepare_data.cancer_data import load_data_splits
from lab.processes.aws.sagemaker import deploy_model_to_sagemaker
from lib.viz import plot_confusion_matrix
from mock import MagicMock
from pathlib import Path
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from types import ModuleType
from typing import Union


def create_classifiers() -> dict:
    """
    Instantiates a number of different sklearn classifiers and returns them in a dictionary.

    Returns:
        (dict of sklearn model objects)
    """

    models = {
        "Logistic regression": LogisticRegression(),
        # "Naive Bayes": GaussianNB(),
        # "K-nearest neighbour": KNeighborsClassifier(),
        # "Random forest": RandomForestClassifier(),
        # "Linear SVM": SVC(kernel="linear"),
        # "GradientBoost": GradientBoostingClassifier(),
        # "AdaBoost": AdaBoostClassifier(),
    }
    return models


def train_classifiers(
    mlflow: Union[ModuleType, MagicMock],
    config: Union[ConfigParser, dict],
    mlflow_url: str,
    mlflow_tags: dict,
) -> None:
    """
    Trains a number of classifiers on the data that is found in the directory specified as dir_processed in config.

    Arguments:
        mlflow {Union[ModuleType, MagicMock]} --  MLflow module or its mock replacement
        config {Union[ConfigParser, dict]} -- configuration for the training, with the required sections:
            - "training": containing "random_seed";
            - "paths": containing "artifacts_temp" and "dir_processed";
            - "mlflow": containing "mlflow_experiment"
        mlflow_url {str} -- MLflow URL (empty if replacing mlflow with a mock)
        mlflow_tags {dict} -- MLflow tags (empty if replacing mlflow with a mock)
    """
    load_dotenv()

    # Unpack config:
    random_seed = int(config["training"]["random_seed"])
    # workspace_dir = Path(config["paths"]["workspace_dir"])
    dir_processed = config["paths"]["dir_processed"]
    dir_artifacts = Path(config["paths"]["artifacts_temp"])
    #full_dir_artifacts = workspace_dir / dir_artifacts
    full_dir_artifacts = dir_artifacts # NOTE: Modified for local testing
    filepath_conf_matrix = full_dir_artifacts / "confusion_matrix.png"
    mlflow_experiment = config["mlflow"]["mlflow_experiment"]
    sagemaker_image_uri = os.getenv("AWS_SAGEMAKER_IMAGE_URI")
    aws_sagemaker_role_arn = os.getenv("AWS_SAGEMAKER_ROLE_ARN")
    aws_region_name = os.getenv("AWS_REGION_NAME")

    # Prepare before run
    np.random.seed(random_seed)
    full_dir_artifacts.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(mlflow_url)
    mlflow.set_experiment(mlflow_experiment)

    with mlflow.start_run(run_name="sklearn_example_train", tags=mlflow_tags):

        # Load training and validation data # NOTE: Read from S3
        X_train, X_val, _, y_train, y_val, _ = load_data_splits(
            dir_processed=dir_processed, as_type="array"
        )

        # Define a number of classifiers
        models = create_classifiers()

        # Iterate fitting and validation through all model types, logging results to MLflow:
    for model_name, model in models.items():

        with mlflow.start_run(run_name=model_name, nested=True, tags=mlflow_tags) as run:

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

            try:
                mlflow.sklearn.log_model(model, "model")
                print(f"Model {model_name} saved successfully.")

            except Exception as e:
                print(f"Error saving model {model_name}: {e}")

            mlflow.log_artifacts(full_dir_artifacts)
            mlflow.log_params({"classifier": model_name})
            mlflow.log_metrics({"val_acc": val_accuracy})
            
            # Get the run_id of the subrun
            current_run_id = run.info.run_id
            current_experiment_id = run.info.experiment_id
            current_run_name = run.info.run_name

            # Build the model uri for SageMaker
            model_uri = f'runs:/{current_run_id}/model'
            deploy_model_to_sagemaker(
                model_uri, 
                "mlflow-pyfunc",
                sagemaker_image_uri, 
                aws_region_name, 
                aws_sagemaker_role_arn)