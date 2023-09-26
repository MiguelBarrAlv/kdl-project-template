import numpy as np

from azure.manager import MLFlowManager
from configparser import ConfigParser
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


def train_classifiers(
    manager_mlflow: MLFlowManager,
    X_train, 
    X_val, 
    X_test, 
    y_train, 
    y_val, 
    y_test
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
    random_seed = 0
    np.random.seed(random_seed)
 
    with manager_mlflow.start_run(run_name="sklearn_example_train"):

        # Define a number of classifiers
        models = create_classifiers()

        # Iterate fitting and validation through all model types, logging results to MLflow:
        for model_name, model in models.items():

            with manager_mlflow.start_run(run_name=model_name, nested=True):
                print(f"Fitting {model_name}...")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                val_accuracy = accuracy_score(y_pred, y_val)
                cm = confusion_matrix(y_val, y_pred)
      

                # Save MLFLow model
                manager_mlflow.log_sklearn_model(model, "model")

                # Register Azure ML Studio model
                description = f"Model {model_name} trained on cancer data"
                manager_mlflow.register_model(
                    model_name=model_name,
                    description=description,
                    tags={"classifier": model_name, "random_seed": random_seed},
                )

                manager_mlflow.log_params({"classifier": random_seed})
                manager_mlflow.log_metrics({"val_acc": val_accuracy})
                
