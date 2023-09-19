import os

from azureml.pipeline.steps import PythonScriptStep
from azureml.core import Environment, RunConfiguration
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azureml.pipeline.core import PipelineData

root_dir = os.path.dirname(os.path.abspath(__file__))  # obtiene la ruta del directorio del script actual
env_path = os.path.join(root_dir, "enviroment", "environment.yaml")

def get_data_upload_step(compute_target):
    data_upload_src_dir = "./prepare_data"
    env = Environment.from_conda_specification(name="data_upload_env", file_path="./prepare_data/environment.yaml")
    run_config = RunConfiguration()
    run_config.environment = env

    blob_name_argument = "--blob_name"

    data_upload_step = PythonScriptStep(name="upload_breast_cancer_data",
                                        script_name="upload_to_blob.py",
                                        arguments=[blob_name_argument, "breast_cancer_data.csv"],
                                        compute_target=compute_target,
                                        source_directory=data_upload_src_dir,
                                        runconfig=run_config,
                                        allow_reuse=True)   
    return data_upload_step


def load_data_from_blob(compute_target, ws):
    data_load_src_dir = "./load_data"
    env = Environment.from_conda_specification(name="data_load_env", file_path=env_path)
    run_config = RunConfiguration()
    run_config.environment = env
    

    X_train_data = PipelineData("X_train_data", datastore=ws.get_default_datastore())
    X_val_data = PipelineData("X_val_data", datastore=ws.get_default_datastore())
    X_test_data = PipelineData("X_test_data", datastore=ws.get_default_datastore())
    y_train_data = PipelineData("y_train_data", datastore=ws.get_default_datastore())
    y_val_data = PipelineData("y_val_data", datastore=ws.get_default_datastore())
    y_test_data = PipelineData("y_test_data", datastore=ws.get_default_datastore())

    step = PythonScriptStep(
        name="load_data_and_split_step",
        script_name="processing_data.py",
        arguments=[
            "--X_train_data", X_train_data,
            "--X_val_data", X_val_data,
            "--X_test_data", X_test_data,
            "--y_train_data", y_train_data,
            "--y_val_data", y_val_data,
            "--y_test_data", y_test_data
        ],
        outputs=[X_train_data, X_val_data, X_test_data, y_train_data, y_val_data, y_test_data],
        compute_target=compute_target,
        source_directory=data_load_src_dir,
        runconfig=run_config,
        allow_reuse=True
    )
    return step, X_train_data, X_val_data, X_test_data, y_train_data, y_val_data, y_test_data


def train_standard_classifier(compute_target, X_train_data, X_val_data, X_test_data, y_train_data, y_val_data, y_test_data):
    env = Environment.from_conda_specification(name="train_env", file_path=env_path)
    run_config = RunConfiguration()
    run_config.environment = env

    step = PythonScriptStep(
        name="train_step",
        script_name="train.py",
        arguments=[
            "--X_train_data", X_train_data,
            "--X_val_data", X_val_data,
            "--X_test_data", X_test_data,
            "--y_train_data", y_train_data,
            "--y_val_data", y_val_data,
            "--y_test_data", y_test_data
        ],
        inputs=[X_train_data, X_val_data, X_test_data, y_train_data, y_val_data, y_test_data],
        compute_target=compute_target,
        source_directory="./train_standard_classifiers",
        runconfig=run_config,
        allow_reuse=True
    )
    return step

