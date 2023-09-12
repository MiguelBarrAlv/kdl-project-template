import mlflow
import json

from azureml.core import Workspace
from dotenv import load_dotenv
from pathlib import Path

class MLFlowManager:
    
    def __init__(self, experiment_name: str, config_filename='azure.json'): # NOTE: Hardcoded config_path        
        load_dotenv()
        
        current_path = Path(__file__).resolve().parent
        config_path = current_path.parent / 'configs' / config_filename

        if not config_path.exists():
            raise ValueError(f"Azure config file not found: '{config_filename}'")

        with open(config_path) as config_file:
            config = json.load(config_file)
            print("Azure config loaded successfully.")

        self.workspace = Workspace.from_config(config_path)

        mlflow.set_tracking_uri(self.workspace.get_mlflow_tracking_uri()) # NOTE: Added tracking uri to config
        mlflow.set_experiment(experiment_name)

    def start_run(self, *args, **kwargs):
        return mlflow.start_run(*args, **kwargs)

    def log_sklearn_model(self, model, name):
        return mlflow.sklearn.log_model(model, name)

    def register_model(self, model_name: str, description: str, tags: dict):
        model_name = model_name.replace(" ", "_")
        tags["description"] = description
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/model", 
            name=model_name)

    def log_artifacts(self, *args, **kwargs):
        return mlflow.log_artifacts(*args, **kwargs)
    
    def log_params(self, *args, **kwargs):
        return mlflow.log_params(*args, **kwargs)

    def log_metrics(self, *args, **kwargs):
        return mlflow.log_metrics(*args, **kwargs)
