import mlflow
import os

from azureml.core import Workspace, Datastore
from dotenv import load_dotenv

class AzureDatastoreManager:
    
    def __init__(self, config_path='../azure.json'):
        load_dotenv() 
        self.ws = Workspace.from_config(config_path)
        self.account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
        self.account_key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
        self.blob_datastore_name = os.getenv("AZURE_STORAGE_BLOB_DATASTORE_NAME")
        self.container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
        self.target_path = os.getenv("AZURE_STORAGE_TARGET_PATH")

    def _container_exists(self):
        try:
            Datastore.get(self.ws, self.blob_datastore_name)
            return True
        except:
            return False

    def upload_data(self, src_dir):
        if not self._container_exists():
            self._create_container()

        try:
            datastore = Datastore.get(self.ws, self.blob_datastore_name)
            datastore.upload(src_dir=src_dir, target_path=self.target_path)
            print("Upload successfully completed.")
            print(f"Data uploaded to: {self.blob_datastore_name}/{self.container_name}")
        except Exception as e:
            print("Error uploading data: ", e)

    def _create_container(self):
        try:
            Datastore.register_azure_blob_container(
                workspace=self.ws, 
                datastore_name=self.blob_datastore_name, 
                container_name=self.container_name, 
                account_name=self.account_name,
                account_key=self.account_key
            )
            print(f"Container '{self.container_name}' created successfully.")
        except Exception as e:
            print("Error creating container ", e)
            raise


class MLFlowManager:
    def __init__(self, experiment_name: str, config_file = '../azure.json'):
        self.workspace = Workspace.from_config(config_file)
        mlflow.set_tracking_uri(self.workspace.get_mlflow_tracking_uri()) #NOTE: Added trackir uri to config
        mlflow.set_experiment(experiment_name)

    def start_run(self, *args, **kwargs):
        return mlflow.start_run(*args, **kwargs)

    def log_sklearn_model(self, model, name):
        return mlflow.sklearn.log_model(model, name)

    def register_model(self, model_name: str, description: str, tags: dict):
        model_name = model_name.replace(" ", "_")
        tags["description"] = description
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        

    
        print(f"Registering model '{model_name}' with tags: {tags} and uri: {model_uri}")


        mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/model", 
            name=model_name)

    def log_artifacts(self, *args, **kwargs):
        print("Logging artifacts", args, kwargs)
        return mlflow.log_artifacts(*args, **kwargs)
    
    def log_params(self, *args, **kwargs):
        return mlflow.log_params(*args, **kwargs)

    def log_metrics(self, *args, **kwargs):
        return mlflow.log_metrics(*args, **kwargs)
    