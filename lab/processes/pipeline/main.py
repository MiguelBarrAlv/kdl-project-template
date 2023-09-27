import asyncio
import os

from azureml.pipeline.core import Pipeline
from azure.storage.blob import BlobServiceClient
from azureml.core import Experiment
from commands import load_data_from_blob, train_standard_classifier
from dotenv import load_dotenv
from lab.processes.azure.workspace import AzureWorkspaceConnector

def main():
    load_dotenv()
    print("Account", os.getenv("AZURE_STORAGE_ACCOUNT_NAME"))
    print("Key", os.getenv("AZURE_STORAGE_ACCOUNT_KEY"))
    azure_connector = AzureWorkspaceConnector()
    ws = azure_connector.ws
    compute_target = 'cpu-cluster'

    data_load_step, data = load_data_from_blob(compute_target, ws)
    train_step = train_standard_classifier(compute_target, data)
    pipeline = Pipeline(workspace=ws, steps=[data_load_step, train_step])
    
    print("Pipeline is built with name: breast_cancer_pipeline")
    pipeline.validate()

    print("Pipeline validation complete")
    pipeline_run = Experiment(ws, 'my_pipeline_run').submit(pipeline)

    print("Pipeline is published.")

if __name__ == "__main__":
    main()











