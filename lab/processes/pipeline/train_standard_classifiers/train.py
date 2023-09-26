import asyncio
import argparse
import numpy as np
import os

from azureml.core import Run
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from io import BytesIO
from azure.manager import MLFlowManager, read_npy_from_blob
from train_classifiers import train_classifiers

def read_npy_from_blob(blob_name):
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    blob_data = blob_client.download_blob()
    blob_content = blob_data.readall()
    blob_io = BytesIO(blob_content)
    return np.load(blob_io, allow_pickle=True)


def get_environment_variable(var_name):
    value = os.environ.get(var_name)
    if not value:
        raise ValueError(f"{var_name} environment variable is not set!")
    return value


def read_data_from_blob(filename):
    try:
        return read_npy_from_blob(filename)
    except Exception as e:
        raise ValueError(f"Error reading {filename} from blob: {str(e)}")


def main():
    try:
        ACCOUNT_NAME = get_environment_variable("account_name")
        ACCOUNT_KEY = get_environment_variable("account_key")

        container_name = "cancer-data"
        blob_service_client = BlobServiceClient(account_url=f"https://{ACCOUNT_NAME}.blob.core.windows.net", credential=ACCOUNT_KEY)
        
        parser = argparse.ArgumentParser(description="Process inputs for training.")

        datasets = ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]
        for dataset in datasets:
            parser.add_argument(f"--{dataset}_data", type=str, required=True)

        args = parser.parse_args()
        
        X_train = read_data_from_blob(args.X_train_data)
        X_val = read_data_from_blob(args.X_val_data)
        X_test = read_data_from_blob(args.X_test_data)
        y_train = read_data_from_blob(args.y_train_data)
        y_val = read_data_from_blob(args.y_val_data)
        y_test = read_data_from_blob(args.y_test_data)

        manager_mlflow = MLFlowManager()
        train_classifiers(manager_mlflow, X_train, X_val, X_test, y_train, y_val, y_test)

    except ValueError as ve:
        print(f"Value Error: {ve}")
    except Exception as e:
        print(f"Unexpected Error: {e}")

if __name__ == "__main__":
    main()


