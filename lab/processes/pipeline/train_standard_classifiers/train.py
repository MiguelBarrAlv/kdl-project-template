import argparse
import numpy as np
import asyncio

from io import BytesIO
from azureml.core import Run
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from manager import MLFlowManager, read_npy_from_blob
from train_classifiers import train_classifiers

def main():

    account_name = "pockldtemplate5378781998"
    account_key = ""
    container_name = "cancer-data"

    blob_service_client = BlobServiceClient(account_url=f"https://{account_name}.blob.core.windows.net", credential=account_key)

    def read_npy_from_blob(blob_name):
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        blob_data = blob_client.download_blob()
        blob_content = blob_data.readall()
        blob_io = BytesIO(blob_content)
        return np.load(blob_io, allow_pickle=True)

    parser = argparse.ArgumentParser(description="Process inputs for training.")

    parser.add_argument("--X_train_data", type=str, required=True)
    parser.add_argument("--X_val_data", type=str, required=True)
    parser.add_argument("--X_test_data", type=str, required=True)
    parser.add_argument("--y_train_data", type=str, required=True)
    parser.add_argument("--y_val_data", type=str, required=True)
    parser.add_argument("--y_test_data", type=str, required=True)

    args = parser.parse_args()

    X_train = read_npy_from_blob("X_train.npy")
    X_val = read_npy_from_blob("X_val.npy")
    X_test = read_npy_from_blob("X_test.npy")
    y_train = read_npy_from_blob("y_train.npy")
    y_val = read_npy_from_blob("y_val.npy")
    y_test = read_npy_from_blob("y_test.npy")


    manager_mlflow = MLFlowManager()
    train_classifiers(manager_mlflow, X_train, X_val, X_test, y_train, y_val, y_test)

if __name__ == "__main__":
    main()



